// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package segments

/*
#cgo pkg-config: milvus_segcore

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"
*/
import "C"

import (
	"context"
	"fmt"
	"sync"

	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"

	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/pkg/eventlog"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/cache"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	. "github.com/milvus-io/milvus/pkg/util/typeutil"
)

type SegmentFilter func(segment Segment) bool

func WithSkipEmpty() SegmentFilter {
	return func(segment Segment) bool {
		return segment.InsertCount() > 0
	}
}

func WithPartition(partitionID UniqueID) SegmentFilter {
	return func(segment Segment) bool {
		return segment.Partition() == partitionID
	}
}

func WithChannel(channel string) SegmentFilter {
	return func(segment Segment) bool {
		return segment.Shard() == channel
	}
}

func WithType(typ SegmentType) SegmentFilter {
	return func(segment Segment) bool {
		return segment.Type() == typ
	}
}

func WithID(id int64) SegmentFilter {
	return func(segment Segment) bool {
		return segment.ID() == id
	}
}

func WithLevel(level datapb.SegmentLevel) SegmentFilter {
	return func(segment Segment) bool {
		return segment.Level() == level
	}
}

type SegmentAction func(segment Segment) bool

func IncreaseVersion(version int64) SegmentAction {
	return func(segment Segment) bool {
		log := log.Ctx(context.Background()).With(
			zap.Int64("segmentID", segment.ID()),
			zap.String("type", segment.Type().String()),
			zap.Int64("segmentVersion", segment.Version()),
			zap.Int64("updateVersion", version),
		)
		for oldVersion := segment.Version(); oldVersion < version; {
			if segment.CASVersion(oldVersion, version) {
				return true
			}
		}
		log.Warn("segment version cannot go backwards, skip update")
		return false
	}
}

type actionType int32

const (
	removeAction actionType = iota
	addAction
)

type Manager struct {
	Collection CollectionManager
	Segment    SegmentManager
	DiskCache  cache.Cache[int64, Segment]
}

func NewManager() *Manager {
	diskCap := paramtable.Get().QueryNodeCfg.DiskCapacityLimit.GetAsInt64()
	segmentMaxSIze := paramtable.Get().DataCoordCfg.SegmentMaxSize.GetAsInt64()
	cacheMaxItemNum := diskCap / segmentMaxSIze

	segMgr := NewSegmentManager()
	sf := singleflight.Group{}
	return &Manager{
		Collection: NewCollectionManager(),
		Segment:    segMgr,
		DiskCache: cache.NewLRUCache[int64, Segment](
			int32(cacheMaxItemNum),
			func(key int64) (Segment, bool) {
				log.Debug("cache missed segment", zap.Int64("segmentID", key))
				segMgr.mu.RLock()
				defer segMgr.mu.RUnlock()

				segment, ok := segMgr.sealedSegments[key]
				if !ok {
					// the segment has been released, just ignore it
					return nil, false
				}

				info := segment.LoadInfo()
				_, err, _ := sf.Do(fmt.Sprint(segment.ID()), func() (interface{}, error) {
					err := loadSealedSegmentFields(context.Background(), segment.(*LocalSegment), info.BinlogPaths, info.GetNumOfRows(), WithLoadStatus(LoadStatusMapped))
					return nil, err
				})
				if err != nil {
					log.Warn("cache sealed segment failed", zap.Error(err))
					return nil, false
				}
				return segment, true
			},
			func(key int64, segment Segment) {
				log.Debug("evict segment from cache", zap.Int64("segmentID", key))
				segment.Release(WithReleaseScope(ReleaseScopeData))
			}),
	}
}

type SegmentManager interface {
	// Put puts the given segments in,
	// and increases the ref count of the corresponding collection,
	// dup segments will not increase the ref count
	Put(segmentType SegmentType, segments ...Segment)
	UpdateBy(action SegmentAction, filters ...SegmentFilter) int
	Get(segmentID UniqueID) Segment
	GetWithType(segmentID UniqueID, typ SegmentType) Segment
	GetBy(filters ...SegmentFilter) []Segment
	// Get segments and acquire the read locks
	GetAndPinBy(filters ...SegmentFilter) ([]Segment, error)
	GetAndPin(segments []int64, filters ...SegmentFilter) ([]Segment, error)
	Unpin(segments []Segment)

	GetSealed(segmentID UniqueID) Segment
	GetGrowing(segmentID UniqueID) Segment
	Empty() bool

	// Remove removes the given segment,
	// and decreases the ref count of the corresponding collection,
	// will not decrease the ref count if the given segment not exists
	Remove(segmentID UniqueID, scope querypb.DataScope) (int, int)
	RemoveBy(filters ...SegmentFilter) (int, int)
	Clear()

	// Deprecated: quick fix critical issue: #30857
	// TODO: All Segment assigned to querynode should be managed by SegmentManager, including loading or releasing to perform a transaction.
	Exist(segmentID UniqueID, typ SegmentType) bool
}

var _ SegmentManager = (*segmentManager)(nil)

// Manager manages all collections and segments
type segmentManager struct {
	mu sync.RWMutex // guards all

	growingSegments map[UniqueID]Segment
	sealedSegments  map[UniqueID]Segment

	growingOnReleasingSegments map[UniqueID]struct{}
	sealedOnReleasingSegments  map[UniqueID]struct{}
}

func NewSegmentManager() *segmentManager {
	mgr := &segmentManager{
		growingSegments:            make(map[int64]Segment),
		sealedSegments:             make(map[int64]Segment),
		growingOnReleasingSegments: make(map[int64]struct{}),
		sealedOnReleasingSegments:  make(map[int64]struct{}),
	}
	return mgr
}

func (mgr *segmentManager) Put(segmentType SegmentType, segments ...Segment) {
	var replacedSegment []Segment
	mgr.mu.Lock()
	defer mgr.mu.Unlock()
	var targetMap map[int64]Segment
	switch segmentType {
	case SegmentTypeGrowing:
		targetMap = mgr.growingSegments
	case SegmentTypeSealed:
		targetMap = mgr.sealedSegments
	default:
		panic("unexpected segment type")
	}

	for _, segment := range segments {
		oldSegment, ok := targetMap[segment.ID()]

		if ok {
			if oldSegment.Version() >= segment.Version() {
				log.Warn("Invalid segment distribution changed, skip it",
					zap.Int64("segmentID", segment.ID()),
					zap.Int64("oldVersion", oldSegment.Version()),
					zap.Int64("newVersion", segment.Version()),
				)
				// delete redundant segment
				segment.Release()
				continue
			}
			replacedSegment = append(replacedSegment, oldSegment)
		}
		targetMap[segment.ID()] = segment

		eventlog.Record(eventlog.NewRawEvt(eventlog.Level_Info, fmt.Sprintf("Segment %d[%d] loaded", segment.ID(), segment.Collection())))
		metrics.QueryNodeNumSegments.WithLabelValues(
			fmt.Sprint(paramtable.GetNodeID()),
			fmt.Sprint(segment.Collection()),
			fmt.Sprint(segment.Partition()),
			segment.Type().String(),
			fmt.Sprint(len(segment.Indexes())),
			segment.Level().String(),
		).Inc()
	}
	mgr.updateMetric()

	// release replaced segment
	if len(replacedSegment) > 0 {
		go func() {
			for _, segment := range replacedSegment {
				mgr.release(segment)
			}
		}()
	}
}

func (mgr *segmentManager) UpdateBy(action SegmentAction, filters ...SegmentFilter) int {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	updated := 0
	for _, segment := range mgr.growingSegments {
		if filter(segment, filters...) {
			if action(segment) {
				updated++
			}
		}
	}

	for _, segment := range mgr.sealedSegments {
		if filter(segment, filters...) {
			if action(segment) {
				updated++
			}
		}
	}
	return updated
}

// Deprecated:
// TODO: All Segment assigned to querynode should be managed by SegmentManager, including loading or releasing to perform a transaction.
func (mgr *segmentManager) Exist(segmentID UniqueID, typ SegmentType) bool {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()
	switch typ {
	case SegmentTypeGrowing:
		if _, ok := mgr.growingSegments[segmentID]; ok {
			return true
		} else if _, ok = mgr.growingOnReleasingSegments[segmentID]; ok {
			return true
		}
	case SegmentTypeSealed:
		if _, ok := mgr.sealedSegments[segmentID]; ok {
			return true
		} else if _, ok = mgr.sealedOnReleasingSegments[segmentID]; ok {
			return true
		}
	}

	return false
}

func (mgr *segmentManager) Get(segmentID UniqueID) Segment {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	if segment, ok := mgr.growingSegments[segmentID]; ok {
		return segment
	} else if segment, ok = mgr.sealedSegments[segmentID]; ok {
		return segment
	}

	return nil
}

func (mgr *segmentManager) GetWithType(segmentID UniqueID, typ SegmentType) Segment {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	switch typ {
	case SegmentTypeSealed:
		return mgr.sealedSegments[segmentID]
	case SegmentTypeGrowing:
		return mgr.growingSegments[segmentID]
	default:
		return nil
	}
}

func (mgr *segmentManager) GetBy(filters ...SegmentFilter) []Segment {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	ret := make([]Segment, 0)
	for _, segment := range mgr.growingSegments {
		if filter(segment, filters...) {
			ret = append(ret, segment)
		}
	}

	for _, segment := range mgr.sealedSegments {
		if filter(segment, filters...) {
			ret = append(ret, segment)
		}
	}
	return ret
}

func (mgr *segmentManager) GetAndPinBy(filters ...SegmentFilter) ([]Segment, error) {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	ret := make([]Segment, 0)
	var err error
	defer func() {
		if err != nil {
			for _, segment := range ret {
				segment.RUnlock()
			}
		}
	}()

	for _, segment := range mgr.growingSegments {
		if filter(segment, filters...) {
			err = segment.RLock()
			if err != nil {
				return nil, err
			}
			ret = append(ret, segment)
		}
	}

	for _, segment := range mgr.sealedSegments {
		if segment.Level() != datapb.SegmentLevel_L0 && filter(segment, filters...) {
			err = segment.RLock()
			if err != nil {
				return nil, err
			}
			ret = append(ret, segment)
		}
	}
	return ret, nil
}

func (mgr *segmentManager) GetAndPin(segments []int64, filters ...SegmentFilter) ([]Segment, error) {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	lockedSegments := make([]Segment, 0, len(segments))
	var err error
	defer func() {
		if err != nil {
			for _, segment := range lockedSegments {
				segment.RUnlock()
			}
		}
	}()

	for _, id := range segments {
		growing, growingExist := mgr.growingSegments[id]
		sealed, sealedExist := mgr.sealedSegments[id]

		// L0 Segment should not be queryable.
		if sealedExist && sealed.Level() == datapb.SegmentLevel_L0 {
			continue
		}

		growingExist = growingExist && filter(growing, filters...)
		sealedExist = sealedExist && filter(sealed, filters...)

		if growingExist {
			err = growing.RLock()
			if err != nil {
				return nil, err
			}
			lockedSegments = append(lockedSegments, growing)
		}
		if sealedExist {
			err = sealed.RLock()
			if err != nil {
				return nil, err
			}
			lockedSegments = append(lockedSegments, sealed)
		}

		if !growingExist && !sealedExist {
			err = merr.WrapErrSegmentNotLoaded(id, "segment not found")
			return nil, err
		}
	}

	return lockedSegments, nil
}

func (mgr *segmentManager) Unpin(segments []Segment) {
	for _, segment := range segments {
		segment.RUnlock()
	}
}

func filter(segment Segment, filters ...SegmentFilter) bool {
	for _, filter := range filters {
		if !filter(segment) {
			return false
		}
	}
	return true
}

func (mgr *segmentManager) GetSealed(segmentID UniqueID) Segment {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	if segment, ok := mgr.sealedSegments[segmentID]; ok {
		return segment
	}

	return nil
}

func (mgr *segmentManager) GetGrowing(segmentID UniqueID) Segment {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	if segment, ok := mgr.growingSegments[segmentID]; ok {
		return segment
	}

	return nil
}

func (mgr *segmentManager) Empty() bool {
	mgr.mu.RLock()
	defer mgr.mu.RUnlock()

	return len(mgr.growingSegments)+len(mgr.sealedSegments) == 0
}

// returns true if the segment exists,
// false otherwise
func (mgr *segmentManager) Remove(segmentID UniqueID, scope querypb.DataScope) (int, int) {
	mgr.mu.Lock()

	var removeGrowing, removeSealed int
	var growing, sealed Segment
	switch scope {
	case querypb.DataScope_Streaming:
		growing = mgr.removeSegmentWithType(SegmentTypeGrowing, segmentID)
		if growing != nil {
			removeGrowing = 1
		}

	case querypb.DataScope_Historical:
		sealed = mgr.removeSegmentWithType(SegmentTypeSealed, segmentID)
		if sealed != nil {
			removeSealed = 1
		}

	case querypb.DataScope_All:
		growing = mgr.removeSegmentWithType(SegmentTypeGrowing, segmentID)
		if growing != nil {
			removeGrowing = 1
		}

		sealed = mgr.removeSegmentWithType(SegmentTypeSealed, segmentID)
		if sealed != nil {
			removeSealed = 1
		}
	}
	mgr.updateMetric()
	mgr.mu.Unlock()

	if growing != nil {
		mgr.release(growing)
	}

	if sealed != nil {
		mgr.release(sealed)
	}

	return removeGrowing, removeSealed
}

func (mgr *segmentManager) removeSegmentWithType(typ SegmentType, segmentID UniqueID) Segment {
	switch typ {
	case SegmentTypeGrowing:
		s, ok := mgr.growingSegments[segmentID]
		if ok {
			delete(mgr.growingSegments, segmentID)
			mgr.growingOnReleasingSegments[segmentID] = struct{}{}
			return s
		}

	case SegmentTypeSealed:
		s, ok := mgr.sealedSegments[segmentID]
		if ok {
			delete(mgr.sealedSegments, segmentID)
			mgr.sealedOnReleasingSegments[segmentID] = struct{}{}
			return s
		}
	default:
		return nil
	}

	return nil
}

func (mgr *segmentManager) RemoveBy(filters ...SegmentFilter) (int, int) {
	mgr.mu.Lock()

	var removeGrowing, removeSealed []Segment
	for id, segment := range mgr.growingSegments {
		if filter(segment, filters...) {
			s := mgr.removeSegmentWithType(SegmentTypeGrowing, id)
			if s != nil {
				removeGrowing = append(removeGrowing, s)
			}
		}
	}

	for id, segment := range mgr.sealedSegments {
		if filter(segment, filters...) {
			s := mgr.removeSegmentWithType(SegmentTypeSealed, id)
			if s != nil {
				removeSealed = append(removeSealed, s)
			}
		}
	}
	mgr.updateMetric()
	mgr.mu.Unlock()

	for _, s := range removeGrowing {
		mgr.release(s)
	}

	for _, s := range removeSealed {
		mgr.release(s)
	}

	return len(removeGrowing), len(removeSealed)
}

func (mgr *segmentManager) Clear() {
	mgr.mu.Lock()

	for id := range mgr.growingSegments {
		mgr.growingOnReleasingSegments[id] = struct{}{}
	}
	growingWaitForRelease := mgr.growingSegments
	mgr.growingSegments = make(map[int64]Segment)

	for id := range mgr.sealedSegments {
		mgr.sealedOnReleasingSegments[id] = struct{}{}
	}
	sealedWaitForRelease := mgr.sealedSegments
	mgr.sealedSegments = make(map[int64]Segment)
	mgr.updateMetric()

	mgr.mu.Unlock()

	for _, segment := range growingWaitForRelease {
		mgr.release(segment)
	}
	for _, segment := range sealedWaitForRelease {
		mgr.release(segment)
	}
}

func (mgr *segmentManager) updateMetric() {
	// update collection and partiation metric
	collections, partiations := make(Set[int64]), make(Set[int64])
	for _, seg := range mgr.growingSegments {
		collections.Insert(seg.Collection())
		partiations.Insert(seg.Partition())
	}
	for _, seg := range mgr.sealedSegments {
		collections.Insert(seg.Collection())
		partiations.Insert(seg.Partition())
	}
	metrics.QueryNodeNumCollections.WithLabelValues(fmt.Sprint(paramtable.GetNodeID())).Set(float64(collections.Len()))
	metrics.QueryNodeNumPartitions.WithLabelValues(fmt.Sprint(paramtable.GetNodeID())).Set(float64(partiations.Len()))
}

func (mgr *segmentManager) release(segment Segment) {
	segment.Release()

	metrics.QueryNodeNumSegments.WithLabelValues(
		fmt.Sprint(paramtable.GetNodeID()),
		fmt.Sprint(segment.Collection()),
		fmt.Sprint(segment.Partition()),
		segment.Type().String(),
		fmt.Sprint(len(segment.Indexes())),
		segment.Level().String(),
	).Dec()

	mgr.mu.Lock()
	defer mgr.mu.Unlock()

	switch segment.Type() {
	case SegmentTypeGrowing:
		delete(mgr.growingOnReleasingSegments, segment.ID())
	case SegmentTypeSealed:
		delete(mgr.sealedOnReleasingSegments, segment.ID())
	}
}
