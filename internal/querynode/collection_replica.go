// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package querynode

/*

#cgo CFLAGS: -I${SRCDIR}/../core/output/include

#cgo LDFLAGS: -L${SRCDIR}/../core/output/lib -lmilvus_segcore -Wl,-rpath=${SRCDIR}/../core/output/lib

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"

*/
import "C"
import (
	"errors"
	"fmt"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"strconv"
	"sync"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
)

/*
 * collectionReplica contains a in-memory local copy of persistent collections.
 * In common cases, the system has multiple query nodes. Data of a collection will be
 * distributed across all the available query nodes, and each query node's collectionReplica
 * will maintain its own share (only part of the collection).
 * Every replica tracks a value called tSafe which is the maximum timestamp that the replica
 * is up-to-date.
 */
type ReplicaInterface interface {
	// collection
	getCollectionIDs() []UniqueID
	addCollection(collectionID UniqueID, schema *schemapb.CollectionSchema) error
	removeCollection(collectionID UniqueID) error
	getCollectionByID(collectionID UniqueID) (*Collection, error)
	hasCollection(collectionID UniqueID) bool
	getCollectionNum() int
	getPartitionIDs(collectionID UniqueID) ([]UniqueID, error)
	getVecFieldIDsByCollectionID(collectionID UniqueID) ([]int64, error)

	// partition
	addPartition(collectionID UniqueID, partitionID UniqueID) error
	removePartition(partitionID UniqueID) error
	getPartitionByID(partitionID UniqueID) (*Partition, error)
	hasPartition(partitionID UniqueID) bool
	getPartitionNum() int
	getSegmentIDs(partitionID UniqueID) ([]UniqueID, error)

	// segment
	addSegment(segmentID UniqueID, partitionID UniqueID, collectionID UniqueID, segType segmentType, onService bool) error
	setSegment(segment *Segment) error
	removeSegment(segmentID UniqueID) error
	getSegmentByID(segmentID UniqueID) (*Segment, error)
	hasSegment(segmentID UniqueID) bool
	getSegmentNum() int
	getSegmentStatistics() []*internalpb.SegmentStats

	// excluded segments
	initExcludedSegments(collectionID UniqueID)
	removeExcludedSegments(collectionID UniqueID)
	addExcludedSegments(collectionID UniqueID, segmentInfos []*querypb.ExcludedSegmentInfo) error
	getExcludedSegments(collectionID UniqueID) ([]*querypb.ExcludedSegmentInfo, error)

	getSegmentsBySegmentType(segType segmentType) ([]UniqueID, []UniqueID, []UniqueID)
	replaceGrowingSegmentBySealedSegment(segment *Segment) error

	freeAll()
}

type collectionReplica struct {
	mu          sync.RWMutex // guards all
	collections map[UniqueID]*Collection
	partitions  map[UniqueID]*Partition
	segments    map[UniqueID]*Segment

	excludedSegments map[UniqueID][]*querypb.ExcludedSegmentInfo // map[collectionID]segmentIDs
}

//----------------------------------------------------------------------------------------------------- collection
func (colReplica *collectionReplica) getCollectionIDs() []UniqueID {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	collectionIDs := make([]UniqueID, 0)
	for id := range colReplica.collections {
		collectionIDs = append(collectionIDs, id)
	}
	return collectionIDs
}

func (colReplica *collectionReplica) addCollection(collectionID UniqueID, schema *schemapb.CollectionSchema) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	if ok := colReplica.hasCollectionPrivate(collectionID); ok {
		return errors.New("collection has been loaded, id %d" + strconv.FormatInt(collectionID, 10))
	}

	var newCollection = newCollection(collectionID, schema)
	colReplica.collections[collectionID] = newCollection

	return nil
}

func (colReplica *collectionReplica) removeCollection(collectionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	return colReplica.removeCollectionPrivate(collectionID)
}

func (colReplica *collectionReplica) removeCollectionPrivate(collectionID UniqueID) error {
	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}

	// delete partitions
	for _, partitionID := range collection.partitionIDs {
		// ignore error, try to delete
		_ = colReplica.removePartitionPrivate(partitionID)
	}

	deleteCollection(collection)
	delete(colReplica.collections, collectionID)

	return nil
}

func (colReplica *collectionReplica) getCollectionByID(collectionID UniqueID) (*Collection, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return colReplica.getCollectionByIDPrivate(collectionID)
}

func (colReplica *collectionReplica) getCollectionByIDPrivate(collectionID UniqueID) (*Collection, error) {
	collection, ok := colReplica.collections[collectionID]
	if !ok {
		return nil, errors.New("collection hasn't been loaded or has been released, collection id = %d" + strconv.FormatInt(collectionID, 10))
	}

	return collection, nil
}

func (colReplica *collectionReplica) hasCollection(collectionID UniqueID) bool {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return colReplica.hasCollectionPrivate(collectionID)
}

func (colReplica *collectionReplica) hasCollectionPrivate(collectionID UniqueID) bool {
	_, ok := colReplica.collections[collectionID]
	return ok
}

func (colReplica *collectionReplica) getCollectionNum() int {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return len(colReplica.collections)
}

func (colReplica *collectionReplica) getPartitionIDs(collectionID UniqueID) ([]UniqueID, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	return collection.partitionIDs, nil
}

func (colReplica *collectionReplica) getVecFieldIDsByCollectionID(collectionID UniqueID) ([]int64, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	fields, err := colReplica.getFieldsByCollectionIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	vecFields := make([]int64, 0)
	for _, field := range fields {
		if field.DataType == schemapb.DataType_BinaryVector || field.DataType == schemapb.DataType_FloatVector {
			vecFields = append(vecFields, field.FieldID)
		}
	}

	if len(vecFields) <= 0 {
		return nil, errors.New("no vector field in collection %d" + strconv.FormatInt(collectionID, 10))
	}

	return vecFields, nil
}

func (colReplica *collectionReplica) getFieldsByCollectionIDPrivate(collectionID UniqueID) ([]*schemapb.FieldSchema, error) {
	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	if len(collection.Schema().Fields) <= 0 {
		return nil, errors.New("no field in collection %d" + strconv.FormatInt(collectionID, 10))
	}

	return collection.Schema().Fields, nil
}

//----------------------------------------------------------------------------------------------------- partition
func (colReplica *collectionReplica) addPartition(collectionID UniqueID, partitionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	return colReplica.addPartitionPrivate(collectionID, partitionID)
}

func (colReplica *collectionReplica) addPartitionPrivate(collectionID UniqueID, partitionID UniqueID) error {
	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}

	collection.addPartitionID(partitionID)
	var newPartition = newPartition(collectionID, partitionID)
	colReplica.partitions[partitionID] = newPartition
	return nil
}

func (colReplica *collectionReplica) removePartition(partitionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	return colReplica.removePartitionPrivate(partitionID)
}

func (colReplica *collectionReplica) removePartitionPrivate(partitionID UniqueID) error {
	partition, err := colReplica.getPartitionByIDPrivate(partitionID)
	if err != nil {
		return err
	}

	collection, err := colReplica.getCollectionByIDPrivate(partition.collectionID)
	if err != nil {
		return err
	}

	// delete segments
	for _, segmentID := range partition.segmentIDs {
		// try to delete, ignore error
		_ = colReplica.removeSegmentPrivate(segmentID)
	}

	collection.removePartitionID(partitionID)
	delete(colReplica.partitions, partitionID)

	return nil
}

func (colReplica *collectionReplica) getPartitionByID(partitionID UniqueID) (*Partition, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return colReplica.getPartitionByIDPrivate(partitionID)
}

func (colReplica *collectionReplica) getPartitionByIDPrivate(partitionID UniqueID) (*Partition, error) {
	partition, ok := colReplica.partitions[partitionID]
	if !ok {
		return nil, errors.New("partition hasn't been loaded or has been released, partition id = %d" + strconv.FormatInt(partitionID, 10))
	}

	return partition, nil
}

func (colReplica *collectionReplica) hasPartition(partitionID UniqueID) bool {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return colReplica.hasPartitionPrivate(partitionID)
}

func (colReplica *collectionReplica) hasPartitionPrivate(partitionID UniqueID) bool {
	_, ok := colReplica.partitions[partitionID]
	return ok
}

func (colReplica *collectionReplica) getPartitionNum() int {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return len(colReplica.partitions)
}

func (colReplica *collectionReplica) getSegmentIDs(partitionID UniqueID) ([]UniqueID, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return colReplica.getSegmentIDsPrivate(partitionID)
}

func (colReplica *collectionReplica) getSegmentIDsPrivate(partitionID UniqueID) ([]UniqueID, error) {
	partition, err2 := colReplica.getPartitionByIDPrivate(partitionID)
	if err2 != nil {
		return nil, err2
	}
	return partition.segmentIDs, nil
}

//----------------------------------------------------------------------------------------------------- segment
func (colReplica *collectionReplica) addSegment(segmentID UniqueID,
	partitionID UniqueID,
	collectionID UniqueID,
	segType segmentType,
	onService bool) error {

	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}
	var newSegment = newSegment(collection, segmentID, partitionID, collectionID, segType, onService)
	return colReplica.addSegmentPrivate(segmentID, partitionID, newSegment)
}

func (colReplica *collectionReplica) addSegmentPrivate(segmentID UniqueID, partitionID UniqueID, segment *Segment) error {
	partition, err := colReplica.getPartitionByIDPrivate(partitionID)
	if err != nil {
		return err
	}

	if colReplica.hasSegmentPrivate(segmentID) {
		return nil
	}
	partition.addSegmentID(segmentID)
	colReplica.segments[segmentID] = segment

	return nil
}

func (colReplica *collectionReplica) setSegment(segment *Segment) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	_, err := colReplica.getCollectionByIDPrivate(segment.collectionID)
	if err != nil {
		return err
	}
	return colReplica.addSegmentPrivate(segment.segmentID, segment.partitionID, segment)
}

func (colReplica *collectionReplica) removeSegment(segmentID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	return colReplica.removeSegmentPrivate(segmentID)
}

func (colReplica *collectionReplica) removeSegmentPrivate(segmentID UniqueID) error {
	log.Debug("remove segment", zap.Int64("segmentID", segmentID))
	segment, err := colReplica.getSegmentByIDPrivate(segmentID)
	if err != nil {
		return err
	}

	partition, err2 := colReplica.getPartitionByIDPrivate(segment.partitionID)
	if err2 != nil {
		return err
	}

	partition.removeSegmentID(segmentID)
	delete(colReplica.segments, segmentID)
	deleteSegment(segment)

	return nil
}

func (colReplica *collectionReplica) getSegmentByID(segmentID UniqueID) (*Segment, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return colReplica.getSegmentByIDPrivate(segmentID)
}

func (colReplica *collectionReplica) getSegmentByIDPrivate(segmentID UniqueID) (*Segment, error) {
	segment, ok := colReplica.segments[segmentID]
	if !ok {
		return nil, errors.New("cannot find segment in query node, id = " + strconv.FormatInt(segmentID, 10))
	}

	return segment, nil
}

func (colReplica *collectionReplica) hasSegment(segmentID UniqueID) bool {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return colReplica.hasSegmentPrivate(segmentID)
}

func (colReplica *collectionReplica) hasSegmentPrivate(segmentID UniqueID) bool {
	_, ok := colReplica.segments[segmentID]
	return ok
}

func (colReplica *collectionReplica) getSegmentNum() int {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()
	return len(colReplica.segments)
}

func (colReplica *collectionReplica) getSegmentStatistics() []*internalpb.SegmentStats {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	var statisticData = make([]*internalpb.SegmentStats, 0)

	for segmentID, segment := range colReplica.segments {
		currentMemSize := segment.getMemSize()
		segment.lastMemSize = currentMemSize
		segmentNumOfRows := segment.getRowCount()

		stat := internalpb.SegmentStats{
			SegmentID:        segmentID,
			MemorySize:       currentMemSize,
			NumRows:          segmentNumOfRows,
			RecentlyModified: segment.getRecentlyModified(),
		}

		statisticData = append(statisticData, &stat)
		segment.setRecentlyModified(false)
	}

	return statisticData
}

func (colReplica *collectionReplica) getSegmentsBySegmentType(segType segmentType) ([]UniqueID, []UniqueID, []UniqueID) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	targetCollectionIDs := make([]UniqueID, 0)
	targetPartitionIDs := make([]UniqueID, 0)
	targetSegmentIDs := make([]UniqueID, 0)

	for _, segment := range colReplica.segments {
		if segment.getType() == segType {
			if segType == segmentTypeSealed && !segment.getEnableIndex() {
				continue
			}

			targetCollectionIDs = append(targetCollectionIDs, segment.collectionID)
			targetPartitionIDs = append(targetPartitionIDs, segment.partitionID)
			targetSegmentIDs = append(targetSegmentIDs, segment.segmentID)
		}
	}

	return targetCollectionIDs, targetPartitionIDs, targetSegmentIDs
}

func (colReplica *collectionReplica) replaceGrowingSegmentBySealedSegment(segment *Segment) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	if segment.segmentType != segmentTypeSealed && segment.segmentType != segmentTypeIndexing {
		deleteSegment(segment)
		return errors.New("unexpected segment type")
	}
	targetSegment, err := colReplica.getSegmentByIDPrivate(segment.ID())
	if err == nil && targetSegment != nil {
		if targetSegment.segmentType != segmentTypeGrowing {
			deleteSegment(segment)
			// target segment has been a sealed segment
			return nil
		}
		deleteSegment(targetSegment)
	}

	colReplica.segments[segment.ID()] = segment
	return nil
}

func (colReplica *collectionReplica) initExcludedSegments(collectionID UniqueID) {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	colReplica.excludedSegments[collectionID] = make([]*querypb.ExcludedSegmentInfo, 0)
}

func (colReplica *collectionReplica) removeExcludedSegments(collectionID UniqueID) {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	delete(colReplica.excludedSegments, collectionID)
}

func (colReplica *collectionReplica) addExcludedSegments(collectionID UniqueID, segmentInfos []*querypb.ExcludedSegmentInfo) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	if _, ok := colReplica.excludedSegments[collectionID]; !ok {
		return errors.New("addExcludedSegments failed, cannot found collection, id =" + fmt.Sprintln(collectionID))
	}

	colReplica.excludedSegments[collectionID] = append(colReplica.excludedSegments[collectionID], segmentInfos...)
	return nil
}

func (colReplica *collectionReplica) getExcludedSegments(collectionID UniqueID) ([]*querypb.ExcludedSegmentInfo, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	if _, ok := colReplica.excludedSegments[collectionID]; !ok {
		return nil, errors.New("getExcludedSegments failed, cannot found collection, id =" + fmt.Sprintln(collectionID))
	}

	return colReplica.excludedSegments[collectionID], nil
}

func (colReplica *collectionReplica) freeAll() {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	for id := range colReplica.collections {
		_ = colReplica.removeCollectionPrivate(id)
	}

	colReplica.collections = make(map[UniqueID]*Collection)
	colReplica.partitions = make(map[UniqueID]*Partition)
	colReplica.segments = make(map[UniqueID]*Segment)
}

func newCollectionReplica() ReplicaInterface {
	collections := make(map[UniqueID]*Collection)
	partitions := make(map[UniqueID]*Partition)
	segments := make(map[UniqueID]*Segment)
	excludedSegments := make(map[UniqueID][]*querypb.ExcludedSegmentInfo)

	var replica ReplicaInterface = &collectionReplica{
		collections: collections,
		partitions:  partitions,
		segments:    segments,

		excludedSegments: excludedSegments,
	}

	return replica
}
