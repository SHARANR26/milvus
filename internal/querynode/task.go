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

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	queryPb "github.com/milvus-io/milvus/internal/proto/querypb"
)

type task interface {
	ID() UniqueID       // return ReqID
	SetID(uid UniqueID) // set ReqID
	Timestamp() Timestamp
	PreExecute(ctx context.Context) error
	Execute(ctx context.Context) error
	PostExecute(ctx context.Context) error
	WaitToFinish() error
	Notify(err error)
	OnEnqueue() error
}

type baseTask struct {
	done chan error
	ctx  context.Context
	id   UniqueID
}

type watchDmChannelsTask struct {
	baseTask
	req  *queryPb.WatchDmChannelsRequest
	node *QueryNode
}

type loadSegmentsTask struct {
	baseTask
	req  *queryPb.LoadSegmentsRequest
	node *QueryNode
}

type releaseCollectionTask struct {
	baseTask
	req  *queryPb.ReleaseCollectionRequest
	node *QueryNode
}

type releasePartitionsTask struct {
	baseTask
	req  *queryPb.ReleasePartitionsRequest
	node *QueryNode
}

func (b *baseTask) ID() UniqueID {
	return b.id
}

func (b *baseTask) SetID(uid UniqueID) {
	b.id = uid
}

func (b *baseTask) WaitToFinish() error {
	err := <-b.done
	return err
}

func (b *baseTask) Notify(err error) {
	b.done <- err
}

// watchDmChannelsTask
func (w *watchDmChannelsTask) Timestamp() Timestamp {
	if w.req.Base == nil {
		log.Error("nil base req in watchDmChannelsTask", zap.Any("collectionID", w.req.CollectionID))
		return 0
	}
	return w.req.Base.Timestamp
}

func (w *watchDmChannelsTask) OnEnqueue() error {
	if w.req == nil || w.req.Base == nil {
		w.SetID(rand.Int63n(100000000000))
	} else {
		w.SetID(w.req.Base.MsgID)
	}
	return nil
}

func (w *watchDmChannelsTask) PreExecute(ctx context.Context) error {
	return nil
}

func (w *watchDmChannelsTask) Execute(ctx context.Context) error {
	log.Debug("starting WatchDmChannels ...", zap.String("ChannelIDs", fmt.Sprintln(w.req.ChannelIDs)))
	// TODO: pass load type, col or partition

	// 1. init channels in collection meta
	collectionID := w.req.CollectionID
	collection, err := w.node.streaming.replica.getCollectionByID(collectionID)
	if err != nil {
		log.Error(err.Error())
		return err
	}
	collection.addWatchedDmChannels(w.req.ChannelIDs)

	// 2. get subscription name
	getUniqueSubName := func() string {
		prefixName := Params.MsgChannelSubName
		return prefixName + "-" + strconv.FormatInt(collectionID, 10)
	}
	consumeSubName := getUniqueSubName()

	// 3. group channels by to seeking or consuming
	consumeChannels := w.req.ChannelIDs
	toSeekInfo := make([]*internalpb.MsgPosition, 0)
	toDirSubChannels := make([]string, 0)
	for _, info := range w.req.Infos {
		if len(info.Pos.MsgID) == 0 {
			toDirSubChannels = append(toDirSubChannels, info.ChannelID)
			continue
		}
		info.Pos.MsgGroup = consumeSubName
		toSeekInfo = append(toSeekInfo, info.Pos)

		log.Debug("prevent inserting segments", zap.String("segmentIDs", fmt.Sprintln(info.ExcludedSegments)))
		err := w.node.streaming.replica.addExcludedSegments(collectionID, info.ExcludedSegments)
		if err != nil {
			log.Error(err.Error())
			return err
		}
	}

	// 4. add flow graph
	err = w.node.streaming.dataSyncService.addCollectionFlowGraph(collectionID, consumeChannels)
	if err != nil {
		return err
	}
	log.Debug("query node add flow graphs, channels = " + strings.Join(consumeChannels, ", "))

	// 5. channels as consumer
	nodeFGs, err := w.node.streaming.dataSyncService.getCollectionFlowGraphs(collectionID)
	if err != nil {
		return err
	}
	for _, channel := range toDirSubChannels {
		for _, fg := range nodeFGs {
			if fg.channel == channel {
				err := fg.consumerFlowGraph(channel, consumeSubName)
				if err != nil {
					errMsg := "msgStream consume error :" + err.Error()
					log.Error(errMsg)
					return errors.New(errMsg)
				}
			}
		}
	}
	log.Debug("as consumer channels", zap.Any("channels", consumeChannels))

	// 6. seek channel
	for _, pos := range toSeekInfo {
		for _, fg := range nodeFGs {
			if fg.channel == pos.ChannelName {
				err := fg.seekQueryNodeFlowGraph(pos)
				if err != nil {
					errMsg := "msgStream seek error :" + err.Error()
					log.Error(errMsg)
					return errors.New(errMsg)
				}
			}
		}
	}

	// 7. start search collection
	w.node.searchService.startSearchCollection(collectionID)

	// 8. start flow graphs
	err = w.node.streaming.dataSyncService.startCollectionFlowGraph(collectionID)
	if err != nil {
		return err
	}

	log.Debug("WatchDmChannels done", zap.String("ChannelIDs", fmt.Sprintln(w.req.ChannelIDs)))
	return nil
}

func (w *watchDmChannelsTask) PostExecute(ctx context.Context) error {
	return nil
}

// loadSegmentsTask
func (l *loadSegmentsTask) Timestamp() Timestamp {
	if l.req.Base == nil {
		log.Error("nil base req in loadSegmentsTask", zap.Any("collectionID", l.req.CollectionID))
		return 0
	}
	return l.req.Base.Timestamp
}

func (l *loadSegmentsTask) OnEnqueue() error {
	if l.req == nil || l.req.Base == nil {
		l.SetID(rand.Int63n(100000000000))
	} else {
		l.SetID(l.req.Base.MsgID)
	}
	return nil
}

func (l *loadSegmentsTask) PreExecute(ctx context.Context) error {
	return nil
}

func (l *loadSegmentsTask) Execute(ctx context.Context) error {
	// TODO: support db
	collectionID := l.req.CollectionID
	partitionID := l.req.PartitionID
	segmentIDs := l.req.SegmentIDs
	fieldIDs := l.req.FieldIDs
	schema := l.req.Schema

	log.Debug("query node load segment", zap.String("loadSegmentRequest", fmt.Sprintln(l.req)))

	hasCollectionInHistorical := l.node.historical.replica.hasCollection(collectionID)
	hasPartitionInHistorical := l.node.historical.replica.hasPartition(partitionID)
	if !hasCollectionInHistorical {
		// loading init
		err := l.node.historical.replica.addCollection(collectionID, schema)
		if err != nil {
			return err
		}

		hasCollectionInStreaming := l.node.streaming.replica.hasCollection(collectionID)
		if !hasCollectionInStreaming {
			err = l.node.streaming.replica.addCollection(collectionID, schema)
			if err != nil {
				return err
			}
		}
		l.node.streaming.replica.initExcludedSegments(collectionID)
	}
	if !hasPartitionInHistorical {
		err := l.node.historical.replica.addPartition(collectionID, partitionID)
		if err != nil {
			return err
		}

		hasPartitionInStreaming := l.node.streaming.replica.hasPartition(partitionID)
		if !hasPartitionInStreaming {
			err = l.node.streaming.replica.addPartition(collectionID, partitionID)
			if err != nil {
				return err
			}
		}
	}
	err := l.node.streaming.replica.enablePartition(partitionID)
	if err != nil {
		return err
	}

	if len(segmentIDs) == 0 {
		return nil
	}

	err = l.node.historical.loadService.loadSegmentPassively(collectionID, partitionID, segmentIDs, fieldIDs)
	if err != nil {
		return err
	}

	log.Debug("LoadSegments done", zap.String("segmentIDs", fmt.Sprintln(l.req.SegmentIDs)))
	return nil
}

func (l *loadSegmentsTask) PostExecute(ctx context.Context) error {
	return nil
}

// releaseCollectionTask
func (r *releaseCollectionTask) Timestamp() Timestamp {
	if r.req.Base == nil {
		log.Error("nil base req in releaseCollectionTask", zap.Any("collectionID", r.req.CollectionID))
		return 0
	}
	return r.req.Base.Timestamp
}

func (r *releaseCollectionTask) OnEnqueue() error {
	if r.req == nil || r.req.Base == nil {
		r.SetID(rand.Int63n(100000000000))
	} else {
		r.SetID(r.req.Base.MsgID)
	}
	return nil
}

func (r *releaseCollectionTask) PreExecute(ctx context.Context) error {
	return nil
}

func (r *releaseCollectionTask) Execute(ctx context.Context) error {
	log.Debug("receive release collection task", zap.Any("collectionID", r.req.CollectionID))
	r.node.streaming.dataSyncService.removeCollectionFlowGraph(r.req.CollectionID)
	collection, err := r.node.historical.replica.getCollectionByID(r.req.CollectionID)
	if err != nil {
		log.Error(err.Error())
	} else {
		// remove all tSafes of the target collection
		for _, channel := range collection.getWatchedDmChannels() {
			r.node.streaming.tSafeReplica.removeTSafe(channel)
		}
	}

	r.node.streaming.replica.removeExcludedSegments(r.req.CollectionID)

	if r.node.searchService.hasSearchCollection(r.req.CollectionID) {
		r.node.searchService.stopSearchCollection(r.req.CollectionID)
	}

	hasCollectionInHistorical := r.node.historical.replica.hasCollection(r.req.CollectionID)
	if hasCollectionInHistorical {
		err := r.node.historical.replica.removeCollection(r.req.CollectionID)
		if err != nil {
			return err
		}
	}

	hasCollectionInStreaming := r.node.streaming.replica.hasCollection(r.req.CollectionID)
	if hasCollectionInStreaming {
		err := r.node.streaming.replica.removeCollection(r.req.CollectionID)
		if err != nil {
			return err
		}
	}

	log.Debug("ReleaseCollection done", zap.Int64("collectionID", r.req.CollectionID))
	return nil
}

func (r *releaseCollectionTask) PostExecute(ctx context.Context) error {
	return nil
}

// releasePartitionsTask
func (r *releasePartitionsTask) Timestamp() Timestamp {
	if r.req.Base == nil {
		log.Error("nil base req in releasePartitionsTask", zap.Any("collectionID", r.req.CollectionID))
		return 0
	}
	return r.req.Base.Timestamp
}

func (r *releasePartitionsTask) OnEnqueue() error {
	if r.req == nil || r.req.Base == nil {
		r.SetID(rand.Int63n(100000000000))
	} else {
		r.SetID(r.req.Base.MsgID)
	}
	return nil
}

func (r *releasePartitionsTask) PreExecute(ctx context.Context) error {
	return nil
}

func (r *releasePartitionsTask) Execute(ctx context.Context) error {
	for _, id := range r.req.PartitionIDs {
		hasPartitionInHistorical := r.node.historical.replica.hasPartition(id)
		if hasPartitionInHistorical {
			err := r.node.historical.replica.removePartition(id)
			if err != nil {
				// not return, try to release all partitions
				log.Error(err.Error())
			}
		}

		hasPartitionInStreaming := r.node.streaming.replica.hasPartition(id)
		if hasPartitionInStreaming {
			err := r.node.streaming.replica.removePartition(id)
			if err != nil {
				log.Error(err.Error())
			}
		}
	}
	return nil
}

func (r *releasePartitionsTask) PostExecute(ctx context.Context) error {
	return nil
}
