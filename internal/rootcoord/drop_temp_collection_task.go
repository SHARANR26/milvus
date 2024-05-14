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

package rootcoord

import (
	"context"
	"fmt"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/internal/metastore/model"
	pb "github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/proto/rootcoordpb"
	"github.com/milvus-io/milvus/pkg/util"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type dropTempCollectionTask struct {
	baseTask
	Req         *indexpb.CollectionWithTempRequest
	TruncateReq *rootcoordpb.TruncateCollectionRequest
	Sync        bool
	collInfo    *model.Collection
}

// todo msg invalid type
func (t *dropTempCollectionTask) validate(ctx context.Context) error {
	return nil
}

func (t *dropTempCollectionTask) Prepare(ctx context.Context) error {
	err := t.validate(ctx)
	if err != nil {
		return err
	}
	var collInfo *model.Collection
	if t.Req == nil {
		collInfo, err = t.core.meta.GetCollectionByName(ctx, t.TruncateReq.GetDbName(), t.TruncateReq.GetCollectionName(), typeutil.MaxTimestamp)
		if err != nil && !errors.Is(err, merr.ErrCollectionNotFound) {
			return err
		}
		t.collInfo, err = t.core.meta.GetCollectionByName(ctx, t.TruncateReq.GetDbName(), util.GenerateTempCollectionName(t.TruncateReq.GetCollectionName()), typeutil.MaxTimestamp)
		if err == nil {
			t.Req = &indexpb.CollectionWithTempRequest{
				CollectionID:     t.collInfo.CollectionID,
				TempCollectionID: t.collInfo.CollectionID,
			}
			if collInfo != nil {
				t.Req.CollectionID = collInfo.CollectionID
			}
		}
	} else {
		t.collInfo, err = t.core.meta.GetCollectionByID(ctx, "", t.Req.TempCollectionID, typeutil.MaxTimestamp, true)
		if err == nil && !t.Sync && t.collInfo.State != pb.CollectionState_CollectionDropping {
			return errors.New("error")
		}
	}
	return err
}

func (t *dropTempCollectionTask) Execute(ctx context.Context) error {
	ts := t.GetTs()

	redoTask := newBaseRedoTask(t.core.stepExecutor)

	// all the step should be 幂等
	if t.Sync {
		redoTask.AddSyncStep(&changeCollectionStateStep{
			baseStep:     baseStep{core: t.core},
			collectionID: t.collInfo.CollectionID,
			state:        pb.CollectionState_CollectionDropping,
			ts:           ts,
		})
		redoTask.AddSyncStep(&releaseCollectionStep{
			baseStep:     baseStep{core: t.core},
			collectionID: t.collInfo.CollectionID,
		})
		redoTask.AddSyncStep(&dropIndexTempStep{
			baseStep: baseStep{core: t.core},
			collID:   t.collInfo.CollectionID,
			partIDs:  nil,
		})
		redoTask.AddSyncStep(&deleteCollectionDataStep{
			baseStep: baseStep{core: t.core},
			coll:     t.collInfo,
			// isSkip:   t.Req.GetBase().GetReplicateInfo().GetIsReplicate(),
		})
		redoTask.AddSyncStep(&removeDmlChannelsStep{
			baseStep:  baseStep{core: t.core},
			pChannels: t.collInfo.PhysicalChannelNames,
		})
		redoTask.AddSyncStep(&deleteCollectionMetaStep{
			baseStep:     baseStep{core: t.core},
			collectionID: t.collInfo.CollectionID,
			// This ts is less than the ts when we notify data nodes to drop collection, but it's OK since we have already
			// marked this collection as deleted. If we want to make this ts greater than the notification's ts, we should
			// wrap a step who will have these three children and connect them with ts.
			ts: ts,
		})
	} else {
		redoTask.AddAsyncStep(&releaseCollectionStep{
			baseStep:     baseStep{core: t.core},
			collectionID: t.collInfo.CollectionID,
		})
		redoTask.AddAsyncStep(&dropIndexTempStep{
			baseStep: baseStep{core: t.core},
			collID:   t.collInfo.CollectionID,
			partIDs:  nil,
		})
		redoTask.AddAsyncStep(&deleteCollectionDataStep{
			baseStep: baseStep{core: t.core},
			coll:     t.collInfo,
			// isSkip:   t.Req.GetBase().GetReplicateInfo().GetIsReplicate(),
		})
		redoTask.AddAsyncStep(&removeDmlChannelsStep{
			baseStep:  baseStep{core: t.core},
			pChannels: t.collInfo.PhysicalChannelNames,
		})
		redoTask.AddAsyncStep(newConfirmGCStep(t.core, t.collInfo.CollectionID, allPartition))
		redoTask.AddAsyncStep(&deleteCollectionMetaStep{
			baseStep:     baseStep{core: t.core},
			collectionID: t.collInfo.CollectionID,
			// This ts is less than the ts when we notify data nodes to drop collection, but it's OK since we have already
			// marked this collection as deleted. If we want to make this ts greater than the notification's ts, we should
			// wrap a step who will have these three children and connect them with ts.
			ts: ts,
		})
	}

	return redoTask.Execute(ctx)
}

type dropIndexTempStep struct {
	baseStep
	collID  UniqueID
	partIDs []UniqueID
}

func (s *dropIndexTempStep) Execute(ctx context.Context) ([]nestedStep, error) {
	err := s.core.broker.DropTempCollectionIndexes(ctx, s.collID, s.partIDs)
	return nil, err
}

func (s *dropIndexTempStep) Desc() string {
	return fmt.Sprintf("drop collection index: %d", s.collID)
}

func (s *dropIndexTempStep) Weight() stepPriority {
	return stepPriorityNormal
}
