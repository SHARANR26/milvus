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

package datanode

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/types"
)

func newSegmentReplica(ms types.MasterService, collID UniqueID) *SegmentReplica {
	metaService := newMetaService(ms, collID)

	var replica = &SegmentReplica{
		collectionID: collID,

		newSegments:     make(map[UniqueID]*Segment),
		normalSegments:  make(map[UniqueID]*Segment),
		flushedSegments: make(map[UniqueID]*Segment),

		metaService: metaService,
	}
	return replica
}

func TestSegmentReplica(t *testing.T) {
	mockMaster := &MasterServiceFactory{}
	collID := UniqueID(1)

	t.Run("Test inner function segment", func(t *testing.T) {
		replica := newSegmentReplica(mockMaster, collID)
		assert.False(t, replica.hasSegment(0))

		startPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(100)}
		endPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(200)}
		err := replica.addNewSegment(0, 1, 2, "insert-01", startPos, endPos)
		assert.NoError(t, err)
		assert.True(t, replica.hasSegment(0))
		assert.Equal(t, 1, len(replica.newSegments))

		seg, ok := replica.newSegments[UniqueID(0)]
		assert.True(t, ok)
		require.NotNil(t, seg)
		assert.Equal(t, UniqueID(0), seg.segmentID)
		assert.Equal(t, UniqueID(1), seg.collectionID)
		assert.Equal(t, UniqueID(2), seg.partitionID)
		assert.Equal(t, "insert-01", seg.channelName)
		assert.Equal(t, Timestamp(100), seg.startPos.Timestamp)
		assert.Equal(t, Timestamp(200), seg.endPos.Timestamp)
		assert.Equal(t, startPos.ChannelName, seg.checkPoint.pos.ChannelName)
		assert.Equal(t, startPos.Timestamp, seg.checkPoint.pos.Timestamp)
		assert.Equal(t, int64(0), seg.numRows)
		assert.True(t, seg.isNew.Load().(bool))
		assert.False(t, seg.isFlushed.Load().(bool))

		err = replica.updateStatistics(0, 10)
		assert.NoError(t, err)
		assert.Equal(t, int64(10), seg.numRows)

		cpPos := &internalpb.MsgPosition{ChannelName: "insert-01", Timestamp: Timestamp(10)}
		cp := &segmentCheckPoint{int64(10), *cpPos}
		err = replica.addNormalSegment(1, 1, 2, "insert-01", int64(10), cp)
		assert.NoError(t, err)
		assert.True(t, replica.hasSegment(1))
		assert.Equal(t, 1, len(replica.normalSegments))
		seg, ok = replica.normalSegments[UniqueID(1)]
		assert.True(t, ok)
		require.NotNil(t, seg)
		assert.Equal(t, UniqueID(1), seg.segmentID)
		assert.Equal(t, UniqueID(1), seg.collectionID)
		assert.Equal(t, UniqueID(2), seg.partitionID)
		assert.Equal(t, "insert-01", seg.channelName)
		assert.Equal(t, cpPos.ChannelName, seg.checkPoint.pos.ChannelName)
		assert.Equal(t, cpPos.Timestamp, seg.checkPoint.pos.Timestamp)
		assert.Equal(t, int64(10), seg.numRows)
		assert.False(t, seg.isNew.Load().(bool))
		assert.False(t, seg.isFlushed.Load().(bool))

		err = replica.updateStatistics(1, 10)
		assert.NoError(t, err)
		assert.Equal(t, int64(20), seg.numRows)

		segPos := replica.listNewSegmentsStartPositions()
		assert.Equal(t, 1, len(segPos))
		assert.Equal(t, UniqueID(0), segPos[0].SegmentID)
		assert.Equal(t, "insert-01", segPos[0].StartPosition.ChannelName)
		assert.Equal(t, Timestamp(100), segPos[0].StartPosition.Timestamp)

		assert.Equal(t, 0, len(replica.newSegments))
		assert.Equal(t, 2, len(replica.normalSegments))

		cps := replica.listSegmentsCheckPoints()
		assert.Equal(t, 2, len(cps))
		assert.Equal(t, startPos.Timestamp, cps[UniqueID(0)].pos.Timestamp)
		assert.Equal(t, int64(0), cps[UniqueID(0)].numRows)
		assert.Equal(t, cp.pos.Timestamp, cps[UniqueID(1)].pos.Timestamp)
		assert.Equal(t, int64(10), cps[UniqueID(1)].numRows)

		updates, err := replica.getSegmentStatisticsUpdates(0)
		assert.NoError(t, err)
		assert.Equal(t, int64(10), updates.NumRows)

		updates, err = replica.getSegmentStatisticsUpdates(1)
		assert.NoError(t, err)
		assert.Equal(t, int64(20), updates.NumRows)

		replica.updateSegmentCheckPoint(0)
		assert.Equal(t, int64(10), replica.normalSegments[UniqueID(0)].checkPoint.numRows)
		replica.updateSegmentCheckPoint(1)
		assert.Equal(t, int64(20), replica.normalSegments[UniqueID(1)].checkPoint.numRows)

	})
}
