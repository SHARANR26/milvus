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

package datacoord

import (
	"context"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus/internal/kv"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/kv/mocks"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/testutils"
)

func TestCluster(t *testing.T) {
	suite.Run(t, new(ClusterSuite))
}

func getWatchKV(t *testing.T) kv.WatchKV {
	rootPath := "/etcd/test/root/" + t.Name()
	kv, err := etcdkv.NewWatchKVFactory(rootPath, &Params.EtcdCfg)
	require.NoError(t, err)

	return kv
}

type ClusterSuite struct {
	testutils.PromMetricsSuite

	mockKv        *mocks.WatchKV
	mockChManager *MockChannelManager
	mockSession   *MockSessionManager
}

func (suite *ClusterSuite) SetupTest() {
	suite.mockKv = mocks.NewWatchKV(suite.T())
	suite.mockChManager = NewMockChannelManager(suite.T())
	suite.mockSession = NewMockSessionManager(suite.T())
}

func (suite *ClusterSuite) TearDownTest() {}

func (suite *ClusterSuite) TestStartup() {
	nodes := []*NodeInfo{
		{NodeID: 1, Address: "addr1"},
		{NodeID: 2, Address: "addr2"},
		{NodeID: 3, Address: "addr3"},
		{NodeID: 4, Address: "addr4"},
	}
	suite.mockSession.EXPECT().AddSession(mock.Anything).Return().Times(len(nodes))
	suite.mockChManager.EXPECT().Startup(mock.Anything, mock.Anything).
		RunAndReturn(func(ctx context.Context, nodeIDs []int64) error {
			suite.ElementsMatch(lo.Map(nodes, func(info *NodeInfo, _ int) int64 { return info.NodeID }), nodeIDs)
			return nil
		}).Once()

	cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
	err := cluster.Startup(context.Background(), nodes)
	suite.NoError(err)
}

func (suite *ClusterSuite) TestRegister() {
	info := &NodeInfo{NodeID: 1, Address: "addr1"}

	suite.mockSession.EXPECT().AddSession(mock.Anything).Return().Once()
	suite.mockChManager.EXPECT().AddNode(mock.Anything).
		RunAndReturn(func(nodeID int64) error {
			suite.EqualValues(info.NodeID, nodeID)
			return nil
		}).Once()

	cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
	err := cluster.Register(info)
	suite.NoError(err)
}

func (suite *ClusterSuite) TestUnregister() {
	info := &NodeInfo{NodeID: 1, Address: "addr1"}

	suite.mockSession.EXPECT().DeleteSession(mock.Anything).Return().Once()
	suite.mockChManager.EXPECT().DeleteNode(mock.Anything).
		RunAndReturn(func(nodeID int64) error {
			suite.EqualValues(info.NodeID, nodeID)
			return nil
		}).Once()

	cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
	err := cluster.UnRegister(info)
	suite.NoError(err)
}

func (suite *ClusterSuite) TestWatch() {
	var (
		ch           string   = "ch-1"
		collectionID UniqueID = 1
	)

	suite.mockChManager.EXPECT().Watch(mock.Anything, mock.Anything).
		RunAndReturn(func(ctx context.Context, channel RWChannel) error {
			suite.EqualValues(ch, channel.GetName())
			suite.EqualValues(collectionID, channel.GetCollectionID())
			return nil
		}).Once()

	cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
	err := cluster.Watch(context.Background(), ch, collectionID)
	suite.NoError(err)
}

func (suite *ClusterSuite) TestFlush() {
	suite.mockChManager.EXPECT().Match(mock.Anything, mock.Anything).
		RunAndReturn(func(nodeID int64, channel string) bool {
			return nodeID != 1
		}).Twice()

	suite.mockChManager.EXPECT().GetCollectionIDByChannel(mock.Anything).Return(true, 100).Once()
	suite.mockSession.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Once()

	cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)

	err := cluster.Flush(context.Background(), 1, "ch-1", nil)
	suite.Error(err)

	err = cluster.Flush(context.Background(), 2, "ch-1", nil)
	suite.NoError(err)
}

func (suite *ClusterSuite) TestFlushChannels() {
	suite.Run("empty channel", func() {
		suite.SetupTest()

		cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
		err := cluster.FlushChannels(context.Background(), 1, 0, nil)
		suite.NoError(err)
	})

	suite.Run("channel not match with node", func() {
		suite.SetupTest()

		suite.mockChManager.EXPECT().Match(mock.Anything, mock.Anything).Return(false).Once()
		cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
		err := cluster.FlushChannels(context.Background(), 1, 0, []string{"ch-1", "ch-2"})
		suite.Error(err)
	})

	suite.Run("channel match with node", func() {
		suite.SetupTest()

		channels := []string{"ch-1", "ch-2"}
		suite.mockChManager.EXPECT().Match(mock.Anything, mock.Anything).Return(true).Times(len(channels))
		suite.mockSession.EXPECT().FlushChannels(mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
		err := cluster.FlushChannels(context.Background(), 1, 0, channels)
		suite.NoError(err)
	})
}

func (suite *ClusterSuite) TestImport() {
	suite.mockSession.EXPECT().Import(mock.Anything, mock.Anything, mock.Anything).Return().Once()
	cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
	suite.NotPanics(func() {
		cluster.Import(context.Background(), 1, nil)
	})
}

func (suite *ClusterSuite) TestAddImportSegment() {
	suite.Run("channel not fount", func() {
		suite.SetupTest()
		suite.mockChManager.EXPECT().GetNodeIDByChannelName(mock.Anything).Return(false, 0)
		cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
		resp, err := cluster.AddImportSegment(context.Background(), &datapb.AddImportSegmentRequest{
			ChannelName: "ch-1",
		})

		suite.ErrorIs(err, merr.ErrChannelNotFound)
		suite.Nil(resp)
	})

	suite.Run("normal", func() {
		suite.SetupTest()
		suite.mockChManager.EXPECT().GetNodeIDByChannelName(mock.Anything).Return(true, 0)
		suite.mockSession.EXPECT().AddImportSegment(mock.Anything, mock.Anything, mock.Anything).Return(&datapb.AddImportSegmentResponse{}, nil)

		cluster := NewClusterImpl(suite.mockSession, suite.mockChManager)
		resp, err := cluster.AddImportSegment(context.Background(), &datapb.AddImportSegmentRequest{
			ChannelName: "ch-1",
		})

		suite.NoError(err)
		suite.NotNil(resp)
	})
}
