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

	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
)

type queryNodeFlowGraph struct {
	ctx          context.Context
	cancel       context.CancelFunc
	collectionID UniqueID
	partitionID  UniqueID
	channel      VChannel
	flowGraph    *flowgraph.TimeTickedFlowGraph
	dmlStream    msgstream.MsgStream
}

func newQueryNodeFlowGraph(ctx context.Context,
	flowGraphType flowGraphType,
	collectionID UniqueID,
	partitionID UniqueID,
	streamingReplica ReplicaInterface,
	tSafeReplica TSafeReplicaInterface,
	channel VChannel,
	factory msgstream.Factory) *queryNodeFlowGraph {

	ctx1, cancel := context.WithCancel(ctx)

	q := &queryNodeFlowGraph{
		ctx:          ctx1,
		cancel:       cancel,
		collectionID: collectionID,
		partitionID:  partitionID,
		channel:      channel,
		flowGraph:    flowgraph.NewTimeTickedFlowGraph(ctx1),
	}

	var dmStreamNode node = q.newDmInputNode(ctx1, factory)
	var filterDmNode node = newFilteredDmNode(streamingReplica, flowGraphType, collectionID, partitionID)
	var insertNode node = newInsertNode(streamingReplica)
	var serviceTimeNode node = newServiceTimeNode(ctx1, tSafeReplica, flowGraphType, collectionID, partitionID, channel, factory)

	q.flowGraph.AddNode(dmStreamNode)
	q.flowGraph.AddNode(filterDmNode)
	q.flowGraph.AddNode(insertNode)
	q.flowGraph.AddNode(serviceTimeNode)

	// dmStreamNode
	var err = q.flowGraph.SetEdges(dmStreamNode.Name(),
		[]string{},
		[]string{filterDmNode.Name()},
	)
	if err != nil {
		log.Error("set edges failed in node:", zap.String("node name", dmStreamNode.Name()))
	}

	// filterDmNode
	err = q.flowGraph.SetEdges(filterDmNode.Name(),
		[]string{dmStreamNode.Name()},
		[]string{insertNode.Name()},
	)
	if err != nil {
		log.Error("set edges failed in node:", zap.String("node name", filterDmNode.Name()))
	}

	// insertNode
	err = q.flowGraph.SetEdges(insertNode.Name(),
		[]string{filterDmNode.Name()},
		[]string{serviceTimeNode.Name()},
	)
	if err != nil {
		log.Error("set edges failed in node:", zap.String("node name", insertNode.Name()))
	}

	// serviceTimeNode
	err = q.flowGraph.SetEdges(serviceTimeNode.Name(),
		[]string{insertNode.Name()},
		[]string{},
	)
	if err != nil {
		log.Error("set edges failed in node:", zap.String("node name", serviceTimeNode.Name()))
	}

	return q
}

func (q *queryNodeFlowGraph) newDmInputNode(ctx context.Context, factory msgstream.Factory) *flowgraph.InputNode {
	insertStream, err := factory.NewTtMsgStream(ctx)
	if err != nil {
		log.Error(err.Error())
	} else {
		q.dmlStream = insertStream
	}

	maxQueueLength := Params.FlowGraphMaxQueueLength
	maxParallelism := Params.FlowGraphMaxParallelism

	node := flowgraph.NewInputNode(&insertStream, "dmlInputNode", maxQueueLength, maxParallelism)
	return node
}

func (q *queryNodeFlowGraph) consumerFlowGraph(channel VChannel, subName ConsumeSubName) error {
	if q.dmlStream == nil {
		return errors.New("null dml message stream in flow graph")
	}
	q.dmlStream.AsConsumer([]string{channel}, subName)
	log.Debug("query node flow graph consumes from virtual channel", zap.Any("vChannel", channel))
	return nil
}

func (q *queryNodeFlowGraph) seekQueryNodeFlowGraph(position *internalpb.MsgPosition) error {
	err := q.dmlStream.Seek([]*internalpb.MsgPosition{position})
	return err
}

func (q *queryNodeFlowGraph) close() {
	q.cancel()
	q.flowGraph.Close()
	log.Debug("stop query node flow graph",
		zap.Any("collectionID", q.collectionID),
		zap.Any("partitionID", q.partitionID),
		zap.Any("channel", q.channel),
	)
}
