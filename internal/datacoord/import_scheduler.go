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
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"sync"
	"time"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

const (
	NullNodeID = -1
	GCDuration = 3 * time.Hour // TODO: dyh, make it configurable
)

type ImportScheduler interface {
	Start()
	Close()
}

type importScheduler struct {
	meta    *meta
	cluster Cluster
	alloc   allocator
	imeta   ImportMeta

	closeOnce sync.Once
	closeChan chan struct{}
}

func NewImportScheduler(meta *meta,
	cluster Cluster,
	alloc allocator,
	imeta ImportMeta,
) ImportScheduler {
	return &importScheduler{
		meta:      meta,
		cluster:   cluster,
		alloc:     alloc,
		imeta:     imeta,
		closeChan: make(chan struct{}),
	}
}

func (s *importScheduler) Start() {
	log.Info("start import scheduler")
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-s.closeChan:
			log.Info("import scheduler exited")
			return
		case <-ticker.C:
			s.process()
		}
	}
}

func (s *importScheduler) Close() {
	s.closeOnce.Do(func() {
		close(s.closeChan)
	})
}

func (s *importScheduler) process() {
	tasks := s.imeta.GetBy()
	for _, task := range tasks {
		switch task.GetState() {
		case internalpb.ImportState_Pending:
			switch task.GetType() {
			case PreImportTaskType:
				s.processPendingPreImport(task)
			case ImportTaskType:
				s.processPendingImport(task)
			}
		case internalpb.ImportState_InProgress:
			switch task.GetType() {
			case PreImportTaskType:
				s.processInProgressPreImport(task)
			case ImportTaskType:
				s.processInProgressImport(task)
			}
		case internalpb.ImportState_Completed:
			s.processCompleted(task)
		case internalpb.ImportState_Failed:
			s.processFailed(task)
		}
	}
}

func (s *importScheduler) checkErr(task ImportTask, err error) {
	if !merr.IsRetryableErr(err) {
		err = s.imeta.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_Failed), UpdateReason(err.Error()))
		if err != nil {
			log.Warn("failed to update import task state to failed", WrapLogFields(task, zap.Error(err))...)
		}
		return
	}
	err = s.imeta.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_Pending))
	if err != nil {
		log.Warn("failed to update import task state to pending", WrapLogFields(task, zap.Error(err))...)
	}
}

func (s *importScheduler) getIdleNode() int64 {
	nodeIDs := lo.Map(s.cluster.GetSessions(), func(s *Session, _ int) int64 {
		return s.info.NodeID
	})
	for _, nodeID := range nodeIDs {
		resp, err := s.cluster.QueryImport(nodeID, &datapb.QueryImportRequest{})
		if err != nil {
			log.Warn("query import failed", zap.Error(err))
			continue
		}
		if resp.GetSlots() > 0 {
			return nodeID
		}
	}
	return NullNodeID
}

func (s *importScheduler) processPendingPreImport(task ImportTask) {
	nodeID := s.getIdleNode()
	if nodeID == NullNodeID {
		log.Warn("no datanode can be scheduled", WrapLogFields(task)...)
		return
	}
	log.Info("processing pending preimport task...", WrapLogFields(task)...)
	req := AssemblePreImportRequest(task)
	err := s.cluster.PreImport(nodeID, req)
	if err != nil {
		log.Warn("preimport failed", WrapLogFields(task, zap.Error(err))...)
		return
	}
	err = s.imeta.Update(task.GetTaskID(),
		UpdateState(internalpb.ImportState_InProgress),
		UpdateNodeID(nodeID))
	if err != nil {
		log.Warn("update import task failed", WrapLogFields(task, zap.Error(err))...)
	}
	log.Info("process pending preimport task done", WrapLogFields(task)...)
}

func (s *importScheduler) processPendingImport(task ImportTask) {
	nodeID := s.getIdleNode()
	if nodeID == NullNodeID {
		log.Warn("no datanode can be scheduled", WrapLogFields(task)...)
		return
	}
	log.Info("processing pending import task...", WrapLogFields(task)...)
	req, err := AssembleImportRequest(task, s.meta, s.alloc)
	if err != nil {
		log.Warn("assemble import request failed", WrapLogFields(task, zap.Error(err))...)
		return
	}
	err = s.cluster.ImportV2(nodeID, req)
	if err != nil {
		log.Warn("import failed", WrapLogFields(task, zap.Error(err))...)
		return
	}
	err = s.imeta.Update(task.GetTaskID(),
		UpdateState(internalpb.ImportState_InProgress),
		UpdateNodeID(nodeID))
	if err != nil {
		log.Warn("update import task failed", WrapLogFields(task, zap.Error(err))...)
	}
	log.Info("processing pending import task done", WrapLogFields(task)...)
}

func (s *importScheduler) processInProgressPreImport(task ImportTask) {
	req := &datapb.QueryPreImportRequest{
		RequestID: task.GetRequestID(),
		TaskID:    task.GetTaskID(),
	}
	resp, err := s.cluster.QueryPreImport(task.GetNodeID(), req)
	if err != nil {
		log.Warn("query preimport failed", WrapLogFields(task, zap.Error(err))...)
		s.checkErr(task, err)
		return
	}
	actions := []UpdateAction{UpdateFileStats(resp.GetFileStats())}
	if resp.GetState() == internalpb.ImportState_Completed {
		actions = append(actions, UpdateState(internalpb.ImportState_Completed))
	} else if resp.GetState() == internalpb.ImportState_Failed {
		actions = append(actions, UpdateState(internalpb.ImportState_Failed), UpdateReason(resp.GetReason()))
	}
	// TODO: dyh, check if rows changed to save meta op
	err = s.imeta.Update(task.GetTaskID(), actions...)
	if err != nil {
		log.Warn("update import task failed", WrapLogFields(task, zap.Error(err))...)
		return
	}
	if resp.GetState() == internalpb.ImportState_Failed {
		log.Warn("preimport failed",
			WrapLogFields(task, zap.String("reason", resp.GetReason()))...)
	} else {
		log.Info("query preimport done",
			WrapLogFields(task, zap.Any("fileStats", resp.GetFileStats()))...)
	}
}

func (s *importScheduler) processInProgressImport(task ImportTask) {
	req := &datapb.QueryImportRequest{
		RequestID: task.GetRequestID(),
		TaskID:    task.GetTaskID(),
	}
	resp, err := s.cluster.QueryImport(task.GetNodeID(), req)
	if err != nil {
		log.Warn("query import failed", WrapLogFields(task, zap.Error(err))...)
		s.checkErr(task, err)
		return
	}
	for _, info := range resp.GetImportSegmentsInfo() {
		// TODO: dyh, check if segment info changed to save meta op
		op1 := UpdateBinlogsOperator(info.GetSegmentID(), info.GetBinlogs(), info.GetStatslogs(), nil)
		op2 := UpdateNumOfRows(info.GetSegmentID(), info.GetImportedRows())
		err = s.meta.UpdateSegmentsInfo(op1, op2)
		if err != nil {
			log.Warn("update import segment info failed", WrapLogFields(task, zap.Error(err))...)
			continue
		}
	}
	if resp.GetState() == internalpb.ImportState_Completed {
		err = s.imeta.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_Completed))
		if err != nil {
			log.Warn("update import task failed", WrapLogFields(task, zap.Error(err))...)
		}
	}
	if resp.GetState() == internalpb.ImportState_Failed {
		err = s.imeta.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_Failed), UpdateReason(resp.GetReason()))
		if err != nil {
			log.Warn("update import task failed", WrapLogFields(task, zap.Error(err))...)
		}
	}
	log.Info("query import done", WrapLogFields(task)...)
}

func (s *importScheduler) processCompleted(task ImportTask) {
	err := DropImportTask(task, s.cluster, s.imeta)
	if err != nil {
		log.Warn("drop import failed", WrapLogFields(task, zap.Error(err))...)
		return
	}
	if time.Since(task.GetLastActiveTime()) >= GCDuration {
		err = s.imeta.Remove(task.GetTaskID())
		if err != nil {
			log.Warn("remove import task failed", WrapLogFields(task, zap.Error(err))...)
		}
	}
}

func (s *importScheduler) processFailed(task ImportTask) {
	if task.GetType() == ImportTaskType {
		segments := task.(*importTask).GetSegmentIDs()
		for _, segment := range segments {
			err := s.meta.DropSegment(segment)
			if err != nil {
				log.Warn("drop import segment failed",
					WrapLogFields(task, zap.Int64("segment", segment), zap.Error(err))...)
				return
			}
		}
		err := s.imeta.Update(task.GetTaskID(), UpdateSegmentIDs(nil))
		if err != nil {
			log.Warn("update import task segments failed", WrapLogFields(task, zap.Error(err))...)
			return
		}
	}
	err := DropImportTask(task, s.cluster, s.imeta)
	if err != nil {
		log.Warn("drop import failed", WrapLogFields(task, zap.Error(err))...)
		return
	}
	if time.Since(task.GetLastActiveTime()) >= GCDuration {
		err = s.imeta.Remove(task.GetTaskID())
		if err != nil {
			log.Warn("remove import task failed", WrapLogFields(task, zap.Error(err))...)
		}
	}
}