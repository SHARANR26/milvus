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

package importv2

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/samber/lo"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/datanode/syncmgr"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/importutilv2"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

const BufferSize = 64 * 1024 * 1024 // TODO: dyh, make it configurable

type Executor interface {
	Start()
	Close()
}

type executor struct {
	manager TaskManager
	syncMgr syncmgr.SyncManager
	cm      storage.ChunkManager

	// TODO: dyh, add thread pool

	closeOnce sync.Once
	closeChan chan struct{}
}

func NewExecutor(manager TaskManager, syncMgr syncmgr.SyncManager, cm storage.ChunkManager) Executor {
	return &executor{
		manager:   manager,
		syncMgr:   syncMgr,
		cm:        cm,
		closeChan: make(chan struct{}),
	}
}

func (e *executor) Start() {
	log.Info("start import executor")
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-e.closeChan:
			log.Info("import executor exited")
			return
		case <-ticker.C:
			tasks := e.manager.GetBy(WithStates(internalpb.ImportState_Pending))
			wg, _ := errgroup.WithContext(context.Background())
			for _, task := range tasks {
				task := task
				wg.Go(func() error {
					switch task.GetType() {
					case PreImportTaskType:
						e.PreImport(task)
					case ImportTaskType:
						e.Import(task)
					}
					return nil
				})
			}
			_ = wg.Wait()
			LogStats(e.manager)
		}
	}
}

func (e *executor) Close() {
	e.closeOnce.Do(func() {
		close(e.closeChan)
	})
}

func (e *executor) estimateReadRows(schema *schemapb.CollectionSchema) (int64, error) {
	sizePerRow, err := typeutil.EstimateSizePerRecord(schema)
	if err != nil {
		return 0, err
	}
	return int64(BufferSize / sizePerRow), nil
}

func WrapLogFields(task Task, fields ...zap.Field) []zap.Field {
	res := []zap.Field{
		zap.Int64("taskID", task.GetTaskID()),
		zap.Int64("requestID", task.GetRequestID()),
		zap.Int64("collectionID", task.GetCollectionID()),
		zap.String("state", task.GetState().String()),
		zap.String("type", task.GetType().String()),
	}
	res = append(res, fields...)
	return res
}

func (e *executor) handleErr(task Task, err error, msg string) {
	log.Warn(msg, WrapLogFields(task, zap.Error(err))...)
	e.manager.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_Failed), UpdateReason(err.Error()))
}

func (e *executor) PreImport(task Task) {
	log.Info("start to preimport", WrapLogFields(task, zap.Any("schema", task.GetSchema()))...)
	e.manager.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_InProgress))
	files := lo.Map(task.(*PreImportTask).GetFileStats(),
		func(fileStat *datapb.ImportFileStats, _ int) *internalpb.ImportFile {
			return fileStat.GetImportFile()
		})

	for i, file := range files {
		reader, err := importutilv2.NewReader(task.GetCtx(), e.cm, task.GetSchema(), file, nil, BufferSize) // TODO: dyh, fix options
		if err != nil {
			e.handleErr(task, err, "new reader failed")
			return
		}
		start := time.Now()
		err = e.readFileStat(reader, task, i)
		if err != nil {
			e.handleErr(task, err, "preimport failed")
			reader.Close()
			return
		}
		reader.Close()
		log.Info("read file stat done", WrapLogFields(task, zap.Strings("files", file.GetPaths()),
			zap.Duration("dur", time.Since(start)))...)
	}

	e.manager.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_Completed))
	log.Info("executor preimport done",
		WrapLogFields(task, zap.Any("fileStats", task.(*PreImportTask).GetFileStats()))...)
}

func (e *executor) readFileStat(reader importutilv2.Reader, task Task, fileIdx int) error {
	totalRows := 0
	hashedRows := make(map[string]*datapb.PartitionRows)
	for {
		data, err := reader.Read()
		if err != nil {
			return err
		}
		if data == nil {
			break
		}
		err = FillDynamicData(data, task.GetSchema())
		if err != nil {
			return err
		}
		err = CheckRowsEqual(task.GetSchema(), data)
		if err != nil {
			return err
		}
		rowsCount, err := GetRowsStats(task, data)
		if err != nil {
			return err
		}
		MergeHashedRowsCount(rowsCount, hashedRows)
		totalRows += data.GetRowNum()
	}

	stat := &datapb.ImportFileStats{
		TotalRows:  int64(totalRows),
		HashedRows: hashedRows,
	}
	e.manager.Update(task.GetTaskID(), UpdateFileStat(fileIdx, stat))
	return nil
}

func (e *executor) Import(task Task) {
	log.Info("start to import", WrapLogFields(task, zap.Any("schema", task.GetSchema()))...)
	e.manager.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_InProgress))

	req := task.(*ImportTask).req
	for _, file := range req.GetFiles() {
		reader, err := importutilv2.NewReader(task.GetCtx(), e.cm, task.GetSchema(), file, nil, BufferSize) // TODO: dyh, fix options
		if err != nil {
			e.handleErr(task, err, fmt.Sprintf("new reader failed, file: %s", file.String()))
			return
		}
		start := time.Now()
		err = e.importFile(reader, task)
		if err != nil {
			e.handleErr(task, err, fmt.Sprintf("do import failed, file: %s", file.String()))
			reader.Close()
			return
		}
		reader.Close()
		log.Info("import file done", WrapLogFields(task, zap.Strings("files", file.GetPaths()),
			zap.Duration("dur", time.Since(start)))...)
	}
	e.manager.Update(task.GetTaskID(), UpdateState(internalpb.ImportState_Completed))
	log.Info("import done", WrapLogFields(task)...)
}

func (e *executor) importFile(reader importutilv2.Reader, task Task) error {
	for {
		data, err := reader.Read()
		if err != nil {
			return err
		}
		if data == nil {
			return nil
		}
		err = FillDynamicData(data, task.GetSchema())
		if err != nil {
			return err
		}
		iTask := task.(*ImportTask)
		err = AppendSystemFieldsData(iTask, data)
		if err != nil {
			return err
		}
		hashedData, err := HashData(iTask, data)
		if err != nil {
			return err
		}
		err = e.Sync(iTask, hashedData)
		if err != nil {
			return err
		}
	}
}

func (e *executor) Sync(task *ImportTask, hashedData HashedData) error {
	log.Info("start to sync import data", WrapLogFields(task)...)
	futures := make([]*conc.Future[error], 0)
	syncTasks := make([]syncmgr.Task, 0)
	for channelIdx, datas := range hashedData {
		channel := task.vchannels[channelIdx]
		for partitionIdx, data := range datas {
			partitionID := task.partitions[partitionIdx]
			segmentID := PickSegment(task, channel, partitionID, data.GetRowNum())
			syncTask, err := NewSyncTask(task.GetCtx(), task, segmentID, partitionID, channel, data)
			if err != nil {
				return err
			}
			future := e.syncMgr.SyncData(task.GetCtx(), syncTask)
			futures = append(futures, future)
			syncTasks = append(syncTasks, syncTask)
		}
	}
	err := conc.AwaitAll(futures...) // TODO: dyh, return futures and syncTasks to increase concurrence
	if err != nil {
		return err
	}
	for _, syncTask := range syncTasks {
		segmentInfo, err := NewImportSegmentInfo(syncTask, task)
		if err != nil {
			return err
		}
		e.manager.Update(task.GetTaskID(), UpdateSegmentInfo(segmentInfo))
	}
	importedRows := lo.Map(task.GetSegmentsInfo(), func(info *datapb.ImportSegmentInfo, _ int) int64 {
		return info.GetImportedRows()
	})
	log.Info("sync import data done", WrapLogFields(task, zap.Any("importedRows", importedRows))...)
	return nil
}