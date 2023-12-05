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

package binlog

import (
	"encoding/json"
	"math"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type reader struct {
	cm     storage.ChunkManager
	schema *schemapb.CollectionSchema

	deleteData *storage.DeleteData
	insertLogs map[int64][]string // fieldID -> binlogs

	readIdx int
	filters []Filter
}

func NewReader(cm storage.ChunkManager,
	schema *schemapb.CollectionSchema,
	paths []string,
	tsStart,
	tsEnd uint64,
) (*reader, error) {
	r := &reader{
		cm:     cm,
		schema: schema,
	}
	err := r.Init(paths, tsStart, tsEnd)
	if err != nil {
		return nil, err
	}
	return r, nil
}

func (r *reader) Init(paths []string, tsStart, tsEnd uint64) error {
	if tsStart != 0 || tsEnd != math.MaxUint64 {
		r.filters = append(r.filters, FilterWithTimeRange(tsStart, tsEnd))
	}
	insertLogs, deltaLogs, err := listBinlogs(r.cm, paths)
	if err != nil {
		return err
	}
	err = verify(r.schema, insertLogs)
	if err != nil {
		return err
	}
	r.insertLogs = insertLogs
	if len(deltaLogs) > 0 {
		r.deleteData, err = r.ReadDelete(deltaLogs, tsStart, tsEnd)
		if err != nil {
			return err
		}
		deleteFilter, err := FilterWithDelete(r)
		if err != nil {
			return err
		}
		r.filters = append(r.filters, deleteFilter)
	}
	return nil
}

func (r *reader) ReadDelete(deltaLogs []string, tsStart, tsEnd uint64) (*storage.DeleteData, error) {
	deleteData := storage.NewDeleteData(nil, nil)
	for _, path := range deltaLogs {
		reader, err := newBinlogReader(r.cm, path)
		if err != nil {
			return nil, err
		}
		data, err := readData(reader, storage.DeleteEventType)
		if err != nil {
			return nil, err
		}
		for _, d := range data {
			dl := &storage.DeleteLog{}
			err = json.Unmarshal([]byte(d.(string)), dl)
			if err != nil {
				return nil, err
			}
			if dl.Ts >= tsStart && dl.Ts <= tsEnd {
				deleteData.Append(dl.Pk, dl.Ts)
			}
		}
	}
	return deleteData, nil
}

func (r *reader) Next(count int64) (*storage.InsertData, error) {
	insertData, err := storage.NewInsertData(r.schema)
	if err != nil {
		return nil, err
	}
	if r.readIdx == len(r.insertLogs[0]) {
		return nil, nil
	}
	for fieldID, binlogs := range r.insertLogs {
		field := typeutil.GetField(r.schema, fieldID)
		if field == nil {
			return nil, merr.WrapErrFieldNotFound(fieldID)
		}
		path := binlogs[r.readIdx]
		cr, err := newColumnReader(r.cm, field, path)
		if err != nil {
			return nil, err
		}
		fieldData, err := cr.Next(count)
		if err != nil {
			return nil, err
		}
		insertData.Data[field.GetFieldID()] = fieldData
	}
	insertData, err = r.Filter(insertData)
	if err != nil {
		return nil, err
	}
	r.readIdx++
	return insertData, nil
}

func (r *reader) Filter(insertData *storage.InsertData) (*storage.InsertData, error) {
	if len(r.filters) == 0 {
		return insertData, nil
	}
	result, err := storage.NewInsertData(r.schema)
	if err != nil {
		return nil, err
	}
	for i := 0; i < insertData.GetRowNum(); i++ {
		row := insertData.GetRow(i)
		for _, filter := range r.filters {
			if !filter(row) {
				continue
			}
		}
		err = result.Append(row)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}