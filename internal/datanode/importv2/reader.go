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
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
)

type Reader interface {
	ReadStats() (*datapb.ImportFileStats, error)
	Next(count int64) (*storage.InsertData, error)
	Close()
}

func NewReader(cm storage.ChunkManager,
	schema *schemapb.CollectionSchema,
	importFile *datapb.ImportFile,
) Reader {
	return nil
}

type Handler interface {
	Hash(*storage.InsertData) map[string]map[int64]*storage.InsertData // vchannel -> {partitionID -> InsertData}
}