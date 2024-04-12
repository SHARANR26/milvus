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

package indexnode

import (
	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

func estimateFieldDataSize(dim int64, numRows int64, dataType schemapb.DataType) (uint64, error) {
	switch dataType {
	case schemapb.DataType_BinaryVector:
		return uint64(dim) / 8 * uint64(numRows), nil
	case schemapb.DataType_FloatVector:
		return uint64(dim) * uint64(numRows) * 4, nil
	case schemapb.DataType_Float16Vector, schemapb.DataType_BFloat16Vector:
		return uint64(dim) * uint64(numRows) * 2, nil
	case schemapb.DataType_SparseFloatVector:
		return 0, errors.New("could not estimate field data size of SparseFloatVector")
	default:
		return 0, nil
	}
}
