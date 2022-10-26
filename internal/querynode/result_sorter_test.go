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

package querynode

import (
	"sort"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"

	"github.com/stretchr/testify/assert"
)

func TestResultSorter_ByIntPK(t *testing.T) {
	result := &segcorepb.RetrieveResults{
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{5, 4, 3, 2, 9, 8, 7, 6},
				}},
		},
		Offset: []int64{5, 4, 3, 2, 9, 8, 7, 6},
		FieldsData: []*schemapb.FieldData{
			genFieldData("int64 field", 100, schemapb.DataType_Int64,
				[]int64{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("double field", 101, schemapb.DataType_Double,
				[]float64{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("string field", 102, schemapb.DataType_VarChar,
				[]string{"5", "4", "3", "2", "9", "8", "7", "6"}, 1),
			genFieldData("bool field", 103, schemapb.DataType_Bool,
				[]bool{false, true, false, true, false, true, false, true}, 1),
			genFieldData("float field", 104, schemapb.DataType_Float,
				[]float32{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("int field", 105, schemapb.DataType_Int32,
				[]int32{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("float vector field", 106, schemapb.DataType_FloatVector,
				[]float32{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("binary vector field", 107, schemapb.DataType_BinaryVector,
				[]byte{5, 4, 3, 2, 9, 8, 7, 6}, 8),
		},
	}

	sort.Sort(&byPK{result})

	assert.Equal(t, []int64{2, 3, 4, 5, 6, 7, 8, 9}, result.GetIds().GetIntId().GetData())
	assert.Equal(t, []int64{2, 3, 4, 5, 6, 7, 8, 9}, result.GetOffset())
	assert.Equal(t, []int64{2, 3, 4, 5, 6, 7, 8, 9}, result.FieldsData[0].GetScalars().GetLongData().Data)
	assert.InDeltaSlice(t, []float64{2, 3, 4, 5, 6, 7, 8, 9}, result.FieldsData[1].GetScalars().GetDoubleData().Data, 10e-10)
	assert.Equal(t, []string{"2", "3", "4", "5", "6", "7", "8", "9"}, result.FieldsData[2].GetScalars().GetStringData().Data)
	assert.Equal(t, []bool{true, false, true, false, true, false, true, false}, result.FieldsData[3].GetScalars().GetBoolData().Data)
	assert.InDeltaSlice(t, []float32{2, 3, 4, 5, 6, 7, 8, 9}, result.FieldsData[4].GetScalars().GetFloatData().Data, 10e-10)
	assert.Equal(t, []int32{2, 3, 4, 5, 6, 7, 8, 9}, result.FieldsData[5].GetScalars().GetIntData().Data)
	assert.InDeltaSlice(t, []float32{2, 3, 4, 5, 6, 7, 8, 9}, result.FieldsData[6].GetVectors().GetFloatVector().GetData(), 10e-10)
	assert.Equal(t, []byte{2, 3, 4, 5, 6, 7, 8, 9}, result.FieldsData[7].GetVectors().GetBinaryVector())
}

func TestResultSorter_ByUint64TS(t *testing.T) {

	result := &segcorepb.RetrieveResults{
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: []int64{5, 4, 3, 2, 9, 8, 7, 6},
				}},
		},
		Offset: []int64{5, 4, 3, 2, 9, 8, 7, 6},
		FieldsData: []*schemapb.FieldData{
			genFieldData(common.TimeStampFieldName, common.TimeStampField, schemapb.DataType_Int64,
				[]int64{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("int64 field", 100, schemapb.DataType_Int64,
				[]int64{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("double field", 101, schemapb.DataType_Double,
				[]float64{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("string field", 102, schemapb.DataType_VarChar,
				[]string{"5", "4", "3", "2", "9", "8", "7", "6"}, 1),
			genFieldData("bool field", 103, schemapb.DataType_Bool,
				[]bool{false, true, false, true, false, true, false, true}, 1),
			genFieldData("float field", 104, schemapb.DataType_Float,
				[]float32{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("int field", 105, schemapb.DataType_Int32,
				[]int32{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("float vector field", 106, schemapb.DataType_FloatVector,
				[]float32{5, 4, 3, 2, 9, 8, 7, 6}, 1),
			genFieldData("binary vector field", 107, schemapb.DataType_BinaryVector,
				[]byte{5, 4, 3, 2, 9, 8, 7, 6}, 8),
		},
	}
	timeStamp := GetTimeStampFieldsData(result)
	sort.Sort(&byTS{result, timeStamp})

	assert.Equal(t, []int64{9, 8, 7, 6, 5, 4, 3, 2}, result.GetIds().GetIntId().GetData())
	assert.Equal(t, []int64{9, 8, 7, 6, 5, 4, 3, 2}, result.GetOffset())
	assert.Equal(t, []int64{9, 8, 7, 6, 5, 4, 3, 2}, result.FieldsData[1].GetScalars().GetLongData().Data)
	assert.InDeltaSlice(t, []float64{9, 8, 7, 6, 5, 4, 3, 2}, result.FieldsData[2].GetScalars().GetDoubleData().Data, 10e-10)
	assert.Equal(t, []string{"9", "8", "7", "6", "5", "4", "3", "2"}, result.FieldsData[3].GetScalars().GetStringData().Data)
	assert.Equal(t, []bool{false, true, false, true, false, true, false, true}, result.FieldsData[4].GetScalars().GetBoolData().Data)
	assert.InDeltaSlice(t, []float32{9, 8, 7, 6, 5, 4, 3, 2}, result.FieldsData[5].GetScalars().GetFloatData().Data, 10e-10)
	assert.Equal(t, []int32{9, 8, 7, 6, 5, 4, 3, 2}, result.FieldsData[6].GetScalars().GetIntData().Data)
	assert.InDeltaSlice(t, []float32{9, 8, 7, 6, 5, 4, 3, 2}, result.FieldsData[7].GetVectors().GetFloatVector().GetData(), 10e-10)
	assert.Equal(t, []byte{9, 8, 7, 6, 5, 4, 3, 2}, result.FieldsData[8].GetVectors().GetBinaryVector())
}
