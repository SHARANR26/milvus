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

package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

const (
	// Ts is blob key "ts"
	Ts = "ts"
	// DDL is blob key "ddl"
	DDL = "ddl"
	// IndexParamsKey is blob key "indexParams"
	IndexParamsKey = "indexParams"
)

// when the blob of index file is too large, we can split blob into several rows,
// fortunately, the blob has no other semantics which differs from other binlog type,
// we then assemble these several rows into a whole blob when deserialize index binlog.
// num rows = math.Ceil(len(blob) / maxLengthPerRowOfIndexFile)
// There is only a string row in the past version index file which is a subset case of splitting into several rows.
// So splitting index file won't introduce incompatibility with past version.
const maxLengthPerRowOfIndexFile = 4 * 1024 * 1024

type (
	// UniqueID is type alias of typeutil.UniqueID
	UniqueID = typeutil.UniqueID

	// FieldID represent the identity number of field in collection and its type is UniqueID
	FieldID = typeutil.UniqueID

	// Timestamp is type alias of typeutil.Timestamp
	Timestamp = typeutil.Timestamp
)

// InvalidUniqueID is used when the UniqueID is not set (like in return with err)
const InvalidUniqueID = UniqueID(-1)

// Blob is a pack of key&value
type Blob struct {
	Key    string
	Value  []byte
	Size   int64
	RowNum int64
}

// BlobList implements sort.Interface for a list of Blob
type BlobList []*Blob

// Len implements Len in sort.Interface
func (s BlobList) Len() int {
	return len(s)
}

// Less implements Less in sort.Interface
func (s BlobList) Less(i, j int) bool {
	_, _, _, _, iLog, ok := metautil.ParseInsertLogPath(s[i].Key)
	if !ok {
		return false
	}
	_, _, _, _, jLog, ok := metautil.ParseInsertLogPath(s[j].Key)
	if !ok {
		return false
	}
	return iLog < jLog
}

// Swap implements Swap in sort.Interface
func (s BlobList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// GetKey returns the key of blob
func (b Blob) GetKey() string {
	return b.Key
}

// GetValue returns the value of blob
func (b Blob) GetValue() []byte {
	return b.Value
}

// InsertCodec serializes and deserializes the insert data
// Blob key example:
// ${tenant}/insert_log/${collection_id}/${partition_id}/${segment_id}/${field_id}/${log_idx}
type InsertCodec struct {
	Schema *etcdpb.CollectionMeta
}

// NewInsertCodec creates an InsertCodec
func NewInsertCodec() *InsertCodec {
	return &InsertCodec{}
}

// NewInsertCodecWithSchema creates an InsertCodec with provided collection meta
func NewInsertCodecWithSchema(schema *etcdpb.CollectionMeta) *InsertCodec {
	return &InsertCodec{Schema: schema}
}

// Serialize Pk stats log
func (insertCodec *InsertCodec) SerializePkStats(stats *PrimaryKeyStats, rowNum int64) (*Blob, error) {
	if stats == nil || stats.BF == nil {
		return nil, fmt.Errorf("sericalize empty pk stats")
	}

	// Serialize by pk stats
	blobKey := fmt.Sprintf("%d", stats.FieldID)
	statsWriter := &StatsWriter{}
	err := statsWriter.Generate(stats)
	if err != nil {
		return nil, err
	}

	buffer := statsWriter.GetBuffer()
	return &Blob{
		Key:    blobKey,
		Value:  buffer,
		RowNum: rowNum,
	}, nil
}

// Serialize Pk stats list to one blob
func (insertCodec *InsertCodec) SerializePkStatsList(stats []*PrimaryKeyStats, rowNum int64) (*Blob, error) {
	if len(stats) == 0 {
		return nil, merr.WrapErrServiceInternal("shall not serialize zero length statslog list")
	}

	blobKey := fmt.Sprintf("%d", stats[0].FieldID)
	statsWriter := &StatsWriter{}
	err := statsWriter.GenerateList(stats)
	if err != nil {
		return nil, err
	}

	buffer := statsWriter.GetBuffer()
	return &Blob{
		Key:    blobKey,
		Value:  buffer,
		RowNum: rowNum,
	}, nil
}

// Serialize Pk stats log by insert data
func (insertCodec *InsertCodec) SerializePkStatsByData(data *InsertData) (*Blob, error) {
	timeFieldData, ok := data.Data[common.TimeStampField]
	if !ok {
		return nil, fmt.Errorf("data doesn't contains timestamp field")
	}
	if timeFieldData.RowNum() <= 0 {
		return nil, fmt.Errorf("there's no data in InsertData")
	}
	rowNum := int64(timeFieldData.RowNum())

	for _, field := range insertCodec.Schema.Schema.Fields {
		// stats fields
		if !field.GetIsPrimaryKey() {
			continue
		}
		singleData := data.Data[field.FieldID]
		blobKey := fmt.Sprintf("%d", field.FieldID)
		statsWriter := &StatsWriter{}
		err := statsWriter.GenerateByData(field.FieldID, field.DataType, singleData)
		if err != nil {
			return nil, err
		}
		buffer := statsWriter.GetBuffer()
		return &Blob{
			Key:    blobKey,
			Value:  buffer,
			RowNum: rowNum,
		}, nil
	}
	return nil, fmt.Errorf("there is no pk field")
}

// Serialize transfer insert data to blob. It will sort insert data by timestamp.
// From schema, it gets all fields.
// For each field, it will create a binlog writer, and write an event to the binlog.
// It returns binlog buffer in the end.
func (insertCodec *InsertCodec) Serialize(partitionID UniqueID, segmentID UniqueID, data *InsertData) ([]*Blob, error) {
	blobs := make([]*Blob, 0)
	var writer *InsertBinlogWriter
	if insertCodec.Schema == nil {
		return nil, fmt.Errorf("schema is not set")
	}
	timeFieldData, ok := data.Data[common.TimeStampField]
	if !ok {
		return nil, fmt.Errorf("data doesn't contains timestamp field")
	}
	if timeFieldData.RowNum() <= 0 {
		return nil, fmt.Errorf("there's no data in InsertData")
	}
	rowNum := int64(timeFieldData.RowNum())

	ts := timeFieldData.(*Int64FieldData).Data
	var startTs, endTs Timestamp
	startTs, endTs = math.MaxUint64, 0
	for _, t := range ts {
		if uint64(t) > endTs {
			endTs = uint64(t)
		}

		if uint64(t) < startTs {
			startTs = uint64(t)
		}
	}

	// sort insert data by rowID
	dataSorter := &DataSorter{
		InsertCodec: insertCodec,
		InsertData:  data,
	}
	sort.Sort(dataSorter)

	for _, field := range insertCodec.Schema.Schema.Fields {
		singleData := data.Data[field.FieldID]

		// encode fields
		writer = NewInsertBinlogWriter(field.DataType, insertCodec.Schema.ID, partitionID, segmentID, field.FieldID)
		var eventWriter *insertEventWriter
		var err error
		if typeutil.IsVectorType(field.DataType) {
			switch field.DataType {
			case schemapb.DataType_FloatVector:
				eventWriter, err = writer.NextInsertEventWriter(singleData.(*FloatVectorFieldData).Dim)
			case schemapb.DataType_BinaryVector:
				eventWriter, err = writer.NextInsertEventWriter(singleData.(*BinaryVectorFieldData).Dim)
			case schemapb.DataType_Float16Vector:
				eventWriter, err = writer.NextInsertEventWriter(singleData.(*Float16VectorFieldData).Dim)
			case schemapb.DataType_BFloat16Vector:
				eventWriter, err = writer.NextInsertEventWriter(singleData.(*BFloat16VectorFieldData).Dim)
			default:
				return nil, fmt.Errorf("undefined data type %d", field.DataType)
			}
		} else {
			eventWriter, err = writer.NextInsertEventWriter()
		}
		if err != nil {
			writer.Close()
			return nil, err
		}

		eventWriter.SetEventTimestamp(startTs, endTs)
		switch field.DataType {
		case schemapb.DataType_Bool:
			err = eventWriter.AddBoolToPayload(singleData.(*BoolFieldData).Data)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*BoolFieldData).GetMemorySize()))
		case schemapb.DataType_Int8:
			err = eventWriter.AddInt8ToPayload(singleData.(*Int8FieldData).Data)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*Int8FieldData).GetMemorySize()))
		case schemapb.DataType_Int16:
			err = eventWriter.AddInt16ToPayload(singleData.(*Int16FieldData).Data)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*Int16FieldData).GetMemorySize()))
		case schemapb.DataType_Int32:
			err = eventWriter.AddInt32ToPayload(singleData.(*Int32FieldData).Data)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*Int32FieldData).GetMemorySize()))
		case schemapb.DataType_Int64:
			err = eventWriter.AddInt64ToPayload(singleData.(*Int64FieldData).Data)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*Int64FieldData).GetMemorySize()))
		case schemapb.DataType_Float:
			err = eventWriter.AddFloatToPayload(singleData.(*FloatFieldData).Data)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*FloatFieldData).GetMemorySize()))
		case schemapb.DataType_Double:
			err = eventWriter.AddDoubleToPayload(singleData.(*DoubleFieldData).Data)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*DoubleFieldData).GetMemorySize()))
		case schemapb.DataType_String, schemapb.DataType_VarChar:
			for _, singleString := range singleData.(*StringFieldData).Data {
				err = eventWriter.AddOneStringToPayload(singleString)
				if err != nil {
					eventWriter.Close()
					writer.Close()
					return nil, err
				}
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*StringFieldData).GetMemorySize()))
		case schemapb.DataType_Array:
			for _, singleArray := range singleData.(*ArrayFieldData).Data {
				err = eventWriter.AddOneArrayToPayload(singleArray)
				if err != nil {
					eventWriter.Close()
					writer.Close()
					return nil, err
				}
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*ArrayFieldData).GetMemorySize()))
		case schemapb.DataType_JSON:
			for _, singleJSON := range singleData.(*JSONFieldData).Data {
				err = eventWriter.AddOneJSONToPayload(singleJSON)
				if err != nil {
					eventWriter.Close()
					writer.Close()
					return nil, err
				}
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*JSONFieldData).GetMemorySize()))
		case schemapb.DataType_BinaryVector:
			err = eventWriter.AddBinaryVectorToPayload(singleData.(*BinaryVectorFieldData).Data, singleData.(*BinaryVectorFieldData).Dim)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*BinaryVectorFieldData).GetMemorySize()))
		case schemapb.DataType_FloatVector:
			err = eventWriter.AddFloatVectorToPayload(singleData.(*FloatVectorFieldData).Data, singleData.(*FloatVectorFieldData).Dim)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*FloatVectorFieldData).GetMemorySize()))
		case schemapb.DataType_Float16Vector:
			err = eventWriter.AddFloat16VectorToPayload(singleData.(*Float16VectorFieldData).Data, singleData.(*Float16VectorFieldData).Dim)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*Float16VectorFieldData).GetMemorySize()))
		case schemapb.DataType_BFloat16Vector:
			err = eventWriter.AddBFloat16VectorToPayload(singleData.(*BFloat16VectorFieldData).Data, singleData.(*BFloat16VectorFieldData).Dim)
			if err != nil {
				eventWriter.Close()
				writer.Close()
				return nil, err
			}
			writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", singleData.(*BFloat16VectorFieldData).GetMemorySize()))
		default:
			return nil, fmt.Errorf("undefined data type %d", field.DataType)
		}
		if err != nil {
			return nil, err
		}
		writer.SetEventTimeStamp(startTs, endTs)

		err = writer.Finish()
		if err != nil {
			eventWriter.Close()
			writer.Close()
			return nil, err
		}

		buffer, err := writer.GetBuffer()
		if err != nil {
			eventWriter.Close()
			writer.Close()
			return nil, err
		}
		blobKey := metautil.BuildInsertLogPath(
			"$ROOTPATH",
			0,
			partitionID,
			segmentID,
			field.FieldID,
			0,
		)
		blobs = append(blobs, &Blob{
			Key:    blobKey,
			Value:  buffer,
			RowNum: rowNum,
		})
		eventWriter.Close()
		writer.Close()
	}

	return blobs, nil
}

func (insertCodec *InsertCodec) DeserializeAll(blobs []*Blob) (
	collectionID UniqueID,
	partitionID UniqueID,
	segmentID UniqueID,
	data *InsertData,
	err error,
) {
	if len(blobs) == 0 {
		return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, nil, fmt.Errorf("blobs is empty")
	}

	var blobList BlobList = blobs
	sort.Sort(blobList)

	data = &InsertData{
		Data: make(map[FieldID]FieldData),
	}
	if collectionID, partitionID, segmentID, err = insertCodec.DeserializeInto(blobs, 0, data); err != nil {
		return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, nil, err
	}

	return
}

func (insertCodec *InsertCodec) DeserializeInto(fieldBinlogs []*Blob, rowNum int, insertData *InsertData) (
	collectionID UniqueID,
	partitionID UniqueID,
	segmentID UniqueID,
	err error,
) {
	for _, blob := range fieldBinlogs {
		binlogReader, err := NewBinlogReader(blob.Value)
		if err != nil {
			return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
		}

		// read partitionID and SegmentID
		collectionID, partitionID, segmentID = binlogReader.CollectionID, binlogReader.PartitionID, binlogReader.SegmentID

		dataType := binlogReader.PayloadDataType
		fieldID := binlogReader.FieldID
		totalLength := 0
		dim := 0

		for {
			eventReader, err := binlogReader.NextEventReader()
			if err != nil {
				return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
			}
			if eventReader == nil {
				break
			}
			switch dataType {
			case schemapb.DataType_Bool:
				singleData, err := eventReader.GetBoolFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &BoolFieldData{
						Data: make([]bool, 0, rowNum),
					}
				}
				boolFieldData := insertData.Data[fieldID].(*BoolFieldData)

				boolFieldData.Data = append(boolFieldData.Data, singleData...)
				totalLength += len(singleData)
				insertData.Data[fieldID] = boolFieldData

			case schemapb.DataType_Int8:
				singleData, err := eventReader.GetInt8FromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &Int8FieldData{
						Data: make([]int8, 0, rowNum),
					}
				}
				int8FieldData := insertData.Data[fieldID].(*Int8FieldData)

				int8FieldData.Data = append(int8FieldData.Data, singleData...)
				totalLength += len(singleData)
				insertData.Data[fieldID] = int8FieldData

			case schemapb.DataType_Int16:
				singleData, err := eventReader.GetInt16FromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &Int16FieldData{
						Data: make([]int16, 0, rowNum),
					}
				}
				int16FieldData := insertData.Data[fieldID].(*Int16FieldData)

				int16FieldData.Data = append(int16FieldData.Data, singleData...)
				totalLength += len(singleData)
				insertData.Data[fieldID] = int16FieldData

			case schemapb.DataType_Int32:
				singleData, err := eventReader.GetInt32FromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &Int32FieldData{
						Data: make([]int32, 0, rowNum),
					}
				}
				int32FieldData := insertData.Data[fieldID].(*Int32FieldData)

				int32FieldData.Data = append(int32FieldData.Data, singleData...)
				totalLength += len(singleData)
				insertData.Data[fieldID] = int32FieldData

			case schemapb.DataType_Int64:
				singleData, err := eventReader.GetInt64FromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &Int64FieldData{
						Data: make([]int64, 0, rowNum),
					}
				}
				int64FieldData := insertData.Data[fieldID].(*Int64FieldData)

				int64FieldData.Data = append(int64FieldData.Data, singleData...)
				totalLength += len(singleData)
				insertData.Data[fieldID] = int64FieldData

			case schemapb.DataType_Float:
				singleData, err := eventReader.GetFloatFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &FloatFieldData{
						Data: make([]float32, 0, rowNum),
					}
				}
				floatFieldData := insertData.Data[fieldID].(*FloatFieldData)

				floatFieldData.Data = append(floatFieldData.Data, singleData...)
				totalLength += len(singleData)
				insertData.Data[fieldID] = floatFieldData

			case schemapb.DataType_Double:
				singleData, err := eventReader.GetDoubleFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &DoubleFieldData{
						Data: make([]float64, 0, rowNum),
					}
				}
				doubleFieldData := insertData.Data[fieldID].(*DoubleFieldData)

				doubleFieldData.Data = append(doubleFieldData.Data, singleData...)
				totalLength += len(singleData)
				insertData.Data[fieldID] = doubleFieldData

			case schemapb.DataType_String, schemapb.DataType_VarChar:
				stringPayload, err := eventReader.GetStringFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &StringFieldData{
						Data: make([]string, 0, rowNum),
					}
				}
				stringFieldData := insertData.Data[fieldID].(*StringFieldData)

				stringFieldData.Data = append(stringFieldData.Data, stringPayload...)
				stringFieldData.DataType = dataType
				totalLength += len(stringPayload)
				insertData.Data[fieldID] = stringFieldData

			case schemapb.DataType_Array:
				arrayPayload, err := eventReader.GetArrayFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &ArrayFieldData{
						Data: make([]*schemapb.ScalarField, 0, rowNum),
					}
				}
				arrayFieldData := insertData.Data[fieldID].(*ArrayFieldData)

				arrayFieldData.Data = append(arrayFieldData.Data, arrayPayload...)
				totalLength += len(arrayPayload)
				insertData.Data[fieldID] = arrayFieldData

			case schemapb.DataType_JSON:
				jsonPayload, err := eventReader.GetJSONFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &JSONFieldData{
						Data: make([][]byte, 0, rowNum),
					}
				}
				jsonFieldData := insertData.Data[fieldID].(*JSONFieldData)

				jsonFieldData.Data = append(jsonFieldData.Data, jsonPayload...)
				totalLength += len(jsonPayload)
				insertData.Data[fieldID] = jsonFieldData

			case schemapb.DataType_BinaryVector:
				var singleData []byte
				singleData, dim, err = eventReader.GetBinaryVectorFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &BinaryVectorFieldData{
						Data: make([]byte, 0, rowNum*dim),
					}
				}
				binaryVectorFieldData := insertData.Data[fieldID].(*BinaryVectorFieldData)

				binaryVectorFieldData.Data = append(binaryVectorFieldData.Data, singleData...)
				length, err := eventReader.GetPayloadLengthFromReader()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}
				totalLength += length
				binaryVectorFieldData.Dim = dim
				insertData.Data[fieldID] = binaryVectorFieldData

			case schemapb.DataType_Float16Vector:
				var singleData []byte
				singleData, dim, err = eventReader.GetFloat16VectorFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &Float16VectorFieldData{
						Data: make([]byte, 0, rowNum*dim),
					}
				}
				float16VectorFieldData := insertData.Data[fieldID].(*Float16VectorFieldData)

				float16VectorFieldData.Data = append(float16VectorFieldData.Data, singleData...)
				length, err := eventReader.GetPayloadLengthFromReader()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}
				totalLength += length
				float16VectorFieldData.Dim = dim
				insertData.Data[fieldID] = float16VectorFieldData

			case schemapb.DataType_BFloat16Vector:
				var singleData []byte
				singleData, dim, err = eventReader.GetBFloat16VectorFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &BFloat16VectorFieldData{
						Data: make([]byte, 0, rowNum*dim),
					}
				}
				bfloat16VectorFieldData := insertData.Data[fieldID].(*BFloat16VectorFieldData)

				bfloat16VectorFieldData.Data = append(bfloat16VectorFieldData.Data, singleData...)
				length, err := eventReader.GetPayloadLengthFromReader()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}
				totalLength += length
				bfloat16VectorFieldData.Dim = dim
				insertData.Data[fieldID] = bfloat16VectorFieldData

			case schemapb.DataType_FloatVector:
				var singleData []float32
				singleData, dim, err = eventReader.GetFloatVectorFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}

				if insertData.Data[fieldID] == nil {
					insertData.Data[fieldID] = &FloatVectorFieldData{
						Data: make([]float32, 0, rowNum*dim),
					}
				}
				floatVectorFieldData := insertData.Data[fieldID].(*FloatVectorFieldData)

				floatVectorFieldData.Data = append(floatVectorFieldData.Data, singleData...)
				length, err := eventReader.GetPayloadLengthFromReader()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, err
				}
				totalLength += length
				floatVectorFieldData.Dim = dim
				insertData.Data[fieldID] = floatVectorFieldData

			default:
				eventReader.Close()
				binlogReader.Close()
				return InvalidUniqueID, InvalidUniqueID, InvalidUniqueID, fmt.Errorf("undefined data type %d", dataType)
			}
			eventReader.Close()
		}

		if rowNum <= 0 {
			rowNum = totalLength
		}

		if fieldID == common.TimeStampField {
			blobInfo := BlobInfo{
				Length: totalLength,
			}
			insertData.Infos = append(insertData.Infos, blobInfo)
		}
		binlogReader.Close()
	}

	return collectionID, partitionID, segmentID, nil
}

// func deserializeEntity[T any, U any](
// 	eventReader *EventReader,
// 	binlogReader *BinlogReader,
// 	insertData *InsertData,
// 	getPayloadFunc func() (U, error),
// 	fillDataFunc func() FieldData,
// ) error {
// 	fieldID := binlogReader.FieldID
// 	stringPayload, err := getPayloadFunc()
// 	if err != nil {
// 		eventReader.Close()
// 		binlogReader.Close()
// 		return err
// 	}
//
// 	if insertData.Data[fieldID] == nil {
// 		insertData.Data[fieldID] = fillDataFunc()
// 	}
// 	stringFieldData := insertData.Data[fieldID].(*T)
//
// 	stringFieldData.Data = append(stringFieldData.Data, stringPayload...)
// 	totalLength += len(stringPayload)
// 	insertData.Data[fieldID] = stringFieldData
// }

// Deserialize transfer blob back to insert data.
// From schema, it get all fields.
// For each field, it will create a binlog reader, and read all event to the buffer.
// It returns origin @InsertData in the end.
func (insertCodec *InsertCodec) Deserialize(blobs []*Blob) (partitionID UniqueID, segmentID UniqueID, data *InsertData, err error) {
	_, partitionID, segmentID, data, err = insertCodec.DeserializeAll(blobs)
	return partitionID, segmentID, data, err
}

type DeleteLog struct {
	Pk     PrimaryKey `json:"pk"`
	Ts     uint64     `json:"ts"`
	PkType int64      `json:"pkType"`
}

func NewDeleteLog(pk PrimaryKey, ts Timestamp) *DeleteLog {
	pkType := pk.Type()

	return &DeleteLog{
		Pk:     pk,
		Ts:     ts,
		PkType: int64(pkType),
	}
}

func (dl *DeleteLog) UnmarshalJSON(data []byte) error {
	var messageMap map[string]*json.RawMessage
	err := json.Unmarshal(data, &messageMap)
	if err != nil {
		return err
	}

	err = json.Unmarshal(*messageMap["pkType"], &dl.PkType)
	if err != nil {
		return err
	}

	switch schemapb.DataType(dl.PkType) {
	case schemapb.DataType_Int64:
		dl.Pk = &Int64PrimaryKey{}
	case schemapb.DataType_VarChar:
		dl.Pk = &VarCharPrimaryKey{}
	}

	err = json.Unmarshal(*messageMap["pk"], dl.Pk)
	if err != nil {
		return err
	}

	err = json.Unmarshal(*messageMap["ts"], &dl.Ts)
	if err != nil {
		return err
	}

	return nil
}

// DeleteData saves each entity delete message represented as <primarykey,timestamp> map.
// timestamp represents the time when this instance was deleted
type DeleteData struct {
	Pks      []PrimaryKey // primary keys
	Tss      []Timestamp  // timestamps
	RowCount int64
	memSize  int64
}

func NewDeleteData(pks []PrimaryKey, tss []Timestamp) *DeleteData {
	return &DeleteData{
		Pks:      pks,
		Tss:      tss,
		RowCount: int64(len(pks)),
		memSize:  lo.SumBy(pks, func(pk PrimaryKey) int64 { return pk.Size() }) + int64(len(tss)*8),
	}
}

// Append append 1 pk&ts pair to DeleteData
func (data *DeleteData) Append(pk PrimaryKey, ts Timestamp) {
	data.Pks = append(data.Pks, pk)
	data.Tss = append(data.Tss, ts)
	data.RowCount++
	data.memSize += pk.Size() + int64(8)
}

// Append append 1 pk&ts pair to DeleteData
func (data *DeleteData) AppendBatch(pks []PrimaryKey, tss []Timestamp) {
	data.Pks = append(data.Pks, pks...)
	data.Tss = append(data.Tss, tss...)
	data.RowCount += int64(len(pks))
	data.memSize += lo.SumBy(pks, func(pk PrimaryKey) int64 { return pk.Size() }) + int64(len(tss)*8)
}

func (data *DeleteData) Merge(other *DeleteData) {
	data.Pks = append(other.Pks, other.Pks...)
	data.Tss = append(other.Tss, other.Tss...)
	data.RowCount += other.RowCount
	data.memSize += other.Size()

	other.Pks = nil
	other.Tss = nil
	other.RowCount = 0
	other.memSize = 0
}

func (data *DeleteData) Size() int64 {
	return data.memSize
}

// DeleteCodec serializes and deserializes the delete data
type DeleteCodec struct{}

// NewDeleteCodec returns a DeleteCodec
func NewDeleteCodec() *DeleteCodec {
	return &DeleteCodec{}
}

// Serialize transfer delete data to blob. .
// For each delete message, it will save "pk,ts" string to binlog.
func (deleteCodec *DeleteCodec) Serialize(collectionID UniqueID, partitionID UniqueID, segmentID UniqueID, data *DeleteData) (*Blob, error) {
	binlogWriter := NewDeleteBinlogWriter(schemapb.DataType_String, collectionID, partitionID, segmentID)
	eventWriter, err := binlogWriter.NextDeleteEventWriter()
	if err != nil {
		binlogWriter.Close()
		return nil, err
	}
	defer binlogWriter.Close()
	defer eventWriter.Close()
	length := len(data.Pks)
	if length != len(data.Tss) {
		return nil, fmt.Errorf("the length of pks, and TimeStamps is not equal")
	}

	sizeTotal := 0
	var startTs, endTs Timestamp
	startTs, endTs = math.MaxUint64, 0
	for i := 0; i < length; i++ {
		ts := data.Tss[i]
		if ts < startTs {
			startTs = ts
		}
		if ts > endTs {
			endTs = ts
		}

		deleteLog := NewDeleteLog(data.Pks[i], ts)
		serializedPayload, err := json.Marshal(deleteLog)
		if err != nil {
			return nil, err
		}
		err = eventWriter.AddOneStringToPayload(string(serializedPayload))
		if err != nil {
			return nil, err
		}
		sizeTotal += binary.Size(serializedPayload)
	}
	eventWriter.SetEventTimestamp(startTs, endTs)
	binlogWriter.SetEventTimeStamp(startTs, endTs)

	// https://github.com/milvus-io/milvus/issues/9620
	// It's a little complicated to count the memory size of a map.
	// See: https://stackoverflow.com/questions/31847549/computing-the-memory-footprint-or-byte-length-of-a-map
	// Since the implementation of golang map may differ from version, so we'd better not to use this magic method.
	binlogWriter.AddExtra(originalSizeKey, fmt.Sprintf("%v", sizeTotal))

	err = binlogWriter.Finish()
	if err != nil {
		return nil, err
	}
	buffer, err := binlogWriter.GetBuffer()
	if err != nil {
		return nil, err
	}
	blob := &Blob{
		Value: buffer,
	}
	return blob, nil
}

// Deserialize deserializes the deltalog blobs into DeleteData
func (deleteCodec *DeleteCodec) Deserialize(blobs []*Blob) (partitionID UniqueID, segmentID UniqueID, data *DeleteData, err error) {
	if len(blobs) == 0 {
		return InvalidUniqueID, InvalidUniqueID, nil, fmt.Errorf("blobs is empty")
	}

	var pid, sid UniqueID
	result := &DeleteData{}

	deserializeBlob := func(blob *Blob) error {
		binlogReader, err := NewBinlogReader(blob.Value)
		if err != nil {
			return err
		}
		defer binlogReader.Close()

		pid, sid = binlogReader.PartitionID, binlogReader.SegmentID
		eventReader, err := binlogReader.NextEventReader()
		if err != nil {
			return err
		}
		defer eventReader.Close()

		rr, err := eventReader.GetArrowRecordReader()
		if err != nil {
			return err
		}
		defer rr.Release()

		for rr.Next() {
			rec := rr.Record()
			defer rec.Release()
			column := rec.Column(0)
			for i := 0; i < column.Len(); i++ {
				deleteLog := &DeleteLog{}
				strVal := column.ValueStr(i)
				if err = json.Unmarshal([]byte(strVal), deleteLog); err != nil {
					// compatible with versions that only support int64 type primary keys
					// compatible with fmt.Sprintf("%d,%d", pk, ts)
					// compatible error info (unmarshal err invalid character ',' after top-level value)
					splits := strings.Split(strVal, ",")
					if len(splits) != 2 {
						return fmt.Errorf("the format of delta log is incorrect, %v can not be split", strVal)
					}
					pk, err := strconv.ParseInt(splits[0], 10, 64)
					if err != nil {
						return err
					}
					deleteLog.Pk = &Int64PrimaryKey{
						Value: pk,
					}
					deleteLog.PkType = int64(schemapb.DataType_Int64)
					deleteLog.Ts, err = strconv.ParseUint(splits[1], 10, 64)
					if err != nil {
						return err
					}
				}

				result.Append(deleteLog.Pk, deleteLog.Ts)
			}
		}
		return nil
	}

	for _, blob := range blobs {
		if err := deserializeBlob(blob); err != nil {
			return InvalidUniqueID, InvalidUniqueID, nil, err
		}
	}

	return pid, sid, result, nil
}

// DataDefinitionCodec serializes and deserializes the data definition
// Blob key example:
// ${tenant}/data_definition_log/${collection_id}/ts/${log_idx}
// ${tenant}/data_definition_log/${collection_id}/ddl/${log_idx}
type DataDefinitionCodec struct {
	collectionID int64
}

// NewDataDefinitionCodec is constructor for DataDefinitionCodec
func NewDataDefinitionCodec(collectionID int64) *DataDefinitionCodec {
	return &DataDefinitionCodec{collectionID: collectionID}
}

// Serialize transfer @ts and @ddRequsts to blob.
// From schema, it get all fields.
// For each field, it will create a binlog writer, and write specific event according
// to the dataDefinition type.
// It returns blobs in the end.
func (dataDefinitionCodec *DataDefinitionCodec) Serialize(ts []Timestamp, ddRequests []string, eventTypes []EventTypeCode) ([]*Blob, error) {
	writer := NewDDLBinlogWriter(schemapb.DataType_Int64, dataDefinitionCodec.collectionID)
	eventWriter, err := writer.NextCreateCollectionEventWriter()
	if err != nil {
		writer.Close()
		return nil, err
	}
	defer writer.Close()
	defer eventWriter.Close()

	var blobs []*Blob

	var int64Ts []int64
	for _, singleTs := range ts {
		int64Ts = append(int64Ts, int64(singleTs))
	}
	err = eventWriter.AddInt64ToPayload(int64Ts)
	if err != nil {
		return nil, err
	}
	eventWriter.SetEventTimestamp(ts[0], ts[len(ts)-1])
	writer.SetEventTimeStamp(ts[0], ts[len(ts)-1])

	// https://github.com/milvus-io/milvus/issues/9620
	writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", binary.Size(int64Ts)))

	err = writer.Finish()
	if err != nil {
		return nil, err
	}
	buffer, err := writer.GetBuffer()
	if err != nil {
		return nil, err
	}
	blobs = append(blobs, &Blob{
		Key:   Ts,
		Value: buffer,
	})
	eventWriter.Close()
	writer.Close()

	writer = NewDDLBinlogWriter(schemapb.DataType_String, dataDefinitionCodec.collectionID)

	sizeTotal := 0
	for pos, req := range ddRequests {
		sizeTotal += len(req)
		switch eventTypes[pos] {
		case CreateCollectionEventType:
			eventWriter, err := writer.NextCreateCollectionEventWriter()
			if err != nil {
				return nil, err
			}
			err = eventWriter.AddOneStringToPayload(req)
			if err != nil {
				return nil, err
			}
			eventWriter.SetEventTimestamp(ts[pos], ts[pos])
		case DropCollectionEventType:
			eventWriter, err := writer.NextDropCollectionEventWriter()
			if err != nil {
				return nil, err
			}
			err = eventWriter.AddOneStringToPayload(req)
			if err != nil {
				return nil, err
			}
			eventWriter.SetEventTimestamp(ts[pos], ts[pos])
		case CreatePartitionEventType:
			eventWriter, err := writer.NextCreatePartitionEventWriter()
			if err != nil {
				return nil, err
			}
			err = eventWriter.AddOneStringToPayload(req)
			if err != nil {
				return nil, err
			}
			eventWriter.SetEventTimestamp(ts[pos], ts[pos])
		case DropPartitionEventType:
			eventWriter, err := writer.NextDropPartitionEventWriter()
			if err != nil {
				return nil, err
			}
			err = eventWriter.AddOneStringToPayload(req)
			if err != nil {
				return nil, err
			}
			eventWriter.SetEventTimestamp(ts[pos], ts[pos])
		}
	}
	writer.SetEventTimeStamp(ts[0], ts[len(ts)-1])

	// https://github.com/milvus-io/milvus/issues/9620
	writer.AddExtra(originalSizeKey, fmt.Sprintf("%v", sizeTotal))

	err = writer.Finish()
	if err != nil {
		return nil, err
	}
	buffer, err = writer.GetBuffer()
	if err != nil {
		return nil, err
	}
	blobs = append(blobs, &Blob{
		Key:   DDL,
		Value: buffer,
	})

	return blobs, nil
}

// Deserialize transfer blob back to data definition data.
// From schema, it get all fields.
// It will sort blob by blob key for blob logid is increasing by time.
// For each field, it will create a binlog reader, and read all event to the buffer.
// It returns origin @ts and @ddRequests in the end.
func (dataDefinitionCodec *DataDefinitionCodec) Deserialize(blobs []*Blob) (ts []Timestamp, ddRequests []string, err error) {
	if len(blobs) == 0 {
		return nil, nil, fmt.Errorf("blobs is empty")
	}
	var requestsStrings []string
	var resultTs []Timestamp

	var blobList BlobList = blobs
	sort.Sort(blobList)

	for _, blob := range blobList {
		binlogReader, err := NewBinlogReader(blob.Value)
		if err != nil {
			return nil, nil, err
		}
		dataType := binlogReader.PayloadDataType

		for {
			eventReader, err := binlogReader.NextEventReader()
			if err != nil {
				binlogReader.Close()
				return nil, nil, err
			}
			if eventReader == nil {
				break
			}
			switch dataType {
			case schemapb.DataType_Int64:
				int64Ts, err := eventReader.GetInt64FromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return nil, nil, err
				}
				for _, singleTs := range int64Ts {
					resultTs = append(resultTs, Timestamp(singleTs))
				}
			case schemapb.DataType_String:
				stringPayload, err := eventReader.GetStringFromPayload()
				if err != nil {
					eventReader.Close()
					binlogReader.Close()
					return nil, nil, err
				}
				requestsStrings = append(requestsStrings, stringPayload...)
			}
			eventReader.Close()
		}
		binlogReader.Close()
	}

	return resultTs, requestsStrings, nil
}
