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

package importutil

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path"
	"runtime/debug"
	"strconv"
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/schemapb"
	"github.com/milvus-io/milvus/internal/common"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

func isCanceled(ctx context.Context) bool {
	// canceled?
	select {
	case <-ctx.Done():
		return true
	default:
		break
	}
	return false
}

func initSegmentData(collectionSchema *schemapb.CollectionSchema) map[storage.FieldID]storage.FieldData {
	segmentData := make(map[storage.FieldID]storage.FieldData)
	// rowID field is a hidden field with fieldID=0, it is always auto-generated by IDAllocator
	// if primary key is int64 and autoID=true, primary key field is equal to rowID field
	segmentData[common.RowIDField] = &storage.Int64FieldData{
		Data: make([]int64, 0),
	}

	for i := 0; i < len(collectionSchema.Fields); i++ {
		schema := collectionSchema.Fields[i]
		switch schema.DataType {
		case schemapb.DataType_Bool:
			segmentData[schema.GetFieldID()] = &storage.BoolFieldData{
				Data: make([]bool, 0),
			}
		case schemapb.DataType_Float:
			segmentData[schema.GetFieldID()] = &storage.FloatFieldData{
				Data: make([]float32, 0),
			}
		case schemapb.DataType_Double:
			segmentData[schema.GetFieldID()] = &storage.DoubleFieldData{
				Data: make([]float64, 0),
			}
		case schemapb.DataType_Int8:
			segmentData[schema.GetFieldID()] = &storage.Int8FieldData{
				Data: make([]int8, 0),
			}
		case schemapb.DataType_Int16:
			segmentData[schema.GetFieldID()] = &storage.Int16FieldData{
				Data: make([]int16, 0),
			}
		case schemapb.DataType_Int32:
			segmentData[schema.GetFieldID()] = &storage.Int32FieldData{
				Data: make([]int32, 0),
			}
		case schemapb.DataType_Int64:
			segmentData[schema.GetFieldID()] = &storage.Int64FieldData{
				Data: make([]int64, 0),
			}
		case schemapb.DataType_BinaryVector:
			dim, _ := getFieldDimension(schema)
			segmentData[schema.GetFieldID()] = &storage.BinaryVectorFieldData{
				Data: make([]byte, 0),
				Dim:  dim,
			}
		case schemapb.DataType_FloatVector:
			dim, _ := getFieldDimension(schema)
			segmentData[schema.GetFieldID()] = &storage.FloatVectorFieldData{
				Data: make([]float32, 0),
				Dim:  dim,
			}
		case schemapb.DataType_String, schemapb.DataType_VarChar:
			segmentData[schema.GetFieldID()] = &storage.StringFieldData{
				Data: make([]string, 0),
			}
		default:
			log.Error("Import util: unsupported data type", zap.String("DataType", getTypeName(schema.DataType)))
			return nil
		}
	}

	return segmentData
}

func parseFloat(s string, bitsize int, fieldName string) (float64, error) {
	value, err := strconv.ParseFloat(s, bitsize)
	if err != nil {
		return 0, fmt.Errorf("failed to parse value '%s' for field '%s', error: %w", s, fieldName, err)
	}

	// not allow not-a-number and infinity
	if math.IsNaN(value) || math.IsInf(value, -1) || math.IsInf(value, 1) {
		return 0, fmt.Errorf("value '%s' is not a number or infinity, field '%s', error: %w", s, fieldName, err)
	}

	return value, nil
}

// initValidators constructs valiator methods and data conversion methods
func initValidators(collectionSchema *schemapb.CollectionSchema, validators map[storage.FieldID]*Validator) error {
	if collectionSchema == nil {
		return errors.New("collection schema is nil")
	}

	for i := 0; i < len(collectionSchema.Fields); i++ {
		schema := collectionSchema.Fields[i]

		validators[schema.GetFieldID()] = &Validator{}
		validators[schema.GetFieldID()].primaryKey = schema.GetIsPrimaryKey()
		validators[schema.GetFieldID()].autoID = schema.GetAutoID()
		validators[schema.GetFieldID()].fieldName = schema.GetName()
		validators[schema.GetFieldID()].isString = false

		switch schema.DataType {
		case schemapb.DataType_Bool:
			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if value, ok := obj.(bool); ok {
					field.(*storage.BoolFieldData).Data = append(field.(*storage.BoolFieldData).Data, value)
				} else {
					return fmt.Errorf("illegal value '%v' for bool type field '%s'", obj, schema.GetName())
				}

				return nil
			}
		case schemapb.DataType_Float:
			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if num, ok := obj.(json.Number); ok {
					value, err := parseFloat(string(num), 32, schema.GetName())
					if err != nil {
						return err
					}
					field.(*storage.FloatFieldData).Data = append(field.(*storage.FloatFieldData).Data, float32(value))
				} else {
					return fmt.Errorf("illegal value '%v' for float type field '%s'", obj, schema.GetName())
				}

				return nil
			}
		case schemapb.DataType_Double:
			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if num, ok := obj.(json.Number); ok {
					value, err := parseFloat(string(num), 64, schema.GetName())
					if err != nil {
						return err
					}
					field.(*storage.DoubleFieldData).Data = append(field.(*storage.DoubleFieldData).Data, value)
				} else {
					return fmt.Errorf("illegal value '%v' for double type field '%s'", obj, schema.GetName())
				}
				return nil
			}
		case schemapb.DataType_Int8:
			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if num, ok := obj.(json.Number); ok {
					value, err := strconv.ParseInt(string(num), 0, 8)
					if err != nil {
						return fmt.Errorf("failed to parse value '%v' for int8 field '%s', error: %w", num, schema.GetName(), err)
					}
					field.(*storage.Int8FieldData).Data = append(field.(*storage.Int8FieldData).Data, int8(value))
				} else {
					return fmt.Errorf("illegal value '%v' for int8 type field '%s'", obj, schema.GetName())
				}
				return nil
			}
		case schemapb.DataType_Int16:
			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if num, ok := obj.(json.Number); ok {
					value, err := strconv.ParseInt(string(num), 0, 16)
					if err != nil {
						return fmt.Errorf("failed to parse value '%v' for int16 field '%s', error: %w", num, schema.GetName(), err)
					}
					field.(*storage.Int16FieldData).Data = append(field.(*storage.Int16FieldData).Data, int16(value))
				} else {
					return fmt.Errorf("illegal value '%v' for int16 type field '%s'", obj, schema.GetName())
				}
				return nil
			}
		case schemapb.DataType_Int32:
			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if num, ok := obj.(json.Number); ok {
					value, err := strconv.ParseInt(string(num), 0, 32)
					if err != nil {
						return fmt.Errorf("failed to parse value '%v' for int32 field '%s', error: %w", num, schema.GetName(), err)
					}
					field.(*storage.Int32FieldData).Data = append(field.(*storage.Int32FieldData).Data, int32(value))
				} else {
					return fmt.Errorf("illegal value '%v' for int32 type field '%s'", obj, schema.GetName())
				}
				return nil
			}
		case schemapb.DataType_Int64:
			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if num, ok := obj.(json.Number); ok {
					value, err := strconv.ParseInt(string(num), 0, 64)
					if err != nil {
						return fmt.Errorf("failed to parse value '%v' for int64 field '%s', error: %w", num, schema.GetName(), err)
					}
					field.(*storage.Int64FieldData).Data = append(field.(*storage.Int64FieldData).Data, value)
				} else {
					return fmt.Errorf("illegal value '%v' for int64 type field '%s'", obj, schema.GetName())
				}
				return nil
			}
		case schemapb.DataType_BinaryVector:
			dim, err := getFieldDimension(schema)
			if err != nil {
				return err
			}
			validators[schema.GetFieldID()].dimension = dim

			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				arr, ok := obj.([]interface{})
				if !ok {
					return fmt.Errorf("'%v' is not an array for binary vector field '%s'", obj, schema.GetName())
				}
				// we use uint8 to represent binary vector in json file, each uint8 value represents 8 dimensions.
				if len(arr)*8 != dim {
					return fmt.Errorf("bit size %d doesn't equal to vector dimension %d of field '%s'", len(arr)*8, dim, schema.GetName())
				}

				for i := 0; i < len(arr); i++ {
					if num, ok := arr[i].(json.Number); ok {
						value, err := strconv.ParseUint(string(num), 0, 8)
						if err != nil {
							return fmt.Errorf("failed to parse value '%v' for binary vector field '%s', error: %w", num, schema.GetName(), err)
						}
						field.(*storage.BinaryVectorFieldData).Data = append(field.(*storage.BinaryVectorFieldData).Data, byte(value))
					} else {
						return fmt.Errorf("illegal value '%v' for binary vector field '%s'", obj, schema.GetName())
					}
				}

				return nil
			}
		case schemapb.DataType_FloatVector:
			dim, err := getFieldDimension(schema)
			if err != nil {
				return err
			}
			validators[schema.GetFieldID()].dimension = dim

			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				arr, ok := obj.([]interface{})
				if !ok {
					return fmt.Errorf("'%v' is not an array for float vector field '%s'", obj, schema.GetName())
				}
				if len(arr) != dim {
					return fmt.Errorf("array size %d doesn't equal to vector dimension %d of field '%s'", len(arr), dim, schema.GetName())
				}

				for i := 0; i < len(arr); i++ {
					if num, ok := arr[i].(json.Number); ok {
						value, err := parseFloat(string(num), 32, schema.GetName())
						if err != nil {
							return err
						}
						field.(*storage.FloatVectorFieldData).Data = append(field.(*storage.FloatVectorFieldData).Data, float32(value))
					} else {
						return fmt.Errorf("illegal value '%v' for float vector field '%s'", obj, schema.GetName())
					}
				}

				return nil
			}
		case schemapb.DataType_String, schemapb.DataType_VarChar:
			validators[schema.GetFieldID()].isString = true

			validators[schema.GetFieldID()].convertFunc = func(obj interface{}, field storage.FieldData) error {
				if value, ok := obj.(string); ok {
					field.(*storage.StringFieldData).Data = append(field.(*storage.StringFieldData).Data, value)
				} else {
					return fmt.Errorf("illegal value '%v' for varchar type field '%s'", obj, schema.GetName())
				}
				return nil
			}
		default:
			return fmt.Errorf("unsupport data type: %s", getTypeName(collectionSchema.Fields[i].DataType))
		}
	}

	return nil
}

func printFieldsDataInfo(fieldsData map[storage.FieldID]storage.FieldData, msg string, files []string) {
	stats := make([]zapcore.Field, 0)
	for k, v := range fieldsData {
		stats = append(stats, zap.Int(strconv.FormatInt(k, 10), v.RowNum()))
	}

	if len(files) > 0 {
		stats = append(stats, zap.Any(Files, files))
	}
	log.Info(msg, stats...)
}

// GetFileNameAndExt extracts file name and extension
// for example: "/a/b/c.ttt" returns "c" and ".ttt"
func GetFileNameAndExt(filePath string) (string, string) {
	fileName := path.Base(filePath)
	fileType := path.Ext(fileName)
	fileNameWithoutExt := strings.TrimSuffix(fileName, fileType)
	return fileNameWithoutExt, fileType
}

// getFieldDimension gets dimension of vecotor field
func getFieldDimension(schema *schemapb.FieldSchema) (int, error) {
	for _, kvPair := range schema.GetTypeParams() {
		key, value := kvPair.GetKey(), kvPair.GetValue()
		if key == "dim" {
			dim, err := strconv.Atoi(value)
			if err != nil {
				return 0, fmt.Errorf("illegal vector dimension '%s' for field '%s', error: %w", value, schema.GetName(), err)
			}
			return dim, nil
		}
	}

	return 0, fmt.Errorf("vector dimension is not defined for field '%s'", schema.GetName())
}

// triggerGC triggers golang gc to return all free memory back to the underlying system at once,
// Note: this operation is expensive, and can lead to latency spikes as it holds the heap lock through the whole process
func triggerGC() {
	debug.FreeOSMemory()
}

// tryFlushBlocks does the two things:
// 1. if accumulate data of a block exceed blockSize, call callFlushFunc to generate new binlog file
// 2. if total accumulate data exceed maxTotalSize, call callFlushFUnc to flush the biggest block
func tryFlushBlocks(ctx context.Context,
	blocksData []map[storage.FieldID]storage.FieldData,
	collectionSchema *schemapb.CollectionSchema,
	callFlushFunc ImportFlushFunc,
	blockSize int64,
	maxTotalSize int64,
	force bool) error {

	totalSize := 0
	biggestSize := 0
	biggestItem := -1

	// 1. if accumulate data of a block exceed blockSize, call callFlushFunc to generate new binlog file
	for i := 0; i < len(blocksData); i++ {
		// outside context might be canceled(service stop, or future enhancement for canceling import task)
		if isCanceled(ctx) {
			log.Error("Import util: import task was canceled")
			return errors.New("import task was canceled")
		}

		blockData := blocksData[i]
		// Note: even rowCount is 0, the size is still non-zero
		size := 0
		rowCount := 0
		for _, fieldData := range blockData {
			size += fieldData.GetMemorySize()
			rowCount = fieldData.RowNum()
		}

		// force to flush, called at the end of Read()
		if force && rowCount > 0 {
			printFieldsDataInfo(blockData, "import util: prepare to force flush a block", nil)
			err := callFlushFunc(blockData, i)
			if err != nil {
				log.Error("Import util: failed to force flush block data", zap.Int("shardID", i), zap.Error(err))
				return fmt.Errorf("failed to force flush block data for shard id %d, error: %w", i, err)
			}
			log.Info("Import util: force flush", zap.Int("rowCount", rowCount), zap.Int("size", size), zap.Int("shardID", i))

			blocksData[i] = initSegmentData(collectionSchema)
			if blocksData[i] == nil {
				log.Error("Import util: failed to initialize FieldData list", zap.Int("shardID", i))
				return fmt.Errorf("failed to initialize FieldData list for shard id %d", i)
			}
			continue
		}

		// if segment size is larger than predefined blockSize, flush to create a new binlog file
		// initialize a new FieldData list for next round batch read
		if size > int(blockSize) && rowCount > 0 {
			printFieldsDataInfo(blockData, "import util: prepare to flush block larger than blockSize", nil)
			err := callFlushFunc(blockData, i)
			if err != nil {
				log.Error("Import util: failed to flush block data", zap.Int("shardID", i), zap.Error(err))
				return fmt.Errorf("failed to flush block data for shard id %d, error: %w", i, err)
			}
			log.Info("Import util: block size exceed limit and flush", zap.Int("rowCount", rowCount),
				zap.Int("size", size), zap.Int("shardID", i), zap.Int64("blockSize", blockSize))

			blocksData[i] = initSegmentData(collectionSchema)
			if blocksData[i] == nil {
				log.Error("Import util: failed to initialize FieldData list", zap.Int("shardID", i))
				return fmt.Errorf("failed to initialize FieldData list for shard id %d", i)
			}
			continue
		}

		// calculate the total size(ignore the flushed blocks)
		// find out the biggest block for the step 2
		totalSize += size
		if size > biggestSize {
			biggestSize = size
			biggestItem = i
		}
	}

	// 2. if total accumulate data exceed maxTotalSize, call callFlushFUnc to flush the biggest block
	if totalSize > int(maxTotalSize) && biggestItem >= 0 {
		// outside context might be canceled(service stop, or future enhancement for canceling import task)
		if isCanceled(ctx) {
			log.Error("Import util: import task was canceled")
			return errors.New("import task was canceled")
		}

		blockData := blocksData[biggestItem]
		// Note: even rowCount is 0, the size is still non-zero
		size := 0
		rowCount := 0
		for _, fieldData := range blockData {
			size += fieldData.GetMemorySize()
			rowCount = fieldData.RowNum()
		}

		if rowCount > 0 {
			printFieldsDataInfo(blockData, "import util: prepare to flush biggest block", nil)
			err := callFlushFunc(blockData, biggestItem)
			if err != nil {
				log.Error("Import util: failed to flush biggest block data", zap.Int("shardID", biggestItem))
				return fmt.Errorf("failed to flush biggest block data for shard id %d, error: %w", biggestItem, err)
			}
			log.Info("Import util: total size exceed limit and flush", zap.Int("rowCount", rowCount),
				zap.Int("size", size), zap.Int("totalSize", totalSize), zap.Int("shardID", biggestItem))

			blocksData[biggestItem] = initSegmentData(collectionSchema)
			if blocksData[biggestItem] == nil {
				log.Error("Import util: failed to initialize FieldData list", zap.Int("shardID", biggestItem))
				return fmt.Errorf("failed to initialize FieldData list for shard id %d", biggestItem)
			}
		}
	}

	return nil
}

func getTypeName(dt schemapb.DataType) string {
	switch dt {
	case schemapb.DataType_Bool:
		return "Bool"
	case schemapb.DataType_Int8:
		return "Int8"
	case schemapb.DataType_Int16:
		return "Int16"
	case schemapb.DataType_Int32:
		return "Int32"
	case schemapb.DataType_Int64:
		return "Int64"
	case schemapb.DataType_Float:
		return "Float"
	case schemapb.DataType_Double:
		return "Double"
	case schemapb.DataType_VarChar:
		return "Varchar"
	case schemapb.DataType_String:
		return "String"
	case schemapb.DataType_BinaryVector:
		return "BinaryVector"
	case schemapb.DataType_FloatVector:
		return "FloatVector"
	default:
		return "InvalidType"
	}
}

func pkToShard(pk interface{}, shardNum uint32) (uint32, error) {
	var shard uint32
	strPK, ok := pk.(string)
	if ok {
		hash := typeutil.HashString2Uint32(strPK)
		shard = hash % shardNum
	} else {
		intPK, ok := pk.(int64)
		if !ok {
			log.Error("Numpy parser: primary key field must be int64 or varchar")
			return 0, fmt.Errorf("primary key field must be int64 or varchar")
		}
		hash, _ := typeutil.Hash32Int64(intPK)
		shard = hash % shardNum
	}

	return shard, nil
}

func UpdateKVInfo(infos *[]*commonpb.KeyValuePair, k string, v string) error {
	if infos == nil {
		return errors.New("Import util: kv array pointer is nil")
	}

	found := false
	for _, kv := range *infos {
		if kv.GetKey() == k {
			kv.Value = v
			found = true
		}
	}
	if !found {
		*infos = append(*infos, &commonpb.KeyValuePair{Key: k, Value: v})
	}

	return nil
}
