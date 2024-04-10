package writebuffer

import (
	"fmt"
	"math"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

const (
	noLimit int64 = -1
)

type BufferBase struct {
	rows      int64
	rowLimit  int64
	size      int64
	sizeLimit int64

	TimestampFrom typeutil.Timestamp
	TimestampTo   typeutil.Timestamp

	startPos *msgpb.MsgPosition
	endPos   *msgpb.MsgPosition
}

func (b *BufferBase) UpdateStatistics(entryNum, size int64, tr TimeRange, startPos, endPos *msgpb.MsgPosition) {
	b.rows += entryNum
	b.size += size

	if tr.timestampMin < b.TimestampFrom {
		b.TimestampFrom = tr.timestampMin
	}
	if tr.timestampMax > b.TimestampTo {
		b.TimestampTo = tr.timestampMax
	}

	if b.startPos == nil || startPos.Timestamp < b.startPos.Timestamp {
		b.startPos = startPos
	}
	if b.endPos == nil || endPos.Timestamp > b.endPos.Timestamp {
		b.endPos = endPos
	}
}

func (b *BufferBase) IsFull() bool {
	return (b.rowLimit != noLimit && b.rows >= b.rowLimit) ||
		(b.sizeLimit != noLimit && b.size >= b.sizeLimit)
}

func (b *BufferBase) IsEmpty() bool {
	return b.rows == 0 && b.size == 0
}

func (b *BufferBase) MinTimestamp() typeutil.Timestamp {
	if b.startPos == nil {
		return math.MaxUint64
	}
	return b.startPos.GetTimestamp()
}

func (b *BufferBase) GetTimeRange() *TimeRange {
	return &TimeRange{
		timestampMin: b.TimestampFrom,
		timestampMax: b.TimestampTo,
	}
}

type InsertBuffer struct {
	BufferBase
	collSchema *schemapb.CollectionSchema

	buffer *storage.InsertData
}

// SegmentInsertBuffer can be reused to buffer all insert data of one segment
// buffer.Serialize will serialize the InsertBuffer and clear it
// pkstats keeps tracking pkstats of the segment until Finish
type SegmentInsertBuffer struct {
	*InsertBuffer

	pkstats      *storage.PrimaryKeyStats
	segmentID    int64
	partitionID  int64
	collectionID int64
}

func NewSegmentInsertBuffer(sch *schemapb.CollectionSchema, maxCount int64, segID, partID, collID int64) (*SegmentInsertBuffer, error) {
	ibuffer, err := NewInsertBuffer(sch)
	if err != nil {
		return nil, err
	}

	var pkField *schemapb.FieldSchema
	for _, fs := range sch.GetFields() {
		if fs.GetIsPrimaryKey() && fs.GetFieldID() >= 100 && typeutil.IsPrimaryFieldType(fs.GetDataType()) {
			pkField = fs
		}
	}
	if pkField == nil {
		log.Warn("failed to get pk field from schema")
		return nil, fmt.Errorf("no pk field in schema")
	}

	stats, err := storage.NewPrimaryKeyStats(pkField.GetFieldID(), int64(pkField.GetDataType()), maxCount)
	if err != nil {
		return nil, err
	}

	return &SegmentInsertBuffer{
		InsertBuffer: ibuffer,
		pkstats:      stats,
		segmentID:    segID,
		partitionID:  partID,
		collectionID: collID,
	}, nil
}

func (b *SegmentInsertBuffer) Clear() {
	ibuffer, _ := NewInsertBuffer(b.collSchema)
	b.InsertBuffer = ibuffer
}

func (b *SegmentInsertBuffer) GetRowNum() int64 {
	return int64(b.rows)
}

func (b *SegmentInsertBuffer) GetCollectionID() int64 {
	return b.collectionID
}

func (b *SegmentInsertBuffer) GetPartitionID() int64 {
	return b.partitionID
}

func (b *SegmentInsertBuffer) GetSegmentID() int64 {
	return b.segmentID
}

func (b *SegmentInsertBuffer) GetPkID() int64 {
	return b.pkstats.FieldID
}

// Serialize the current InsertBuffer
func (b *SegmentInsertBuffer) SerializeYield() ([]*storage.Blob, *TimeRange, error) {
	codec := storage.NewInsertCodecWithSchema(&etcdpb.CollectionMeta{ID: b.collectionID, Schema: b.collSchema})
	blobs, err := codec.Serialize(b.partitionID, b.segmentID, b.Yield())
	tr := b.GetTimeRange()
	b.Clear()

	return blobs, tr, err
}

// End the life cycle of this buffer and return the stats blobs
func (b *SegmentInsertBuffer) Finish(actualRowCount int64) (*storage.Blob, error) {
	codec := storage.NewInsertCodecWithSchema(&etcdpb.CollectionMeta{ID: b.collectionID, Schema: b.collSchema})
	return codec.SerializePkStats(b.pkstats, actualRowCount)
}

func (b *SegmentInsertBuffer) BufferRow(row map[int64]interface{}, pk storage.PrimaryKey, ts uint64) error {
	b.pkstats.Update(pk)
	return b.InsertBuffer.BufferRow(row, ts)
}

func NewInsertBuffer(sch *schemapb.CollectionSchema) (*InsertBuffer, error) {
	estSize, err := typeutil.EstimateSizePerRecord(sch)
	if err != nil {
		log.Warn("failed to estimate size per record", zap.Error(err))
		return nil, err
	}

	if estSize == 0 {
		return nil, errors.New("Invalid schema")
	}
	buffer, err := storage.NewInsertData(sch)
	if err != nil {
		return nil, err
	}
	sizeLimit := paramtable.Get().DataNodeCfg.FlushInsertBufferSize.GetAsInt64()

	return &InsertBuffer{
		BufferBase: BufferBase{
			rowLimit:      noLimit,
			sizeLimit:     sizeLimit,
			TimestampFrom: math.MaxUint64,
			TimestampTo:   0,
		},
		collSchema: sch,
		buffer:     buffer,
	}, nil
}

func (ib *InsertBuffer) Yield() *storage.InsertData {
	if ib.IsEmpty() {
		return nil
	}

	return ib.buffer
}

func (ib *InsertBuffer) BufferRow(row map[int64]interface{}, ts uint64) error {
	if err := ib.buffer.Append(row); err != nil {
		return err
	}

	ib.rows += 1
	ib.size += int64(ib.buffer.GetRowSize(ib.buffer.GetRowNum() - 1))

	if ts < ib.TimestampFrom {
		ib.TimestampFrom = ts
	}
	if ts > ib.TimestampTo {
		ib.TimestampTo = ts
	}

	return nil
}

func (ib *InsertBuffer) Buffer(inData *inData, startPos, endPos *msgpb.MsgPosition) int64 {
	bufferedSize := int64(0)
	for idx, data := range inData.data {
		storage.MergeInsertData(ib.buffer, data)
		tsData := inData.tsField[idx]

		// update buffer size
		ib.UpdateStatistics(int64(data.GetRowNum()), int64(data.GetMemorySize()), ib.getTimestampRange(tsData), startPos, endPos)
		bufferedSize += int64(data.GetMemorySize())
	}
	return bufferedSize
}

func (ib *InsertBuffer) getTimestampRange(tsData *storage.Int64FieldData) TimeRange {
	tr := TimeRange{
		timestampMin: math.MaxUint64,
		timestampMax: 0,
	}

	for _, data := range tsData.Data {
		if uint64(data) < tr.timestampMin {
			tr.timestampMin = typeutil.Timestamp(data)
		}
		if uint64(data) > tr.timestampMax {
			tr.timestampMax = typeutil.Timestamp(data)
		}
	}
	return tr
}
