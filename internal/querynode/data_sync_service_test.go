package querynode

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/zilliztech/milvus-distributed/internal/msgstream"
	"github.com/zilliztech/milvus-distributed/internal/msgstream/pulsarms"
	"github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb2"
)

// NOTE: start pulsar before test
func TestDataSyncService_Start(t *testing.T) {
	ctx := context.Background()

	node := newQueryNodeMock()
	initTestMeta(t, node, 0, 0)
	// test data generate
	const msgLength = 10
	const DIM = 16
	const N = 10

	var vec = [DIM]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	var rawData []byte
	for _, ele := range vec {
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, math.Float32bits(ele))
		rawData = append(rawData, buf...)
	}
	bs := make([]byte, 4)
	binary.LittleEndian.PutUint32(bs, 1)
	rawData = append(rawData, bs...)
	var records []*commonpb.Blob
	for i := 0; i < N; i++ {
		blob := &commonpb.Blob{
			Value: rawData,
		}
		records = append(records, blob)
	}

	timeRange := TimeRange{
		timestampMin: 0,
		timestampMax: math.MaxUint64,
	}

	// messages generate
	insertMessages := make([]msgstream.TsMsg, 0)
	for i := 0; i < msgLength; i++ {
		var msg msgstream.TsMsg = &msgstream.InsertMsg{
			BaseMsg: msgstream.BaseMsg{
				HashValues: []uint32{
					uint32(i), uint32(i),
				},
			},
			InsertRequest: internalpb2.InsertRequest{
				Base: &commonpb.MsgBase{
					MsgType:   commonpb.MsgType_Insert,
					MsgID:     0,
					Timestamp: uint64(i + 1000),
					SourceID:  0,
				},
				CollectionID: UniqueID(0),
				PartitionID:  defaultPartitionID,
				SegmentID:    int64(0),
				ChannelID:    "0",
				Timestamps:   []uint64{uint64(i + 1000), uint64(i + 1000)},
				RowIDs:       []int64{int64(i), int64(i)},
				RowData: []*commonpb.Blob{
					{Value: rawData},
					{Value: rawData},
				},
			},
		}
		insertMessages = append(insertMessages, msg)
	}

	msgPack := msgstream.MsgPack{
		BeginTs: timeRange.timestampMin,
		EndTs:   timeRange.timestampMax,
		Msgs:    insertMessages,
	}

	// generate timeTick
	timeTickMsgPack := msgstream.MsgPack{}
	baseMsg := msgstream.BaseMsg{
		BeginTimestamp: 0,
		EndTimestamp:   0,
		HashValues:     []uint32{0},
	}
	timeTickResult := internalpb2.TimeTickMsg{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_TimeTick,
			MsgID:     0,
			Timestamp: math.MaxUint64,
			SourceID:  0,
		},
	}
	timeTickMsg := &msgstream.TimeTickMsg{
		BaseMsg:     baseMsg,
		TimeTickMsg: timeTickResult,
	}
	timeTickMsgPack.Msgs = append(timeTickMsgPack.Msgs, timeTickMsg)

	// pulsar produce
	const receiveBufSize = 1024
	insertChannels := Params.InsertChannelNames
	pulsarURL := Params.PulsarAddress

	msFactory := pulsarms.NewFactory()
	m := map[string]interface{}{
		"receiveBufSize": receiveBufSize,
		"pulsarAddress":  pulsarURL,
		"pulsarBufSize":  1024}
	err := msFactory.SetParams(m)
	assert.Nil(t, err)

	insertStream, _ := msFactory.NewMsgStream(node.queryNodeLoopCtx)
	insertStream.AsProducer(insertChannels)

	var insertMsgStream msgstream.MsgStream = insertStream
	insertMsgStream.Start()

	err = insertMsgStream.Produce(ctx, &msgPack)
	assert.NoError(t, err)

	err = insertMsgStream.Broadcast(ctx, &timeTickMsgPack)
	assert.NoError(t, err)

	// dataSync
	node.dataSyncService = newDataSyncService(node.queryNodeLoopCtx, node.replica, msFactory)
	go node.dataSyncService.start()

	<-node.queryNodeLoopCtx.Done()
	node.Stop()
}
