package datanode

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/zilliztech/milvus-distributed/internal/msgstream"
	"github.com/zilliztech/milvus-distributed/internal/msgstream/pulsarms"
	"github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb2"
)

// NOTE: start pulsar before test
func TestDataSyncService_Start(t *testing.T) {
	const ctxTimeInMillisecond = 2000
	const closeWithDeadline = true
	var ctx context.Context

	if closeWithDeadline {
		var cancel context.CancelFunc
		d := time.Now().Add(ctxTimeInMillisecond * time.Millisecond)
		ctx, cancel = context.WithDeadline(context.Background(), d)
		defer cancel()
	} else {
		ctx = context.Background()
	}

	// init data node
	pulsarURL := Params.PulsarAddress

	Factory := &MetaFactory{}
	collMeta := Factory.CollectionMetaFactory(UniqueID(0), "coll1")

	chanSize := 100
	flushChan := make(chan *flushMsg, chanSize)
	replica := newReplica()
	allocFactory := AllocatorFactory{}
	sync := newDataSyncService(ctx, flushChan, replica, allocFactory)
	sync.replica.addCollection(collMeta.ID, collMeta.Schema)
	sync.init()
	go sync.start()

	timeRange := TimeRange{
		timestampMin: 0,
		timestampMax: math.MaxUint64,
	}
	dataFactory := NewDataFactory()
	insertMessages := dataFactory.GetMsgStreamTsInsertMsgs(2)

	msgPack := msgstream.MsgPack{
		BeginTs: timeRange.timestampMin,
		EndTs:   timeRange.timestampMax,
		Msgs:    insertMessages,
	}

	// generate timeTick
	timeTickMsgPack := msgstream.MsgPack{}

	timeTickMsg := &msgstream.TimeTickMsg{
		BaseMsg: msgstream.BaseMsg{
			BeginTimestamp: Timestamp(0),
			EndTimestamp:   Timestamp(0),
			HashValues:     []uint32{0},
		},
		TimeTickMsg: internalpb2.TimeTickMsg{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_kTimeTick,
				MsgID:     UniqueID(0),
				Timestamp: math.MaxUint64,
				SourceID:  0,
			},
		},
	}
	timeTickMsgPack.Msgs = append(timeTickMsgPack.Msgs, timeTickMsg)

	// pulsar produce
	const receiveBufSize = 1024
	insertChannels := Params.InsertChannelNames
	ddChannels := Params.DDChannelNames

	factory := msgstream.ProtoUDFactory{}
	insertStream := pulsarms.NewPulsarMsgStream(ctx, receiveBufSize, 1024, factory.NewUnmarshalDispatcher())
	insertStream.SetPulsarClient(pulsarURL)
	insertStream.CreatePulsarProducers(insertChannels)

	ddStream := pulsarms.NewPulsarMsgStream(ctx, receiveBufSize, 1024, factory.NewUnmarshalDispatcher())
	ddStream.SetPulsarClient(pulsarURL)
	ddStream.CreatePulsarProducers(ddChannels)

	var insertMsgStream msgstream.MsgStream = insertStream
	insertMsgStream.Start()

	var ddMsgStream msgstream.MsgStream = ddStream
	ddMsgStream.Start()

	err := insertMsgStream.Produce(&msgPack)
	assert.NoError(t, err)

	err = insertMsgStream.Broadcast(&timeTickMsgPack)
	assert.NoError(t, err)
	err = ddMsgStream.Broadcast(&timeTickMsgPack)
	assert.NoError(t, err)

	// dataSync
	Params.FlushInsertBufferSize = 1
	<-sync.ctx.Done()

	sync.close()
}
