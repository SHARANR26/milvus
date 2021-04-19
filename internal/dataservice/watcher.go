package dataservice

import (
	"log"

	"github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/datapb"

	"golang.org/x/net/context"

	"github.com/zilliztech/milvus-distributed/internal/msgstream"
)

type (
	proxyTimeTickWatcher struct {
		allocator segmentAllocator
		msgQueue  chan *msgstream.TimeTickMsg
	}
	dataNodeTimeTickWatcher struct {
		meta      *meta
		cluster   *dataNodeCluster
		allocator segmentAllocator
		msgQueue  chan *msgstream.TimeTickMsg
	}
)

func newProxyTimeTickWatcher(allocator segmentAllocator) *proxyTimeTickWatcher {
	return &proxyTimeTickWatcher{
		allocator: allocator,
		msgQueue:  make(chan *msgstream.TimeTickMsg, 1),
	}
}

func (watcher *proxyTimeTickWatcher) StartBackgroundLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			log.Println("proxy time tick watcher closed")
			return
		case msg := <-watcher.msgQueue:
			if err := watcher.allocator.ExpireAllocations(msg.Base.Timestamp); err != nil {
				log.Printf("expire allocations error : %s", err.Error())
			}
		}
	}
}

func (watcher *proxyTimeTickWatcher) Watch(msg *msgstream.TimeTickMsg) {
	watcher.msgQueue <- msg
}

func newDataNodeTimeTickWatcher(meta *meta, allocator segmentAllocator, cluster *dataNodeCluster) *dataNodeTimeTickWatcher {
	return &dataNodeTimeTickWatcher{
		meta:      meta,
		allocator: allocator,
		cluster:   cluster,
		msgQueue:  make(chan *msgstream.TimeTickMsg, 1),
	}
}

func (watcher *dataNodeTimeTickWatcher) Watch(msg *msgstream.TimeTickMsg) {
	watcher.msgQueue <- msg
}

func (watcher *dataNodeTimeTickWatcher) StartBackgroundLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			log.Println("data node time tick watcher closed")
			return
		case msg := <-watcher.msgQueue:
			if err := watcher.handleTimeTickMsg(msg); err != nil {
				log.Println(err.Error())
				continue
			}
		}
	}
}

func (watcher *dataNodeTimeTickWatcher) handleTimeTickMsg(msg *msgstream.TimeTickMsg) error {
	segments, err := watcher.allocator.GetSealedSegments()
	if err != nil {
		return err
	}
	for _, id := range segments {
		expired, err := watcher.allocator.IsAllocationsExpired(id, msg.Base.Timestamp)
		if err != nil {
			log.Printf("check allocations expired error %s", err.Error())
			continue
		}
		if expired {
			segmentInfo, err := watcher.meta.GetSegment(id)
			if err != nil {
				log.Println(err.Error())
				continue
			}
			if err = watcher.meta.SetSegmentState(id, commonpb.SegmentState_SegmentSealed); err != nil {
				log.Println(err.Error())
				continue
			}
			watcher.cluster.FlushSegment(&datapb.FlushSegRequest{
				Base: &commonpb.MsgBase{
					MsgType:   commonpb.MsgType_kShowCollections,
					MsgID:     -1, // todo add msg id
					Timestamp: 0,  // todo
					SourceID:  Params.NodeID,
				},
				CollectionID: segmentInfo.CollectionID,
				SegmentIDs:   []int64{segmentInfo.SegmentID},
			})
			watcher.allocator.DropSegment(id)
		}
	}
	return nil
}
