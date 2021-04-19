package proxynode

import (
	"context"
	"errors"
	"strconv"

	"log"
	"time"

	"github.com/opentracing/opentracing-go"
	"github.com/zilliztech/milvus-distributed/internal/msgstream"
	"github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb2"
	"github.com/zilliztech/milvus-distributed/internal/proto/milvuspb"
	"github.com/zilliztech/milvus-distributed/internal/proto/schemapb"
	"github.com/zilliztech/milvus-distributed/internal/proto/servicepb"
)

const (
	reqTimeoutInterval = time.Second * 10
)

func (p *Proxy) Insert(ctx context.Context, in *servicepb.RowBatch) (*servicepb.IntegerRangeResponse, error) {
	span, ctx := opentracing.StartSpanFromContext(ctx, "insert grpc received")
	defer span.Finish()
	span.SetTag("collection name", in.CollectionName)
	span.SetTag("partition tag", in.PartitionTag)
	log.Println("insert into: ", in.CollectionName)
	it := &InsertTask{
		ctx:       ctx,
		Condition: NewTaskCondition(ctx),
		BaseInsertTask: BaseInsertTask{
			BaseMsg: msgstream.BaseMsg{
				HashValues: in.HashKeys,
			},
			InsertRequest: internalpb2.InsertRequest{
				Base: &commonpb.MsgBase{
					MsgType: commonpb.MsgType_kInsert,
					MsgID:   0,
				},
				CollectionName: in.CollectionName,
				PartitionName:  in.PartitionTag,
				RowData:        in.RowData,
			},
		},
		manipulationMsgStream: p.manipulationMsgStream,
		rowIDAllocator:        p.idAllocator,
	}
	if len(it.PartitionName) <= 0 {
		it.PartitionName = Params.defaultPartitionTag()
	}

	var cancel func()
	it.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)

	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("insert timeout")
		default:
			return p.sched.DmQueue.Enqueue(it)
		}
	}
	err := fn()

	if err != nil {
		return &servicepb.IntegerRangeResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	err = it.WaitToFinish()
	if err != nil {
		return &servicepb.IntegerRangeResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	return it.result, nil
}

func (p *Proxy) CreateCollection(ctx context.Context, req *schemapb.CollectionSchema) (*commonpb.Status, error) {
	log.Println("create collection: ", req)
	cct := &CreateCollectionTask{
		Condition: NewTaskCondition(ctx),
		CreateCollectionRequest: milvuspb.CreateCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_kCreateCollection,
			},
		},
		masterClient: p.masterClient,
		schema:       req,
	}
	var cancel func()
	cct.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create collection timeout")
		default:
			return p.sched.DdQueue.Enqueue(cct)
		}
	}
	err := fn()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}

	err = cct.WaitToFinish()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}

	return cct.result, nil
}

func (p *Proxy) Search(ctx context.Context, req *servicepb.Query) (*servicepb.QueryResult, error) {
	span, ctx := opentracing.StartSpanFromContext(ctx, "search grpc received")
	defer span.Finish()
	span.SetTag("collection name", req.CollectionName)
	span.SetTag("partition tag", req.PartitionTags)
	span.SetTag("dsl", req.Dsl)
	log.Println("search: ", req.CollectionName, req.Dsl)
	qt := &SearchTask{
		ctx:       ctx,
		Condition: NewTaskCondition(ctx),
		SearchRequest: internalpb2.SearchRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kSearch,
				SourceID: Params.ProxyID(),
			},
			ResultChannelID: strconv.FormatInt(Params.ProxyID(), 10),
		},
		queryMsgStream: p.queryMsgStream,
		resultBuf:      make(chan []*internalpb2.SearchResults),
		query:          req,
	}
	var cancel func()
	qt.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	log.Printf("grpc address of query task: %p", qt)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create collection timeout")
		default:
			return p.sched.DqQueue.Enqueue(qt)
		}
	}
	err := fn()
	if err != nil {
		return &servicepb.QueryResult{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	err = qt.WaitToFinish()
	if err != nil {
		return &servicepb.QueryResult{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	return qt.result, nil
}

func (p *Proxy) DropCollection(ctx context.Context, req *servicepb.CollectionName) (*commonpb.Status, error) {
	log.Println("drop collection: ", req)
	dct := &DropCollectionTask{
		Condition: NewTaskCondition(ctx),
		DropCollectionRequest: milvuspb.DropCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kDropCollection,
				SourceID: Params.ProxyID(),
			},
			CollectionName: req.CollectionName,
		},
		masterClient: p.masterClient,
	}
	var cancel func()
	dct.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create collection timeout")
		default:
			return p.sched.DdQueue.Enqueue(dct)
		}
	}
	err := fn()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}

	err = dct.WaitToFinish()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}

	return dct.result, nil
}

func (p *Proxy) HasCollection(ctx context.Context, req *servicepb.CollectionName) (*servicepb.BoolResponse, error) {
	log.Println("has collection: ", req)
	hct := &HasCollectionTask{
		Condition: NewTaskCondition(ctx),
		HasCollectionRequest: milvuspb.HasCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kHasCollection,
				SourceID: Params.ProxyID(),
			},
			CollectionName: req.CollectionName,
		},
		masterClient: p.masterClient,
	}
	var cancel func()
	hct.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create collection timeout")
		default:
			return p.sched.DdQueue.Enqueue(hct)
		}
	}
	err := fn()
	if err != nil {
		return &servicepb.BoolResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	err = hct.WaitToFinish()
	if err != nil {
		return &servicepb.BoolResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	return hct.result, nil
}

func (p *Proxy) DescribeCollection(ctx context.Context, req *servicepb.CollectionName) (*servicepb.CollectionDescription, error) {
	log.Println("describe collection: ", req)
	dct := &DescribeCollectionTask{
		Condition: NewTaskCondition(ctx),
		DescribeCollectionRequest: milvuspb.DescribeCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_kDescribeCollection,
			},
			CollectionName: req.CollectionName,
		},
		masterClient: p.masterClient,
	}
	var cancel func()
	dct.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create collection timeout")
		default:
			return p.sched.DdQueue.Enqueue(dct)
		}
	}
	err := fn()
	if err != nil {
		return &servicepb.CollectionDescription{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	err = dct.WaitToFinish()
	if err != nil {
		return &servicepb.CollectionDescription{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	return dct.result, nil
}

func (p *Proxy) ShowCollections(ctx context.Context, req *commonpb.Empty) (*servicepb.StringListResponse, error) {
	log.Println("show collections")
	sct := &ShowCollectionsTask{
		Condition: NewTaskCondition(ctx),
		ShowCollectionRequest: milvuspb.ShowCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kShowCollections,
				SourceID: Params.ProxyID(),
			},
		},
		masterClient: p.masterClient,
	}
	var cancel func()
	sct.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create collection timeout")
		default:
			return p.sched.DdQueue.Enqueue(sct)
		}
	}
	err := fn()
	if err != nil {
		return &servicepb.StringListResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	err = sct.WaitToFinish()
	if err != nil {
		return &servicepb.StringListResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
		}, nil
	}

	return sct.result, nil
}

func (p *Proxy) CreatePartition(ctx context.Context, in *servicepb.PartitionName) (*commonpb.Status, error) {
	log.Println("create partition", in)
	cpt := &CreatePartitionTask{
		Condition: NewTaskCondition(ctx),
		CreatePartitionRequest: milvuspb.CreatePartitionRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kCreatePartition,
				SourceID: Params.ProxyID(),
			},
			CollectionName: in.CollectionName,
			PartitionName:  in.Tag,
			//TODO, ReqID,Timestamp,ProxyID
		},
		masterClient: p.masterClient,
		result:       nil,
		ctx:          nil,
	}
	var cancel func()
	cpt.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	err := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create partition timeout")
		default:
			return p.sched.DdQueue.Enqueue(cpt)
		}
	}()

	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}
	err = cpt.WaitToFinish()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}
	return cpt.result, nil

}

func (p *Proxy) DropPartition(ctx context.Context, in *servicepb.PartitionName) (*commonpb.Status, error) {
	log.Println("drop partition: ", in)
	dpt := &DropPartitionTask{
		Condition: NewTaskCondition(ctx),
		DropPartitionRequest: milvuspb.DropPartitionRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kDropPartition,
				SourceID: Params.ProxyID(),
			},
			CollectionName: in.CollectionName,
			PartitionName:  in.Tag,
			//TODO, ReqID,Timestamp,ProxyID
		},
		masterClient: p.masterClient,
		result:       nil,
		ctx:          nil,
	}

	var cancel func()
	dpt.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	err := func() error {
		select {
		case <-ctx.Done():
			return errors.New("drop partition timeout")
		default:
			return p.sched.DdQueue.Enqueue(dpt)
		}
	}()

	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}
	err = dpt.WaitToFinish()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}
	return dpt.result, nil

}

func (p *Proxy) HasPartition(ctx context.Context, in *servicepb.PartitionName) (*servicepb.BoolResponse, error) {
	log.Println("has partition: ", in)
	hpt := &HasPartitionTask{
		Condition: NewTaskCondition(ctx),
		HasPartitionRequest: milvuspb.HasPartitionRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kHasPartition,
				SourceID: Params.ProxyID(),
			},
			CollectionName: in.CollectionName,
			PartitionName:  in.Tag,
			//TODO, ReqID,Timestamp,ProxyID
		},
		masterClient: p.masterClient,
		result:       nil,
		ctx:          nil,
	}

	var cancel func()
	hpt.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	err := func() error {
		select {
		case <-ctx.Done():
			return errors.New("has partition timeout")
		default:
			return p.sched.DdQueue.Enqueue(hpt)
		}
	}()

	if err != nil {
		return &servicepb.BoolResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			Value: false,
		}, nil
	}
	err = hpt.WaitToFinish()
	if err != nil {
		return &servicepb.BoolResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			Value: false,
		}, nil
	}
	return hpt.result, nil

}

func (p *Proxy) DescribePartition(ctx context.Context, in *servicepb.PartitionName) (*servicepb.PartitionDescription, error) {
	log.Println("describe partition: ", in)

	return &servicepb.PartitionDescription{
		Status: &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    "Deprecated!",
		},
		Name:       in,
		Statistics: nil,
	}, nil

}

//func (p *Proxy) DescribePartition2(ctx context.Context, in *servicepb.PartitionName) (*servicepb.PartitionDescription, error) {
//	log.Println("describe partition: ", in)
//	dpt := &DescribePartitionTask{
//		Condition: NewTaskCondition(ctx),
//		DescribePartitionRequest: internalpb.DescribePartitionRequest{
//			MsgType:       commonpb.MsgType_kDescribePartition,
//			ReqID:         0,
//			Timestamp:     0,
//			ProxyID:       0,
//			PartitionName: in,
//			//TODO, ReqID,Timestamp,ProxyID
//		},
//		masterClient: p.masterClient,
//		result:       nil,
//		ctx:          nil,
//	}
//
//	var cancel func()
//	dpt.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
//	defer cancel()
//
//	err := func() error {
//		select {
//		case <-ctx.Done():
//			return errors.New("describe partion timeout")
//		default:
//			return p.sched.DdQueue.Enqueue(dpt)
//		}
//	}()
//
//	if err != nil {
//		return &servicepb.PartitionDescription{
//			Status: &commonpb.Status{
//				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
//				Reason:    err.Error(),
//			},
//			Name:       in,
//			Statistics: nil,
//		}, nil
//	}
//
//	err = dpt.WaitToFinish()
//	if err != nil {
//		return &servicepb.PartitionDescription{
//			Status: &commonpb.Status{
//				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
//				Reason:    err.Error(),
//			},
//			Name:       in,
//			Statistics: nil,
//		}, nil
//	}
//	return dpt.result, nil
//}

func (p *Proxy) ShowPartitions(ctx context.Context, req *servicepb.CollectionName) (*servicepb.StringListResponse, error) {
	log.Println("show partitions: ", req)
	spt := &ShowPartitionsTask{
		Condition: NewTaskCondition(ctx),
		ShowPartitionRequest: milvuspb.ShowPartitionRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kShowPartitions,
				SourceID: Params.ProxyID(),
			},
			CollectionName: req.CollectionName,
		},
		masterClient: p.masterClient,
		result:       nil,
		ctx:          nil,
	}

	var cancel func()
	spt.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	err := func() error {
		select {
		case <-ctx.Done():
			return errors.New("show partition timeout")
		default:
			return p.sched.DdQueue.Enqueue(spt)
		}
	}()

	if err != nil {
		return &servicepb.StringListResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			Values: nil,
		}, nil
	}

	err = spt.WaitToFinish()
	if err != nil {
		return &servicepb.StringListResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			Values: nil,
		}, nil
	}
	return spt.result, nil
}

func (p *Proxy) CreateIndex(ctx context.Context, indexParam *servicepb.IndexParam) (*commonpb.Status, error) {
	log.Println("create index for: ", indexParam.FieldName)
	cit := &CreateIndexTask{
		Condition: NewTaskCondition(ctx),
		CreateIndexRequest: milvuspb.CreateIndexRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kCreateIndex,
				SourceID: Params.ProxyID(),
			},
			CollectionName: indexParam.CollectionName,
			FieldName:      indexParam.FieldName,
			ExtraParams:    indexParam.ExtraParams,
		},
		masterClient: p.masterClient,
	}

	var cancel func()
	cit.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create index timeout")
		default:
			return p.sched.DdQueue.Enqueue(cit)
		}
	}
	err := fn()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}

	err = cit.WaitToFinish()
	if err != nil {
		return &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
			Reason:    err.Error(),
		}, nil
	}

	return cit.result, nil
}

func (p *Proxy) DescribeIndex(ctx context.Context, req *servicepb.DescribeIndexRequest) (*servicepb.DescribeIndexResponse, error) {
	log.Println("Describe index for: ", req.FieldName)
	dit := &DescribeIndexTask{
		Condition: NewTaskCondition(ctx),
		DescribeIndexRequest: milvuspb.DescribeIndexRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kDescribeIndex,
				SourceID: Params.ProxyID(),
			},
			CollectionName: req.CollectionName,
			FieldName:      req.FieldName,
		},
		masterClient: p.masterClient,
	}

	var cancel func()
	dit.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create index timeout")
		default:
			return p.sched.DdQueue.Enqueue(dit)
		}
	}
	err := fn()
	if err != nil {
		return &servicepb.DescribeIndexResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			CollectionName: req.CollectionName,
			FieldName:      req.FieldName,
			ExtraParams:    nil,
		}, nil
	}

	err = dit.WaitToFinish()
	if err != nil {
		return &servicepb.DescribeIndexResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			CollectionName: req.CollectionName,
			FieldName:      req.FieldName,
			ExtraParams:    nil,
		}, nil
	}

	return dit.result, nil
}

func (p *Proxy) DescribeIndexProgress(ctx context.Context, req *servicepb.DescribeIndexProgressRequest) (*servicepb.BoolResponse, error) {
	log.Println("Describe index progress for: ", req.FieldName)
	dipt := &DescribeIndexProgressTask{
		Condition: NewTaskCondition(ctx),
		IndexStateRequest: milvuspb.IndexStateRequest{
			Base: &commonpb.MsgBase{
				MsgType:  commonpb.MsgType_kGetIndexState,
				SourceID: Params.ProxyID(),
			},
			CollectionName: req.CollectionName,
			FieldName:      req.FieldName,
		},
		masterClient: p.masterClient,
	}

	var cancel func()
	dipt.ctx, cancel = context.WithTimeout(ctx, reqTimeoutInterval)
	defer cancel()

	fn := func() error {
		select {
		case <-ctx.Done():
			return errors.New("create index timeout")
		default:
			return p.sched.DdQueue.Enqueue(dipt)
		}
	}
	err := fn()
	if err != nil {
		return &servicepb.BoolResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			Value: false,
		}, nil
	}

	err = dipt.WaitToFinish()
	if err != nil {
		return &servicepb.BoolResponse{
			Status: &commonpb.Status{
				ErrorCode: commonpb.ErrorCode_UNEXPECTED_ERROR,
				Reason:    err.Error(),
			},
			Value: false,
		}, nil
	}

	return dipt.result, nil
}
