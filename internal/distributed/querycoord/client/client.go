// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package grpcquerycoordclient

import (
	"context"
	"fmt"
	"time"

	grpc_middleware "github.com/grpc-ecosystem/go-grpc-middleware"
	grpc_retry "github.com/grpc-ecosystem/go-grpc-middleware/retry"
	grpc_opentracing "github.com/grpc-ecosystem/go-grpc-middleware/tracing/opentracing"
	"github.com/milvus-io/milvus/internal/util/retry"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/trace"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
)

type Client struct {
	ctx        context.Context
	grpcClient querypb.QueryCoordClient
	conn       *grpc.ClientConn

	addr      string
	timeout   time.Duration
	sess      *sessionutil.Session
	reconnTry int
	recallTry int
}

func getQueryCoordAddress(sess *sessionutil.Session) (string, error) {
	key := typeutil.QueryCoordRole
	msess, _, err := sess.GetSessions(key)
	if err != nil {
		log.Debug("QueryCoordClient GetSessions failed", zap.Error(err))
		return "", err
	}
	ms, ok := msess[key]
	if !ok {
		log.Debug("QueryCoordClient msess key not existed", zap.Any("key", key))
		return "", fmt.Errorf("number of querycoord is incorrect, %d", len(msess))
	}
	return ms.Address, nil
}

// NewClient creates a client for QueryCoord grpc call.
func NewClient(metaRootPath string, etcdEndpoints []string, timeout time.Duration) (*Client, error) {
	sess := sessionutil.NewSession(context.Background(), metaRootPath, etcdEndpoints)

	return &Client{
		ctx:        context.Background(),
		grpcClient: nil,
		conn:       nil,
		timeout:    timeout,
		reconnTry:  10,
		recallTry:  3,
		sess:       sess,
	}, nil
}

func (c *Client) Init() error {
	// for now, we must try many times in Init Stage
	initFunc := func() error {
		return c.connect()
	}
	err := retry.Retry(10000, 3*time.Second, initFunc)
	return err
}

func (c *Client) connect() error {
	var err error
	getQueryCoordAddressFn := func() error {
		c.addr, err = getQueryCoordAddress(c.sess)
		if err != nil {
			return err
		}
		return nil
	}
	err = retry.Retry(c.reconnTry, 3*time.Second, getQueryCoordAddressFn)
	if err != nil {
		log.Debug("QueryCoordClient getQueryCoordAddress failed", zap.Error(err))
		return err
	}
	connectGrpcFunc := func() error {
		ctx, cancelFunc := context.WithTimeout(c.ctx, c.timeout)
		defer cancelFunc()
		opts := trace.GetInterceptorOpts()
		log.Debug("QueryCoordClient try reconnect ", zap.String("address", c.addr))
		conn, err := grpc.DialContext(ctx, c.addr, grpc.WithInsecure(), grpc.WithBlock(),
			grpc.WithUnaryInterceptor(
				grpc_middleware.ChainUnaryClient(
					grpc_retry.UnaryClientInterceptor(),
					grpc_opentracing.UnaryClientInterceptor(opts...),
				)),
			grpc.WithStreamInterceptor(
				grpc_middleware.ChainStreamClient(
					grpc_retry.StreamClientInterceptor(),
					grpc_opentracing.StreamClientInterceptor(opts...),
				)),
		)
		if err != nil {
			return err
		}
		c.conn = conn
		return nil
	}

	err = retry.Retry(c.reconnTry, 500*time.Millisecond, connectGrpcFunc)
	if err != nil {
		log.Debug("QueryCoordClient try reconnect failed", zap.Error(err))
		return err
	}
	log.Debug("QueryCoordClient try reconnect success")
	c.grpcClient = querypb.NewQueryCoordClient(c.conn)
	return nil
}
func (c *Client) recall(caller func() (interface{}, error)) (interface{}, error) {
	ret, err := caller()
	if err == nil {
		return ret, nil
	}
	for i := 0; i < c.reconnTry; i++ {
		err = c.connect()
		if err == nil {
			break
		}
	}
	if err != nil {
		return nil, err
	}
	ret, err = caller()
	if err == nil {
		return ret, nil
	}
	return ret, err
}

func (c *Client) Start() error {
	return nil
}

func (c *Client) Stop() error {
	return c.conn.Close()
}

// Register dummy
func (c *Client) Register() error {
	return nil
}

func (c *Client) GetComponentStates(ctx context.Context) (*internalpb.ComponentStates, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.GetComponentStates(ctx, &internalpb.GetComponentStatesRequest{})
	})
	return ret.(*internalpb.ComponentStates), err
}

func (c *Client) GetTimeTickChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.GetTimeTickChannel(ctx, &internalpb.GetTimeTickChannelRequest{})
	})
	return ret.(*milvuspb.StringResponse), err
}

func (c *Client) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.GetStatisticsChannel(ctx, &internalpb.GetStatisticsChannelRequest{})
	})
	return ret.(*milvuspb.StringResponse), err
}

func (c *Client) ShowCollections(ctx context.Context, req *querypb.ShowCollectionsRequest) (*querypb.ShowCollectionsResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.ShowCollections(ctx, req)
	})
	return ret.(*querypb.ShowCollectionsResponse), err
}

func (c *Client) LoadCollection(ctx context.Context, req *querypb.LoadCollectionRequest) (*commonpb.Status, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.LoadCollection(ctx, req)
	})
	return ret.(*commonpb.Status), err
}

func (c *Client) ReleaseCollection(ctx context.Context, req *querypb.ReleaseCollectionRequest) (*commonpb.Status, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.ReleaseCollection(ctx, req)
	})
	return ret.(*commonpb.Status), err
}

func (c *Client) ShowPartitions(ctx context.Context, req *querypb.ShowPartitionsRequest) (*querypb.ShowPartitionsResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.ShowPartitions(ctx, req)
	})
	return ret.(*querypb.ShowPartitionsResponse), err
}

func (c *Client) LoadPartitions(ctx context.Context, req *querypb.LoadPartitionsRequest) (*commonpb.Status, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.LoadPartitions(ctx, req)
	})
	return ret.(*commonpb.Status), err
}

func (c *Client) ReleasePartitions(ctx context.Context, req *querypb.ReleasePartitionsRequest) (*commonpb.Status, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.ReleasePartitions(ctx, req)
	})
	return ret.(*commonpb.Status), err
}

func (c *Client) CreateQueryChannel(ctx context.Context, req *querypb.CreateQueryChannelRequest) (*querypb.CreateQueryChannelResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.CreateQueryChannel(ctx, req)
	})
	return ret.(*querypb.CreateQueryChannelResponse), err
}

func (c *Client) GetPartitionStates(ctx context.Context, req *querypb.GetPartitionStatesRequest) (*querypb.GetPartitionStatesResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.GetPartitionStates(ctx, req)
	})
	return ret.(*querypb.GetPartitionStatesResponse), err
}

func (c *Client) GetSegmentInfo(ctx context.Context, req *querypb.GetSegmentInfoRequest) (*querypb.GetSegmentInfoResponse, error) {
	ret, err := c.recall(func() (interface{}, error) {
		return c.grpcClient.GetSegmentInfo(ctx, req)
	})
	return ret.(*querypb.GetSegmentInfoResponse), err
}
