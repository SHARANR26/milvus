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

package accesslog

import (
	"context"
	"fmt"
	"net"
	"testing"

	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/internal/proxy/connection"
	"github.com/milvus-io/milvus/pkg/util"
	"github.com/milvus-io/milvus/pkg/util/crypto"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

type GrpcAccessInfoSuite struct {
	suite.Suite

	username string
	traceID  string
	info     *GrpcAccessInfo
}

func (s *GrpcAccessInfoSuite) SetupTest() {
	s.username = "test-user"
	s.traceID = "test-trace"

	ctx := peer.NewContext(
		context.Background(),
		&peer.Peer{
			Addr: &net.IPAddr{
				IP:   net.IPv4(0, 0, 0, 0),
				Zone: "test",
			},
		})

	md := metadata.Pairs(util.HeaderAuthorize, crypto.Base64Encode("mockUser:mockPass"))
	ctx = metadata.NewIncomingContext(ctx, md)
	serverinfo := &grpc.UnaryServerInfo{
		FullMethod: "test",
	}

	s.info = &GrpcAccessInfo{
		ctx:      ctx,
		grpcInfo: serverinfo,
	}
}

func (s *GrpcAccessInfoSuite) TestErrorCode() {
	s.info.resp = &milvuspb.QueryResults{
		Status: merr.Status(nil),
	}
	result := s.info.Get("$error_code")
	s.Equal(fmt.Sprint(0), result[0])

	s.info.resp = merr.Status(nil)
	result = s.info.Get("$error_code")
	s.Equal(fmt.Sprint(0), result[0])
}

func (s *GrpcAccessInfoSuite) TestErrorMsg() {
	s.info.resp = &milvuspb.QueryResults{
		Status: merr.Status(merr.ErrChannelLack),
	}
	result := s.info.Get("$error_msg")
	s.Equal(merr.ErrChannelLack.Error(), result[0])

	s.info.resp = merr.Status(merr.ErrChannelLack)
	result = s.info.Get("$error_msg")
	s.Equal(merr.ErrChannelLack.Error(), result[0])

	s.info.err = status.Errorf(codes.Unavailable, "mock")
	result = s.info.Get("$error_msg")
	s.Equal("rpc error: code = Unavailable desc = mock", result[0])
}

func (s *GrpcAccessInfoSuite) TestDbName() {
	s.info.req = nil
	result := s.info.Get("$database_name")
	s.Equal(unknownString, result[0])

	s.info.req = &milvuspb.QueryRequest{
		DbName: "test",
	}
	result = s.info.Get("$database_name")
	s.Equal("test", result[0])
}

func (s *GrpcAccessInfoSuite) TestSdkInfo() {
	ctx := context.Background()
	clientInfo := &commonpb.ClientInfo{
		SdkType:    "test",
		SdkVersion: "1.0",
	}

	s.info.ctx = ctx
	result := s.info.Get("$sdk_version")
	s.Equal(unknownString, result[0])

	md := metadata.MD{}
	ctx = metadata.NewIncomingContext(ctx, md)
	s.info.ctx = ctx
	result = s.info.Get("$sdk_version")
	s.Equal(unknownString, result[0])

	md = metadata.MD{util.HeaderUserAgent: []string{"invalid"}}
	ctx = metadata.NewIncomingContext(ctx, md)
	s.info.ctx = ctx
	result = s.info.Get("$sdk_version")
	s.Equal(unknownString, result[0])

	md = metadata.MD{util.HeaderUserAgent: []string{"grpc-go.test"}}
	ctx = metadata.NewIncomingContext(ctx, md)
	s.info.ctx = ctx
	result = s.info.Get("$sdk_version")
	s.Equal("Golang"+"-"+unknownString, result[0])

	s.info.req = &milvuspb.ConnectRequest{
		ClientInfo: clientInfo,
	}
	result = s.info.Get("$sdk_version")
	s.Equal(clientInfo.SdkType+"-"+clientInfo.SdkVersion, result[0])

	identifier := 11111
	md = metadata.MD{util.IdentifierKey: []string{fmt.Sprint(identifier)}}
	ctx = metadata.NewIncomingContext(ctx, md)
	connection.GetManager().Register(ctx, int64(identifier), clientInfo)

	s.info.ctx = ctx
	result = s.info.Get("$sdk_version")
	s.Equal(clientInfo.SdkType+"-"+clientInfo.SdkVersion, result[0])
}

func (s *GrpcAccessInfoSuite) TestExpression() {
	result := s.info.Get("$method_expr")
	s.Equal(unknownString, result[0])

	testExpr := "test"
	s.info.req = &milvuspb.QueryRequest{
		Expr: testExpr,
	}
	result = s.info.Get("$method_expr")
	s.Equal(testExpr, result[0])

	s.info.req = &milvuspb.SearchRequest{
		Dsl: testExpr,
	}
	result = s.info.Get("$method_expr")
	s.Equal(testExpr, result[0])
}

func (s *GrpcAccessInfoSuite) TestOutputFields() {
	result := s.info.Get("$output_fields")
	s.Equal(unknownString, result[0])

	fileds := []string{"pk"}
	s.info.req = &milvuspb.QueryRequest{
		OutputFields: fileds,
	}
	result = s.info.Get("$output_fields")
	s.Equal(fmt.Sprint(fileds), result[0])
}

func (s *GrpcAccessInfoSuite) TestClusterPrefix() {
	cluster := "instance-test"
	paramtable.Init()
	paramtable.Get().Save(paramtable.Get().CommonCfg.ClusterPrefix.Key, cluster)
	result := s.info.Get("$cluster_prefix")
	s.Equal(cluster, result[0])
}

func TestGrpcAccssInfo(t *testing.T) {
	suite.Run(t, new(GrpcAccessInfoSuite))
}
