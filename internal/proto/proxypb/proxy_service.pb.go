// Code generated by protoc-gen-go. DO NOT EDIT.
// source: proxy_service.proto

package proxypb

import (
	context "context"
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	commonpb "github.com/milvus-io/milvus/internal/proto/commonpb"
	internalpb "github.com/milvus-io/milvus/internal/proto/internalpb"
	milvuspb "github.com/milvus-io/milvus/internal/proto/milvuspb"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type RegisterNodeRequest struct {
	Base                 *commonpb.MsgBase `protobuf:"bytes,1,opt,name=base,proto3" json:"base,omitempty"`
	Address              *commonpb.Address `protobuf:"bytes,2,opt,name=address,proto3" json:"address,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *RegisterNodeRequest) Reset()         { *m = RegisterNodeRequest{} }
func (m *RegisterNodeRequest) String() string { return proto.CompactTextString(m) }
func (*RegisterNodeRequest) ProtoMessage()    {}
func (*RegisterNodeRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_34ca2fbc94d169de, []int{0}
}

func (m *RegisterNodeRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RegisterNodeRequest.Unmarshal(m, b)
}
func (m *RegisterNodeRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RegisterNodeRequest.Marshal(b, m, deterministic)
}
func (m *RegisterNodeRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegisterNodeRequest.Merge(m, src)
}
func (m *RegisterNodeRequest) XXX_Size() int {
	return xxx_messageInfo_RegisterNodeRequest.Size(m)
}
func (m *RegisterNodeRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_RegisterNodeRequest.DiscardUnknown(m)
}

var xxx_messageInfo_RegisterNodeRequest proto.InternalMessageInfo

func (m *RegisterNodeRequest) GetBase() *commonpb.MsgBase {
	if m != nil {
		return m.Base
	}
	return nil
}

func (m *RegisterNodeRequest) GetAddress() *commonpb.Address {
	if m != nil {
		return m.Address
	}
	return nil
}

type RegisterNodeResponse struct {
	InitParams           *internalpb.InitParams `protobuf:"bytes,1,opt,name=init_params,json=initParams,proto3" json:"init_params,omitempty"`
	Status               *commonpb.Status       `protobuf:"bytes,2,opt,name=status,proto3" json:"status,omitempty"`
	XXX_NoUnkeyedLiteral struct{}               `json:"-"`
	XXX_unrecognized     []byte                 `json:"-"`
	XXX_sizecache        int32                  `json:"-"`
}

func (m *RegisterNodeResponse) Reset()         { *m = RegisterNodeResponse{} }
func (m *RegisterNodeResponse) String() string { return proto.CompactTextString(m) }
func (*RegisterNodeResponse) ProtoMessage()    {}
func (*RegisterNodeResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_34ca2fbc94d169de, []int{1}
}

func (m *RegisterNodeResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RegisterNodeResponse.Unmarshal(m, b)
}
func (m *RegisterNodeResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RegisterNodeResponse.Marshal(b, m, deterministic)
}
func (m *RegisterNodeResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegisterNodeResponse.Merge(m, src)
}
func (m *RegisterNodeResponse) XXX_Size() int {
	return xxx_messageInfo_RegisterNodeResponse.Size(m)
}
func (m *RegisterNodeResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_RegisterNodeResponse.DiscardUnknown(m)
}

var xxx_messageInfo_RegisterNodeResponse proto.InternalMessageInfo

func (m *RegisterNodeResponse) GetInitParams() *internalpb.InitParams {
	if m != nil {
		return m.InitParams
	}
	return nil
}

func (m *RegisterNodeResponse) GetStatus() *commonpb.Status {
	if m != nil {
		return m.Status
	}
	return nil
}

type InvalidateCollMetaCacheRequest struct {
	Base                 *commonpb.MsgBase `protobuf:"bytes,1,opt,name=base,proto3" json:"base,omitempty"`
	DbName               string            `protobuf:"bytes,2,opt,name=db_name,json=dbName,proto3" json:"db_name,omitempty"`
	CollectionName       string            `protobuf:"bytes,3,opt,name=collection_name,json=collectionName,proto3" json:"collection_name,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *InvalidateCollMetaCacheRequest) Reset()         { *m = InvalidateCollMetaCacheRequest{} }
func (m *InvalidateCollMetaCacheRequest) String() string { return proto.CompactTextString(m) }
func (*InvalidateCollMetaCacheRequest) ProtoMessage()    {}
func (*InvalidateCollMetaCacheRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_34ca2fbc94d169de, []int{2}
}

func (m *InvalidateCollMetaCacheRequest) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_InvalidateCollMetaCacheRequest.Unmarshal(m, b)
}
func (m *InvalidateCollMetaCacheRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_InvalidateCollMetaCacheRequest.Marshal(b, m, deterministic)
}
func (m *InvalidateCollMetaCacheRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_InvalidateCollMetaCacheRequest.Merge(m, src)
}
func (m *InvalidateCollMetaCacheRequest) XXX_Size() int {
	return xxx_messageInfo_InvalidateCollMetaCacheRequest.Size(m)
}
func (m *InvalidateCollMetaCacheRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_InvalidateCollMetaCacheRequest.DiscardUnknown(m)
}

var xxx_messageInfo_InvalidateCollMetaCacheRequest proto.InternalMessageInfo

func (m *InvalidateCollMetaCacheRequest) GetBase() *commonpb.MsgBase {
	if m != nil {
		return m.Base
	}
	return nil
}

func (m *InvalidateCollMetaCacheRequest) GetDbName() string {
	if m != nil {
		return m.DbName
	}
	return ""
}

func (m *InvalidateCollMetaCacheRequest) GetCollectionName() string {
	if m != nil {
		return m.CollectionName
	}
	return ""
}

func init() {
	proto.RegisterType((*RegisterNodeRequest)(nil), "milvus.proto.proxy.RegisterNodeRequest")
	proto.RegisterType((*RegisterNodeResponse)(nil), "milvus.proto.proxy.RegisterNodeResponse")
	proto.RegisterType((*InvalidateCollMetaCacheRequest)(nil), "milvus.proto.proxy.InvalidateCollMetaCacheRequest")
}

func init() { proto.RegisterFile("proxy_service.proto", fileDescriptor_34ca2fbc94d169de) }

var fileDescriptor_34ca2fbc94d169de = []byte{
	// 501 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xc4, 0x93, 0xcf, 0x6e, 0xd3, 0x40,
	0x10, 0xc6, 0x13, 0x5a, 0xa5, 0x62, 0x6a, 0x15, 0xb4, 0xad, 0x44, 0x65, 0xfe, 0x08, 0x8c, 0x44,
	0x2b, 0x24, 0x9c, 0xca, 0x45, 0xdc, 0x49, 0x90, 0xa2, 0x1e, 0x5a, 0x55, 0x4e, 0x4f, 0x5c, 0xa2,
	0xb5, 0x3d, 0x4a, 0x56, 0x78, 0x77, 0xdd, 0xdd, 0x4d, 0x04, 0x27, 0x1e, 0x81, 0x0b, 0x6f, 0xc3,
	0x3b, 0xf0, 0x4c, 0xc8, 0xeb, 0x3f, 0xd4, 0x49, 0x63, 0x14, 0x71, 0xc8, 0xcd, 0x6b, 0xfd, 0x66,
	0xbe, 0xf9, 0x76, 0xbe, 0x85, 0xc3, 0x4c, 0xc9, 0xaf, 0xdf, 0x26, 0x1a, 0xd5, 0x82, 0xc5, 0xe8,
	0x67, 0x4a, 0x1a, 0x49, 0x08, 0x67, 0xe9, 0x62, 0xae, 0x8b, 0x93, 0x6f, 0x09, 0xd7, 0x89, 0x25,
	0xe7, 0x52, 0x14, 0xff, 0xdc, 0x03, 0x26, 0x0c, 0x2a, 0x41, 0xd3, 0xf2, 0xec, 0xdc, 0xad, 0xf0,
	0xbe, 0xc3, 0x61, 0x88, 0x53, 0xa6, 0x0d, 0xaa, 0x2b, 0x99, 0x60, 0x88, 0xb7, 0x73, 0xd4, 0x86,
	0x9c, 0xc1, 0x6e, 0x44, 0x35, 0x1e, 0x77, 0x5f, 0x76, 0x4f, 0xf7, 0x83, 0x67, 0x7e, 0x43, 0xa5,
	0x6c, 0x7f, 0xa9, 0xa7, 0x03, 0xaa, 0x31, 0xb4, 0x24, 0xf9, 0x00, 0x7b, 0x34, 0x49, 0x14, 0x6a,
	0x7d, 0xfc, 0xa0, 0xa5, 0xe8, 0x63, 0xc1, 0x84, 0x15, 0xec, 0xfd, 0xe8, 0xc2, 0x51, 0x73, 0x02,
	0x9d, 0x49, 0xa1, 0x91, 0x0c, 0x60, 0x9f, 0x09, 0x66, 0x26, 0x19, 0x55, 0x94, 0xeb, 0x72, 0x92,
	0x57, 0xcd, 0xa6, 0xb5, 0xb5, 0x0b, 0xc1, 0xcc, 0xb5, 0x05, 0x43, 0x60, 0xf5, 0x37, 0x39, 0x87,
	0x9e, 0x36, 0xd4, 0xcc, 0xab, 0x99, 0x9e, 0xde, 0x3b, 0xd3, 0xd8, 0x22, 0x61, 0x89, 0x7a, 0x3f,
	0xbb, 0xf0, 0xe2, 0x42, 0x2c, 0x68, 0xca, 0x12, 0x6a, 0x70, 0x28, 0xd3, 0xf4, 0x12, 0x0d, 0x1d,
	0xd2, 0x78, 0xf6, 0x1f, 0xd7, 0xf3, 0x04, 0xf6, 0x92, 0x68, 0x22, 0x28, 0x47, 0x3b, 0xca, 0xc3,
	0xb0, 0x97, 0x44, 0x57, 0x94, 0x23, 0x39, 0x81, 0x47, 0xb1, 0x4c, 0x53, 0x8c, 0x0d, 0x93, 0xa2,
	0x00, 0x76, 0x2c, 0x70, 0xf0, 0xf7, 0x77, 0x0e, 0x06, 0xbf, 0x76, 0xc1, 0xb9, 0xce, 0xf7, 0x3b,
	0x2e, 0x02, 0x40, 0x32, 0x20, 0x23, 0x34, 0x43, 0xc9, 0x33, 0x29, 0x50, 0x98, 0xdc, 0x05, 0x6a,
	0x72, 0xb6, 0xe6, 0x86, 0x56, 0xd1, 0xd2, 0x8c, 0xfb, 0x66, 0x4d, 0xc5, 0x12, 0xee, 0x75, 0x08,
	0xb7, 0x8a, 0x37, 0x8c, 0xe3, 0x0d, 0x8b, 0xbf, 0x0c, 0x67, 0x54, 0x08, 0x4c, 0xdb, 0x14, 0x97,
	0xd0, 0x4a, 0xf1, 0x75, 0xb3, 0xa2, 0x3c, 0x8c, 0x8d, 0x62, 0x62, 0x5a, 0xed, 0xdf, 0xeb, 0x90,
	0x5b, 0x38, 0x1a, 0xa1, 0x55, 0x67, 0xda, 0xb0, 0x58, 0x57, 0x82, 0xc1, 0x7a, 0xc1, 0x15, 0x78,
	0x43, 0xc9, 0x18, 0x9c, 0xbb, 0x61, 0x24, 0x27, 0xfe, 0xea, 0xfb, 0xf2, 0xef, 0x79, 0x30, 0xee,
	0xe9, 0xbf, 0xc1, 0x5a, 0x44, 0xc1, 0xf3, 0x66, 0xbe, 0x8a, 0x2d, 0xd7, 0x29, 0x5b, 0x36, 0x58,
	0x34, 0x6b, 0x8f, 0xa4, 0xdb, 0x16, 0x6d, 0xaf, 0x13, 0xfc, 0xde, 0x81, 0xc7, 0x36, 0x3d, 0xf9,
	0x2c, 0xdb, 0x4b, 0xd0, 0x16, 0x56, 0xba, 0x85, 0xdb, 0x26, 0x14, 0x9c, 0x11, 0x9a, 0x4f, 0x49,
	0x65, 0xef, 0xed, 0x7a, 0x7b, 0x35, 0xb4, 0x99, 0xad, 0xc1, 0xfb, 0xcf, 0xc1, 0x94, 0x99, 0xd9,
	0x3c, 0xca, 0xc5, 0xfb, 0x05, 0xf5, 0x8e, 0xc9, 0xf2, 0xab, 0x5f, 0x49, 0xf4, 0x6d, 0x97, 0xbe,
	0x35, 0x95, 0x45, 0x51, 0xcf, 0x1e, 0xcf, 0xff, 0x04, 0x00, 0x00, 0xff, 0xff, 0x1f, 0x4d, 0x0f,
	0xc9, 0x4c, 0x06, 0x00, 0x00,
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// ProxyServiceClient is the client API for ProxyService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type ProxyServiceClient interface {
	GetComponentStates(ctx context.Context, in *internalpb.GetComponentStatesRequest, opts ...grpc.CallOption) (*internalpb.ComponentStates, error)
	GetTimeTickChannel(ctx context.Context, in *internalpb.GetTimeTickChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error)
	GetStatisticsChannel(ctx context.Context, in *internalpb.GetStatisticsChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error)
	RegisterNode(ctx context.Context, in *RegisterNodeRequest, opts ...grpc.CallOption) (*RegisterNodeResponse, error)
	InvalidateCollectionMetaCache(ctx context.Context, in *InvalidateCollMetaCacheRequest, opts ...grpc.CallOption) (*commonpb.Status, error)
}

type proxyServiceClient struct {
	cc *grpc.ClientConn
}

func NewProxyServiceClient(cc *grpc.ClientConn) ProxyServiceClient {
	return &proxyServiceClient{cc}
}

func (c *proxyServiceClient) GetComponentStates(ctx context.Context, in *internalpb.GetComponentStatesRequest, opts ...grpc.CallOption) (*internalpb.ComponentStates, error) {
	out := new(internalpb.ComponentStates)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyService/GetComponentStates", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *proxyServiceClient) GetTimeTickChannel(ctx context.Context, in *internalpb.GetTimeTickChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error) {
	out := new(milvuspb.StringResponse)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyService/GetTimeTickChannel", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *proxyServiceClient) GetStatisticsChannel(ctx context.Context, in *internalpb.GetStatisticsChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error) {
	out := new(milvuspb.StringResponse)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyService/GetStatisticsChannel", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *proxyServiceClient) RegisterNode(ctx context.Context, in *RegisterNodeRequest, opts ...grpc.CallOption) (*RegisterNodeResponse, error) {
	out := new(RegisterNodeResponse)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyService/RegisterNode", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *proxyServiceClient) InvalidateCollectionMetaCache(ctx context.Context, in *InvalidateCollMetaCacheRequest, opts ...grpc.CallOption) (*commonpb.Status, error) {
	out := new(commonpb.Status)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyService/InvalidateCollectionMetaCache", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ProxyServiceServer is the server API for ProxyService service.
type ProxyServiceServer interface {
	GetComponentStates(context.Context, *internalpb.GetComponentStatesRequest) (*internalpb.ComponentStates, error)
	GetTimeTickChannel(context.Context, *internalpb.GetTimeTickChannelRequest) (*milvuspb.StringResponse, error)
	GetStatisticsChannel(context.Context, *internalpb.GetStatisticsChannelRequest) (*milvuspb.StringResponse, error)
	RegisterNode(context.Context, *RegisterNodeRequest) (*RegisterNodeResponse, error)
	InvalidateCollectionMetaCache(context.Context, *InvalidateCollMetaCacheRequest) (*commonpb.Status, error)
}

// UnimplementedProxyServiceServer can be embedded to have forward compatible implementations.
type UnimplementedProxyServiceServer struct {
}

func (*UnimplementedProxyServiceServer) GetComponentStates(ctx context.Context, req *internalpb.GetComponentStatesRequest) (*internalpb.ComponentStates, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetComponentStates not implemented")
}
func (*UnimplementedProxyServiceServer) GetTimeTickChannel(ctx context.Context, req *internalpb.GetTimeTickChannelRequest) (*milvuspb.StringResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetTimeTickChannel not implemented")
}
func (*UnimplementedProxyServiceServer) GetStatisticsChannel(ctx context.Context, req *internalpb.GetStatisticsChannelRequest) (*milvuspb.StringResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetStatisticsChannel not implemented")
}
func (*UnimplementedProxyServiceServer) RegisterNode(ctx context.Context, req *RegisterNodeRequest) (*RegisterNodeResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method RegisterNode not implemented")
}
func (*UnimplementedProxyServiceServer) InvalidateCollectionMetaCache(ctx context.Context, req *InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
	return nil, status.Errorf(codes.Unimplemented, "method InvalidateCollectionMetaCache not implemented")
}

func RegisterProxyServiceServer(s *grpc.Server, srv ProxyServiceServer) {
	s.RegisterService(&_ProxyService_serviceDesc, srv)
}

func _ProxyService_GetComponentStates_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(internalpb.GetComponentStatesRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyServiceServer).GetComponentStates(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyService/GetComponentStates",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyServiceServer).GetComponentStates(ctx, req.(*internalpb.GetComponentStatesRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ProxyService_GetTimeTickChannel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(internalpb.GetTimeTickChannelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyServiceServer).GetTimeTickChannel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyService/GetTimeTickChannel",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyServiceServer).GetTimeTickChannel(ctx, req.(*internalpb.GetTimeTickChannelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ProxyService_GetStatisticsChannel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(internalpb.GetStatisticsChannelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyServiceServer).GetStatisticsChannel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyService/GetStatisticsChannel",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyServiceServer).GetStatisticsChannel(ctx, req.(*internalpb.GetStatisticsChannelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ProxyService_RegisterNode_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RegisterNodeRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyServiceServer).RegisterNode(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyService/RegisterNode",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyServiceServer).RegisterNode(ctx, req.(*RegisterNodeRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ProxyService_InvalidateCollectionMetaCache_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(InvalidateCollMetaCacheRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyServiceServer).InvalidateCollectionMetaCache(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyService/InvalidateCollectionMetaCache",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyServiceServer).InvalidateCollectionMetaCache(ctx, req.(*InvalidateCollMetaCacheRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _ProxyService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "milvus.proto.proxy.ProxyService",
	HandlerType: (*ProxyServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetComponentStates",
			Handler:    _ProxyService_GetComponentStates_Handler,
		},
		{
			MethodName: "GetTimeTickChannel",
			Handler:    _ProxyService_GetTimeTickChannel_Handler,
		},
		{
			MethodName: "GetStatisticsChannel",
			Handler:    _ProxyService_GetStatisticsChannel_Handler,
		},
		{
			MethodName: "RegisterNode",
			Handler:    _ProxyService_RegisterNode_Handler,
		},
		{
			MethodName: "InvalidateCollectionMetaCache",
			Handler:    _ProxyService_InvalidateCollectionMetaCache_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "proxy_service.proto",
}

// ProxyNodeServiceClient is the client API for ProxyNodeService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type ProxyNodeServiceClient interface {
	GetComponentStates(ctx context.Context, in *internalpb.GetComponentStatesRequest, opts ...grpc.CallOption) (*internalpb.ComponentStates, error)
	GetStatisticsChannel(ctx context.Context, in *internalpb.GetStatisticsChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error)
	InvalidateCollectionMetaCache(ctx context.Context, in *InvalidateCollMetaCacheRequest, opts ...grpc.CallOption) (*commonpb.Status, error)
	GetDdChannel(ctx context.Context, in *internalpb.GetDdChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error)
}

type proxyNodeServiceClient struct {
	cc *grpc.ClientConn
}

func NewProxyNodeServiceClient(cc *grpc.ClientConn) ProxyNodeServiceClient {
	return &proxyNodeServiceClient{cc}
}

func (c *proxyNodeServiceClient) GetComponentStates(ctx context.Context, in *internalpb.GetComponentStatesRequest, opts ...grpc.CallOption) (*internalpb.ComponentStates, error) {
	out := new(internalpb.ComponentStates)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyNodeService/GetComponentStates", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *proxyNodeServiceClient) GetStatisticsChannel(ctx context.Context, in *internalpb.GetStatisticsChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error) {
	out := new(milvuspb.StringResponse)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyNodeService/GetStatisticsChannel", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *proxyNodeServiceClient) InvalidateCollectionMetaCache(ctx context.Context, in *InvalidateCollMetaCacheRequest, opts ...grpc.CallOption) (*commonpb.Status, error) {
	out := new(commonpb.Status)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyNodeService/InvalidateCollectionMetaCache", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *proxyNodeServiceClient) GetDdChannel(ctx context.Context, in *internalpb.GetDdChannelRequest, opts ...grpc.CallOption) (*milvuspb.StringResponse, error) {
	out := new(milvuspb.StringResponse)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyNodeService/GetDdChannel", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ProxyNodeServiceServer is the server API for ProxyNodeService service.
type ProxyNodeServiceServer interface {
	GetComponentStates(context.Context, *internalpb.GetComponentStatesRequest) (*internalpb.ComponentStates, error)
	GetStatisticsChannel(context.Context, *internalpb.GetStatisticsChannelRequest) (*milvuspb.StringResponse, error)
	InvalidateCollectionMetaCache(context.Context, *InvalidateCollMetaCacheRequest) (*commonpb.Status, error)
	GetDdChannel(context.Context, *internalpb.GetDdChannelRequest) (*milvuspb.StringResponse, error)
}

// UnimplementedProxyNodeServiceServer can be embedded to have forward compatible implementations.
type UnimplementedProxyNodeServiceServer struct {
}

func (*UnimplementedProxyNodeServiceServer) GetComponentStates(ctx context.Context, req *internalpb.GetComponentStatesRequest) (*internalpb.ComponentStates, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetComponentStates not implemented")
}
func (*UnimplementedProxyNodeServiceServer) GetStatisticsChannel(ctx context.Context, req *internalpb.GetStatisticsChannelRequest) (*milvuspb.StringResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetStatisticsChannel not implemented")
}
func (*UnimplementedProxyNodeServiceServer) InvalidateCollectionMetaCache(ctx context.Context, req *InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
	return nil, status.Errorf(codes.Unimplemented, "method InvalidateCollectionMetaCache not implemented")
}
func (*UnimplementedProxyNodeServiceServer) GetDdChannel(ctx context.Context, req *internalpb.GetDdChannelRequest) (*milvuspb.StringResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetDdChannel not implemented")
}

func RegisterProxyNodeServiceServer(s *grpc.Server, srv ProxyNodeServiceServer) {
	s.RegisterService(&_ProxyNodeService_serviceDesc, srv)
}

func _ProxyNodeService_GetComponentStates_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(internalpb.GetComponentStatesRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyNodeServiceServer).GetComponentStates(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyNodeService/GetComponentStates",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyNodeServiceServer).GetComponentStates(ctx, req.(*internalpb.GetComponentStatesRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ProxyNodeService_GetStatisticsChannel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(internalpb.GetStatisticsChannelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyNodeServiceServer).GetStatisticsChannel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyNodeService/GetStatisticsChannel",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyNodeServiceServer).GetStatisticsChannel(ctx, req.(*internalpb.GetStatisticsChannelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ProxyNodeService_InvalidateCollectionMetaCache_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(InvalidateCollMetaCacheRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyNodeServiceServer).InvalidateCollectionMetaCache(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyNodeService/InvalidateCollectionMetaCache",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyNodeServiceServer).InvalidateCollectionMetaCache(ctx, req.(*InvalidateCollMetaCacheRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ProxyNodeService_GetDdChannel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(internalpb.GetDdChannelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyNodeServiceServer).GetDdChannel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyNodeService/GetDdChannel",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyNodeServiceServer).GetDdChannel(ctx, req.(*internalpb.GetDdChannelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _ProxyNodeService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "milvus.proto.proxy.ProxyNodeService",
	HandlerType: (*ProxyNodeServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetComponentStates",
			Handler:    _ProxyNodeService_GetComponentStates_Handler,
		},
		{
			MethodName: "GetStatisticsChannel",
			Handler:    _ProxyNodeService_GetStatisticsChannel_Handler,
		},
		{
			MethodName: "InvalidateCollectionMetaCache",
			Handler:    _ProxyNodeService_InvalidateCollectionMetaCache_Handler,
		},
		{
			MethodName: "GetDdChannel",
			Handler:    _ProxyNodeService_GetDdChannel_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "proxy_service.proto",
}
