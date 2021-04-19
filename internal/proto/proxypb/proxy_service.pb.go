// Code generated by protoc-gen-go. DO NOT EDIT.
// source: proxy_service.proto

package proxypb

import (
	context "context"
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	commonpb "github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	internalpb2 "github.com/zilliztech/milvus-distributed/internal/proto/internalpb2"
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
	InitParams           *internalpb2.InitParams `protobuf:"bytes,1,opt,name=init_params,json=initParams,proto3" json:"init_params,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                `json:"-"`
	XXX_unrecognized     []byte                  `json:"-"`
	XXX_sizecache        int32                   `json:"-"`
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

func (m *RegisterNodeResponse) GetInitParams() *internalpb2.InitParams {
	if m != nil {
		return m.InitParams
	}
	return nil
}

type RegisterLinkResponse struct {
	Address              *commonpb.Address `protobuf:"bytes,1,opt,name=address,proto3" json:"address,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *RegisterLinkResponse) Reset()         { *m = RegisterLinkResponse{} }
func (m *RegisterLinkResponse) String() string { return proto.CompactTextString(m) }
func (*RegisterLinkResponse) ProtoMessage()    {}
func (*RegisterLinkResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_34ca2fbc94d169de, []int{2}
}

func (m *RegisterLinkResponse) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_RegisterLinkResponse.Unmarshal(m, b)
}
func (m *RegisterLinkResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_RegisterLinkResponse.Marshal(b, m, deterministic)
}
func (m *RegisterLinkResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegisterLinkResponse.Merge(m, src)
}
func (m *RegisterLinkResponse) XXX_Size() int {
	return xxx_messageInfo_RegisterLinkResponse.Size(m)
}
func (m *RegisterLinkResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_RegisterLinkResponse.DiscardUnknown(m)
}

var xxx_messageInfo_RegisterLinkResponse proto.InternalMessageInfo

func (m *RegisterLinkResponse) GetAddress() *commonpb.Address {
	if m != nil {
		return m.Address
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
	return fileDescriptor_34ca2fbc94d169de, []int{3}
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
	proto.RegisterType((*RegisterLinkResponse)(nil), "milvus.proto.proxy.RegisterLinkResponse")
	proto.RegisterType((*InvalidateCollMetaCacheRequest)(nil), "milvus.proto.proxy.InvalidateCollMetaCacheRequest")
}

func init() { proto.RegisterFile("proxy_service.proto", fileDescriptor_34ca2fbc94d169de) }

var fileDescriptor_34ca2fbc94d169de = []byte{
	// 429 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xc4, 0x52, 0x4d, 0x6b, 0x14, 0x41,
	0x10, 0xdd, 0x89, 0x92, 0x60, 0x67, 0x89, 0xd2, 0x11, 0x0c, 0xe3, 0x07, 0x3a, 0x97, 0xe4, 0xe2,
	0x8c, 0xac, 0xe0, 0x55, 0xb2, 0xc1, 0x43, 0xc0, 0x2c, 0x61, 0x02, 0x1e, 0x72, 0x59, 0x7a, 0xa6,
	0x8b, 0xdd, 0xc2, 0x9e, 0xee, 0xb1, 0xbb, 0x66, 0x31, 0xb9, 0x78, 0xf3, 0x17, 0xf8, 0x7f, 0xfc,
	0x6b, 0x32, 0xdd, 0xd9, 0x2f, 0x1c, 0x06, 0xc4, 0x43, 0x6e, 0x53, 0x3d, 0xaf, 0x5e, 0xbd, 0x7a,
	0xf5, 0xd8, 0x61, 0x6d, 0xcd, 0xf7, 0x9b, 0xa9, 0x03, 0xbb, 0xc0, 0x12, 0xd2, 0xda, 0x1a, 0x32,
	0x9c, 0x57, 0xa8, 0x16, 0x8d, 0x0b, 0x55, 0xea, 0x11, 0xf1, 0xb0, 0x34, 0x55, 0x65, 0x74, 0x78,
	0x8b, 0x0f, 0x50, 0x13, 0x58, 0x2d, 0x54, 0xa8, 0x93, 0x1f, 0xec, 0x30, 0x87, 0x19, 0x3a, 0x02,
	0x3b, 0x31, 0x12, 0x72, 0xf8, 0xd6, 0x80, 0x23, 0xfe, 0x8e, 0x3d, 0x2c, 0x84, 0x83, 0xa3, 0xe8,
	0x75, 0x74, 0xb2, 0x3f, 0x7a, 0x91, 0x6e, 0xf1, 0xde, 0x11, 0x5e, 0xb8, 0xd9, 0x58, 0x38, 0xc8,
	0x3d, 0x92, 0x7f, 0x60, 0x7b, 0x42, 0x4a, 0x0b, 0xce, 0x1d, 0xed, 0xf4, 0x34, 0x9d, 0x06, 0x4c,
	0xbe, 0x04, 0x27, 0xd7, 0xec, 0xe9, 0xb6, 0x00, 0x57, 0x1b, 0xed, 0x80, 0x8f, 0xd9, 0x3e, 0x6a,
	0xa4, 0x69, 0x2d, 0xac, 0xa8, 0xdc, 0x9d, 0x90, 0x37, 0xdb, 0x9c, 0xab, 0x5d, 0xce, 0x35, 0xd2,
	0xa5, 0x07, 0xe6, 0x0c, 0x57, 0xdf, 0xc9, 0x64, 0xcd, 0xfd, 0x19, 0xf5, 0xd7, 0x15, 0xf7, 0x86,
	0xd6, 0xe8, 0x5f, 0xb4, 0xfe, 0x8a, 0xd8, 0xab, 0x73, 0xbd, 0x10, 0x0a, 0xa5, 0x20, 0x38, 0x33,
	0x4a, 0x5d, 0x00, 0x89, 0x33, 0x51, 0xce, 0xff, 0xc3, 0xb8, 0x67, 0x6c, 0x4f, 0x16, 0x53, 0x2d,
	0x2a, 0xf0, 0xc6, 0x3d, 0xca, 0x77, 0x65, 0x31, 0x11, 0x15, 0xf0, 0x63, 0xf6, 0xb8, 0x34, 0x4a,
	0x41, 0x49, 0x68, 0x74, 0x00, 0x3c, 0xf0, 0x80, 0x83, 0xf5, 0x73, 0x0b, 0x1c, 0xfd, 0xde, 0x61,
	0xc3, 0xcb, 0xf6, 0xd6, 0x57, 0x21, 0x0c, 0xfc, 0x0b, 0x1b, 0x6e, 0xee, 0xcd, 0xe3, 0x4e, 0x19,
	0x9f, 0xaa, 0x9a, 0x6e, 0xe2, 0x93, 0xf4, 0xef, 0xcc, 0xa4, 0x5d, 0xae, 0x25, 0x03, 0x5e, 0xae,
	0x79, 0xdb, 0x5b, 0xf1, 0xe3, 0xbe, 0xde, 0x8d, 0x38, 0xf5, 0x0f, 0xd9, 0x3c, 0x7b, 0x32, 0xe0,
	0x96, 0xbd, 0xdc, 0xf6, 0x38, 0x6c, 0xba, 0x72, 0x9a, 0x8f, 0xba, 0xc8, 0xfa, 0xcf, 0x12, 0x3f,
	0xef, 0x74, 0xe0, 0x8a, 0x04, 0x35, 0x2e, 0x19, 0x8c, 0x7e, 0x46, 0xec, 0x89, 0x77, 0xb0, 0xd5,
	0xb2, 0x74, 0xf1, 0x1e, 0x84, 0x8c, 0x4f, 0xaf, 0x3f, 0xce, 0x90, 0xe6, 0x4d, 0xd1, 0xfe, 0xc9,
	0x6e, 0x51, 0x29, 0xbc, 0x25, 0x28, 0xe7, 0x59, 0xe8, 0x7a, 0x2b, 0xd1, 0x91, 0xc5, 0xa2, 0x21,
	0x90, 0xd9, 0x32, 0xfd, 0x99, 0xa7, 0xca, 0xfc, 0xf8, 0xba, 0x28, 0x76, 0x7d, 0xf9, 0xfe, 0x4f,
	0x00, 0x00, 0x00, 0xff, 0xff, 0x25, 0x02, 0x4c, 0xbc, 0x21, 0x04, 0x00, 0x00,
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
	RegisterLink(ctx context.Context, in *commonpb.Empty, opts ...grpc.CallOption) (*RegisterLinkResponse, error)
	RegisterNode(ctx context.Context, in *RegisterNodeRequest, opts ...grpc.CallOption) (*RegisterNodeResponse, error)
	InvalidateCollectionMetaCache(ctx context.Context, in *InvalidateCollMetaCacheRequest, opts ...grpc.CallOption) (*commonpb.Status, error)
}

type proxyServiceClient struct {
	cc *grpc.ClientConn
}

func NewProxyServiceClient(cc *grpc.ClientConn) ProxyServiceClient {
	return &proxyServiceClient{cc}
}

func (c *proxyServiceClient) RegisterLink(ctx context.Context, in *commonpb.Empty, opts ...grpc.CallOption) (*RegisterLinkResponse, error) {
	out := new(RegisterLinkResponse)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyService/RegisterLink", in, out, opts...)
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
	RegisterLink(context.Context, *commonpb.Empty) (*RegisterLinkResponse, error)
	RegisterNode(context.Context, *RegisterNodeRequest) (*RegisterNodeResponse, error)
	InvalidateCollectionMetaCache(context.Context, *InvalidateCollMetaCacheRequest) (*commonpb.Status, error)
}

// UnimplementedProxyServiceServer can be embedded to have forward compatible implementations.
type UnimplementedProxyServiceServer struct {
}

func (*UnimplementedProxyServiceServer) RegisterLink(ctx context.Context, req *commonpb.Empty) (*RegisterLinkResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method RegisterLink not implemented")
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

func _ProxyService_RegisterLink_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(commonpb.Empty)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ProxyServiceServer).RegisterLink(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/milvus.proto.proxy.ProxyService/RegisterLink",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ProxyServiceServer).RegisterLink(ctx, req.(*commonpb.Empty))
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
			MethodName: "RegisterLink",
			Handler:    _ProxyService_RegisterLink_Handler,
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
	InvalidateCollectionMetaCache(ctx context.Context, in *InvalidateCollMetaCacheRequest, opts ...grpc.CallOption) (*commonpb.Status, error)
}

type proxyNodeServiceClient struct {
	cc *grpc.ClientConn
}

func NewProxyNodeServiceClient(cc *grpc.ClientConn) ProxyNodeServiceClient {
	return &proxyNodeServiceClient{cc}
}

func (c *proxyNodeServiceClient) InvalidateCollectionMetaCache(ctx context.Context, in *InvalidateCollMetaCacheRequest, opts ...grpc.CallOption) (*commonpb.Status, error) {
	out := new(commonpb.Status)
	err := c.cc.Invoke(ctx, "/milvus.proto.proxy.ProxyNodeService/InvalidateCollectionMetaCache", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ProxyNodeServiceServer is the server API for ProxyNodeService service.
type ProxyNodeServiceServer interface {
	InvalidateCollectionMetaCache(context.Context, *InvalidateCollMetaCacheRequest) (*commonpb.Status, error)
}

// UnimplementedProxyNodeServiceServer can be embedded to have forward compatible implementations.
type UnimplementedProxyNodeServiceServer struct {
}

func (*UnimplementedProxyNodeServiceServer) InvalidateCollectionMetaCache(ctx context.Context, req *InvalidateCollMetaCacheRequest) (*commonpb.Status, error) {
	return nil, status.Errorf(codes.Unimplemented, "method InvalidateCollectionMetaCache not implemented")
}

func RegisterProxyNodeServiceServer(s *grpc.Server, srv ProxyNodeServiceServer) {
	s.RegisterService(&_ProxyNodeService_serviceDesc, srv)
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

var _ProxyNodeService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "milvus.proto.proxy.ProxyNodeService",
	HandlerType: (*ProxyNodeServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "InvalidateCollectionMetaCache",
			Handler:    _ProxyNodeService_InvalidateCollectionMetaCache_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "proxy_service.proto",
}
