// Code generated by protoc-gen-go. DO NOT EDIT.
// source: schema.proto

package schemapb

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	commonpb "github.com/milvus-io/milvus/internal/proto/commonpb"
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

//*
// @brief Field data type
type DataType int32

const (
	DataType_None         DataType = 0
	DataType_Bool         DataType = 1
	DataType_Int8         DataType = 2
	DataType_Int16        DataType = 3
	DataType_Int32        DataType = 4
	DataType_Int64        DataType = 5
	DataType_Float        DataType = 10
	DataType_Double       DataType = 11
	DataType_String       DataType = 20
	DataType_BinaryVector DataType = 100
	DataType_FloatVector  DataType = 101
)

var DataType_name = map[int32]string{
	0:   "None",
	1:   "Bool",
	2:   "Int8",
	3:   "Int16",
	4:   "Int32",
	5:   "Int64",
	10:  "Float",
	11:  "Double",
	20:  "String",
	100: "BinaryVector",
	101: "FloatVector",
}

var DataType_value = map[string]int32{
	"None":         0,
	"Bool":         1,
	"Int8":         2,
	"Int16":        3,
	"Int32":        4,
	"Int64":        5,
	"Float":        10,
	"Double":       11,
	"String":       20,
	"BinaryVector": 100,
	"FloatVector":  101,
}

func (x DataType) String() string {
	return proto.EnumName(DataType_name, int32(x))
}

func (DataType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_1c5fb4d8cc22d66a, []int{0}
}

//*
// @brief Field schema
type FieldSchema struct {
	FieldID              int64                    `protobuf:"varint,1,opt,name=fieldID,proto3" json:"fieldID,omitempty"`
	Name                 string                   `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
	IsPrimaryKey         bool                     `protobuf:"varint,3,opt,name=is_primary_key,json=isPrimaryKey,proto3" json:"is_primary_key,omitempty"`
	Description          string                   `protobuf:"bytes,4,opt,name=description,proto3" json:"description,omitempty"`
	DataType             DataType                 `protobuf:"varint,5,opt,name=data_type,json=dataType,proto3,enum=milvus.proto.schema.DataType" json:"data_type,omitempty"`
	TypeParams           []*commonpb.KeyValuePair `protobuf:"bytes,6,rep,name=type_params,json=typeParams,proto3" json:"type_params,omitempty"`
	IndexParams          []*commonpb.KeyValuePair `protobuf:"bytes,7,rep,name=index_params,json=indexParams,proto3" json:"index_params,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                 `json:"-"`
	XXX_unrecognized     []byte                   `json:"-"`
	XXX_sizecache        int32                    `json:"-"`
}

func (m *FieldSchema) Reset()         { *m = FieldSchema{} }
func (m *FieldSchema) String() string { return proto.CompactTextString(m) }
func (*FieldSchema) ProtoMessage()    {}
func (*FieldSchema) Descriptor() ([]byte, []int) {
	return fileDescriptor_1c5fb4d8cc22d66a, []int{0}
}

func (m *FieldSchema) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_FieldSchema.Unmarshal(m, b)
}
func (m *FieldSchema) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_FieldSchema.Marshal(b, m, deterministic)
}
func (m *FieldSchema) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FieldSchema.Merge(m, src)
}
func (m *FieldSchema) XXX_Size() int {
	return xxx_messageInfo_FieldSchema.Size(m)
}
func (m *FieldSchema) XXX_DiscardUnknown() {
	xxx_messageInfo_FieldSchema.DiscardUnknown(m)
}

var xxx_messageInfo_FieldSchema proto.InternalMessageInfo

func (m *FieldSchema) GetFieldID() int64 {
	if m != nil {
		return m.FieldID
	}
	return 0
}

func (m *FieldSchema) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *FieldSchema) GetIsPrimaryKey() bool {
	if m != nil {
		return m.IsPrimaryKey
	}
	return false
}

func (m *FieldSchema) GetDescription() string {
	if m != nil {
		return m.Description
	}
	return ""
}

func (m *FieldSchema) GetDataType() DataType {
	if m != nil {
		return m.DataType
	}
	return DataType_None
}

func (m *FieldSchema) GetTypeParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.TypeParams
	}
	return nil
}

func (m *FieldSchema) GetIndexParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.IndexParams
	}
	return nil
}

//*
// @brief Collection schema
type CollectionSchema struct {
	Name                 string         `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Description          string         `protobuf:"bytes,2,opt,name=description,proto3" json:"description,omitempty"`
	AutoID               bool           `protobuf:"varint,3,opt,name=autoID,proto3" json:"autoID,omitempty"`
	Fields               []*FieldSchema `protobuf:"bytes,4,rep,name=fields,proto3" json:"fields,omitempty"`
	XXX_NoUnkeyedLiteral struct{}       `json:"-"`
	XXX_unrecognized     []byte         `json:"-"`
	XXX_sizecache        int32          `json:"-"`
}

func (m *CollectionSchema) Reset()         { *m = CollectionSchema{} }
func (m *CollectionSchema) String() string { return proto.CompactTextString(m) }
func (*CollectionSchema) ProtoMessage()    {}
func (*CollectionSchema) Descriptor() ([]byte, []int) {
	return fileDescriptor_1c5fb4d8cc22d66a, []int{1}
}

func (m *CollectionSchema) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_CollectionSchema.Unmarshal(m, b)
}
func (m *CollectionSchema) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_CollectionSchema.Marshal(b, m, deterministic)
}
func (m *CollectionSchema) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CollectionSchema.Merge(m, src)
}
func (m *CollectionSchema) XXX_Size() int {
	return xxx_messageInfo_CollectionSchema.Size(m)
}
func (m *CollectionSchema) XXX_DiscardUnknown() {
	xxx_messageInfo_CollectionSchema.DiscardUnknown(m)
}

var xxx_messageInfo_CollectionSchema proto.InternalMessageInfo

func (m *CollectionSchema) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *CollectionSchema) GetDescription() string {
	if m != nil {
		return m.Description
	}
	return ""
}

func (m *CollectionSchema) GetAutoID() bool {
	if m != nil {
		return m.AutoID
	}
	return false
}

func (m *CollectionSchema) GetFields() []*FieldSchema {
	if m != nil {
		return m.Fields
	}
	return nil
}

func init() {
	proto.RegisterEnum("milvus.proto.schema.DataType", DataType_name, DataType_value)
	proto.RegisterType((*FieldSchema)(nil), "milvus.proto.schema.FieldSchema")
	proto.RegisterType((*CollectionSchema)(nil), "milvus.proto.schema.CollectionSchema")
}

func init() { proto.RegisterFile("schema.proto", fileDescriptor_1c5fb4d8cc22d66a) }

var fileDescriptor_1c5fb4d8cc22d66a = []byte{
	// 443 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x91, 0x41, 0x8b, 0xd4, 0x30,
	0x1c, 0xc5, 0xcd, 0x4c, 0xa7, 0x3b, 0xf3, 0x6f, 0x59, 0x43, 0x14, 0x29, 0x82, 0x50, 0x17, 0x0f,
	0x45, 0x70, 0x06, 0x67, 0x74, 0x59, 0x3c, 0x8e, 0x65, 0x61, 0x58, 0x90, 0xa1, 0x2b, 0x7b, 0xf0,
	0x32, 0x64, 0xda, 0xb8, 0x1b, 0x4c, 0x93, 0x92, 0xa6, 0x62, 0x3f, 0x85, 0x77, 0xbf, 0x91, 0xdf,
	0x4a, 0x92, 0x66, 0x60, 0xd4, 0x3d, 0xec, 0xed, 0xbd, 0x7f, 0xf2, 0xfe, 0xe4, 0xfd, 0x02, 0x71,
	0x5b, 0xde, 0xb1, 0x9a, 0xce, 0x1b, 0xad, 0x8c, 0x22, 0x4f, 0x6a, 0x2e, 0xbe, 0x77, 0xed, 0xe0,
	0xe6, 0xc3, 0xd1, 0xf3, 0xb8, 0x54, 0x75, 0xad, 0xe4, 0x30, 0x3c, 0xfb, 0x3d, 0x82, 0xe8, 0x92,
	0x33, 0x51, 0x5d, 0xbb, 0x53, 0x92, 0xc0, 0xc9, 0x57, 0x6b, 0x37, 0x79, 0x82, 0x52, 0x94, 0x8d,
	0x8b, 0x83, 0x25, 0x04, 0x02, 0x49, 0x6b, 0x96, 0x8c, 0x52, 0x94, 0xcd, 0x0a, 0xa7, 0xc9, 0x2b,
	0x38, 0xe5, 0xed, 0xae, 0xd1, 0xbc, 0xa6, 0xba, 0xdf, 0x7d, 0x63, 0x7d, 0x32, 0x4e, 0x51, 0x36,
	0x2d, 0x62, 0xde, 0x6e, 0x87, 0xe1, 0x15, 0xeb, 0x49, 0x0a, 0x51, 0xc5, 0xda, 0x52, 0xf3, 0xc6,
	0x70, 0x25, 0x93, 0xc0, 0x2d, 0x38, 0x1e, 0x91, 0x0f, 0x30, 0xab, 0xa8, 0xa1, 0x3b, 0xd3, 0x37,
	0x2c, 0x99, 0xa4, 0x28, 0x3b, 0x5d, 0xbe, 0x98, 0xdf, 0xf3, 0xf8, 0x79, 0x4e, 0x0d, 0xfd, 0xdc,
	0x37, 0xac, 0x98, 0x56, 0x5e, 0x91, 0x35, 0x44, 0x36, 0xb6, 0x6b, 0xa8, 0xa6, 0x75, 0x9b, 0x84,
	0xe9, 0x38, 0x8b, 0x96, 0x2f, 0xff, 0x4e, 0xfb, 0xca, 0x57, 0xac, 0xbf, 0xa1, 0xa2, 0x63, 0x5b,
	0xca, 0x75, 0x01, 0x36, 0xb5, 0x75, 0x21, 0x92, 0x43, 0xcc, 0x65, 0xc5, 0x7e, 0x1c, 0x96, 0x9c,
	0x3c, 0x74, 0x49, 0xe4, 0x62, 0xc3, 0x96, 0xb3, 0x5f, 0x08, 0xf0, 0x47, 0x25, 0x04, 0x2b, 0x6d,
	0x29, 0x0f, 0xf4, 0x80, 0x0d, 0x1d, 0x61, 0xfb, 0x07, 0xc8, 0xe8, 0x7f, 0x20, 0xcf, 0x20, 0xa4,
	0x9d, 0x51, 0x9b, 0xdc, 0x03, 0xf5, 0x8e, 0x5c, 0x40, 0xe8, 0xfe, 0xa3, 0x4d, 0x02, 0xf7, 0xc4,
	0xf4, 0x5e, 0x4a, 0x47, 0x1f, 0x5a, 0xf8, 0xfb, 0xaf, 0x7f, 0x22, 0x98, 0x1e, 0xe8, 0x91, 0x29,
	0x04, 0x9f, 0x94, 0x64, 0xf8, 0x91, 0x55, 0x6b, 0xa5, 0x04, 0x46, 0x56, 0x6d, 0xa4, 0xb9, 0xc0,
	0x23, 0x32, 0x83, 0xc9, 0x46, 0x9a, 0xb7, 0xe7, 0x78, 0xec, 0xe5, 0x6a, 0x89, 0x03, 0x2f, 0xcf,
	0xdf, 0xe1, 0x89, 0x95, 0x97, 0x42, 0x51, 0x83, 0x81, 0x00, 0x84, 0xb9, 0xea, 0xf6, 0x82, 0xe1,
	0xc8, 0xea, 0x6b, 0xa3, 0xb9, 0xbc, 0xc5, 0x4f, 0x09, 0x86, 0x78, 0xcd, 0x25, 0xd5, 0xfd, 0x0d,
	0x2b, 0x8d, 0xd2, 0xb8, 0x22, 0x8f, 0x21, 0x72, 0x21, 0x3f, 0x60, 0xeb, 0xf7, 0x5f, 0x56, 0xb7,
	0xdc, 0xdc, 0x75, 0x7b, 0x4b, 0x76, 0x31, 0xf4, 0x78, 0xc3, 0x95, 0x57, 0x0b, 0x2e, 0x0d, 0xd3,
	0x92, 0x8a, 0x85, 0xab, 0xb6, 0x18, 0xaa, 0x35, 0xfb, 0x7d, 0xe8, 0xfc, 0xea, 0x4f, 0x00, 0x00,
	0x00, 0xff, 0xff, 0xa8, 0x56, 0xc7, 0x76, 0xeb, 0x02, 0x00, 0x00,
}
