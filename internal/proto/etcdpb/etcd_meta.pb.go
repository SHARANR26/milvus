// Code generated by protoc-gen-go. DO NOT EDIT.
// source: etcd_meta.proto

package etcdpb

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	commonpb "github.com/zilliztech/milvus-distributed/internal/proto/commonpb"
	schemapb "github.com/zilliztech/milvus-distributed/internal/proto/schemapb"
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

type TenantMeta struct {
	ID                   int64    `protobuf:"varint,1,opt,name=ID,proto3" json:"ID,omitempty"`
	NumQueryNodes        int64    `protobuf:"varint,2,opt,name=num_query_nodes,json=numQueryNodes,proto3" json:"num_query_nodes,omitempty"`
	InsertChannelIDs     []string `protobuf:"bytes,3,rep,name=insert_channelIDs,json=insertChannelIDs,proto3" json:"insert_channelIDs,omitempty"`
	QueryChannelID       string   `protobuf:"bytes,4,opt,name=query_channelID,json=queryChannelID,proto3" json:"query_channelID,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *TenantMeta) Reset()         { *m = TenantMeta{} }
func (m *TenantMeta) String() string { return proto.CompactTextString(m) }
func (*TenantMeta) ProtoMessage()    {}
func (*TenantMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{0}
}

func (m *TenantMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_TenantMeta.Unmarshal(m, b)
}
func (m *TenantMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_TenantMeta.Marshal(b, m, deterministic)
}
func (m *TenantMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TenantMeta.Merge(m, src)
}
func (m *TenantMeta) XXX_Size() int {
	return xxx_messageInfo_TenantMeta.Size(m)
}
func (m *TenantMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_TenantMeta.DiscardUnknown(m)
}

var xxx_messageInfo_TenantMeta proto.InternalMessageInfo

func (m *TenantMeta) GetID() int64 {
	if m != nil {
		return m.ID
	}
	return 0
}

func (m *TenantMeta) GetNumQueryNodes() int64 {
	if m != nil {
		return m.NumQueryNodes
	}
	return 0
}

func (m *TenantMeta) GetInsertChannelIDs() []string {
	if m != nil {
		return m.InsertChannelIDs
	}
	return nil
}

func (m *TenantMeta) GetQueryChannelID() string {
	if m != nil {
		return m.QueryChannelID
	}
	return ""
}

type ProxyMeta struct {
	ID                   int64             `protobuf:"varint,1,opt,name=ID,proto3" json:"ID,omitempty"`
	Address              *commonpb.Address `protobuf:"bytes,2,opt,name=address,proto3" json:"address,omitempty"`
	ResultChannelIDs     []string          `protobuf:"bytes,3,rep,name=result_channelIDs,json=resultChannelIDs,proto3" json:"result_channelIDs,omitempty"`
	XXX_NoUnkeyedLiteral struct{}          `json:"-"`
	XXX_unrecognized     []byte            `json:"-"`
	XXX_sizecache        int32             `json:"-"`
}

func (m *ProxyMeta) Reset()         { *m = ProxyMeta{} }
func (m *ProxyMeta) String() string { return proto.CompactTextString(m) }
func (*ProxyMeta) ProtoMessage()    {}
func (*ProxyMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{1}
}

func (m *ProxyMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ProxyMeta.Unmarshal(m, b)
}
func (m *ProxyMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ProxyMeta.Marshal(b, m, deterministic)
}
func (m *ProxyMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ProxyMeta.Merge(m, src)
}
func (m *ProxyMeta) XXX_Size() int {
	return xxx_messageInfo_ProxyMeta.Size(m)
}
func (m *ProxyMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_ProxyMeta.DiscardUnknown(m)
}

var xxx_messageInfo_ProxyMeta proto.InternalMessageInfo

func (m *ProxyMeta) GetID() int64 {
	if m != nil {
		return m.ID
	}
	return 0
}

func (m *ProxyMeta) GetAddress() *commonpb.Address {
	if m != nil {
		return m.Address
	}
	return nil
}

func (m *ProxyMeta) GetResultChannelIDs() []string {
	if m != nil {
		return m.ResultChannelIDs
	}
	return nil
}

type PartitionInfo struct {
	PartitionName        string   `protobuf:"bytes,1,opt,name=partition_name,json=partitionName,proto3" json:"partition_name,omitempty"`
	PartitionID          int64    `protobuf:"varint,2,opt,name=partitionID,proto3" json:"partitionID,omitempty"`
	SegmentIDs           []int64  `protobuf:"varint,3,rep,packed,name=segmentIDs,proto3" json:"segmentIDs,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *PartitionInfo) Reset()         { *m = PartitionInfo{} }
func (m *PartitionInfo) String() string { return proto.CompactTextString(m) }
func (*PartitionInfo) ProtoMessage()    {}
func (*PartitionInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{2}
}

func (m *PartitionInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_PartitionInfo.Unmarshal(m, b)
}
func (m *PartitionInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_PartitionInfo.Marshal(b, m, deterministic)
}
func (m *PartitionInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_PartitionInfo.Merge(m, src)
}
func (m *PartitionInfo) XXX_Size() int {
	return xxx_messageInfo_PartitionInfo.Size(m)
}
func (m *PartitionInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_PartitionInfo.DiscardUnknown(m)
}

var xxx_messageInfo_PartitionInfo proto.InternalMessageInfo

func (m *PartitionInfo) GetPartitionName() string {
	if m != nil {
		return m.PartitionName
	}
	return ""
}

func (m *PartitionInfo) GetPartitionID() int64 {
	if m != nil {
		return m.PartitionID
	}
	return 0
}

func (m *PartitionInfo) GetSegmentIDs() []int64 {
	if m != nil {
		return m.SegmentIDs
	}
	return nil
}

type IndexParamsInfo struct {
	FiledID              int64                    `protobuf:"varint,1,opt,name=filedID,proto3" json:"filedID,omitempty"`
	IndexParams          []*commonpb.KeyValuePair `protobuf:"bytes,2,rep,name=index_params,json=indexParams,proto3" json:"index_params,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                 `json:"-"`
	XXX_unrecognized     []byte                   `json:"-"`
	XXX_sizecache        int32                    `json:"-"`
}

func (m *IndexParamsInfo) Reset()         { *m = IndexParamsInfo{} }
func (m *IndexParamsInfo) String() string { return proto.CompactTextString(m) }
func (*IndexParamsInfo) ProtoMessage()    {}
func (*IndexParamsInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{3}
}

func (m *IndexParamsInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_IndexParamsInfo.Unmarshal(m, b)
}
func (m *IndexParamsInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_IndexParamsInfo.Marshal(b, m, deterministic)
}
func (m *IndexParamsInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_IndexParamsInfo.Merge(m, src)
}
func (m *IndexParamsInfo) XXX_Size() int {
	return xxx_messageInfo_IndexParamsInfo.Size(m)
}
func (m *IndexParamsInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_IndexParamsInfo.DiscardUnknown(m)
}

var xxx_messageInfo_IndexParamsInfo proto.InternalMessageInfo

func (m *IndexParamsInfo) GetFiledID() int64 {
	if m != nil {
		return m.FiledID
	}
	return 0
}

func (m *IndexParamsInfo) GetIndexParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.IndexParams
	}
	return nil
}

type CollectionInfo struct {
	ID                   int64                      `protobuf:"varint,1,opt,name=ID,proto3" json:"ID,omitempty"`
	Schema               *schemapb.CollectionSchema `protobuf:"bytes,2,opt,name=schema,proto3" json:"schema,omitempty"`
	CreateTime           uint64                     `protobuf:"varint,3,opt,name=create_time,json=createTime,proto3" json:"create_time,omitempty"`
	PartitionIDs         []int64                    `protobuf:"varint,4,rep,packed,name=partitionIDs,proto3" json:"partitionIDs,omitempty"`
	IndexParams          []*IndexParamsInfo         `protobuf:"bytes,5,rep,name=index_params,json=indexParams,proto3" json:"index_params,omitempty"`
	IndexNames           []string                   `protobuf:"bytes,6,rep,name=index_names,json=indexNames,proto3" json:"index_names,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                   `json:"-"`
	XXX_unrecognized     []byte                     `json:"-"`
	XXX_sizecache        int32                      `json:"-"`
}

func (m *CollectionInfo) Reset()         { *m = CollectionInfo{} }
func (m *CollectionInfo) String() string { return proto.CompactTextString(m) }
func (*CollectionInfo) ProtoMessage()    {}
func (*CollectionInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{4}
}

func (m *CollectionInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_CollectionInfo.Unmarshal(m, b)
}
func (m *CollectionInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_CollectionInfo.Marshal(b, m, deterministic)
}
func (m *CollectionInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CollectionInfo.Merge(m, src)
}
func (m *CollectionInfo) XXX_Size() int {
	return xxx_messageInfo_CollectionInfo.Size(m)
}
func (m *CollectionInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_CollectionInfo.DiscardUnknown(m)
}

var xxx_messageInfo_CollectionInfo proto.InternalMessageInfo

func (m *CollectionInfo) GetID() int64 {
	if m != nil {
		return m.ID
	}
	return 0
}

func (m *CollectionInfo) GetSchema() *schemapb.CollectionSchema {
	if m != nil {
		return m.Schema
	}
	return nil
}

func (m *CollectionInfo) GetCreateTime() uint64 {
	if m != nil {
		return m.CreateTime
	}
	return 0
}

func (m *CollectionInfo) GetPartitionIDs() []int64 {
	if m != nil {
		return m.PartitionIDs
	}
	return nil
}

func (m *CollectionInfo) GetIndexParams() []*IndexParamsInfo {
	if m != nil {
		return m.IndexParams
	}
	return nil
}

func (m *CollectionInfo) GetIndexNames() []string {
	if m != nil {
		return m.IndexNames
	}
	return nil
}

type IndexInfo struct {
	IndexName            string                   `protobuf:"bytes,1,opt,name=index_name,json=indexName,proto3" json:"index_name,omitempty"`
	IndexID              int64                    `protobuf:"varint,2,opt,name=indexID,proto3" json:"indexID,omitempty"`
	IndexParams          []*commonpb.KeyValuePair `protobuf:"bytes,3,rep,name=index_params,json=indexParams,proto3" json:"index_params,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                 `json:"-"`
	XXX_unrecognized     []byte                   `json:"-"`
	XXX_sizecache        int32                    `json:"-"`
}

func (m *IndexInfo) Reset()         { *m = IndexInfo{} }
func (m *IndexInfo) String() string { return proto.CompactTextString(m) }
func (*IndexInfo) ProtoMessage()    {}
func (*IndexInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{5}
}

func (m *IndexInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_IndexInfo.Unmarshal(m, b)
}
func (m *IndexInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_IndexInfo.Marshal(b, m, deterministic)
}
func (m *IndexInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_IndexInfo.Merge(m, src)
}
func (m *IndexInfo) XXX_Size() int {
	return xxx_messageInfo_IndexInfo.Size(m)
}
func (m *IndexInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_IndexInfo.DiscardUnknown(m)
}

var xxx_messageInfo_IndexInfo proto.InternalMessageInfo

func (m *IndexInfo) GetIndexName() string {
	if m != nil {
		return m.IndexName
	}
	return ""
}

func (m *IndexInfo) GetIndexID() int64 {
	if m != nil {
		return m.IndexID
	}
	return 0
}

func (m *IndexInfo) GetIndexParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.IndexParams
	}
	return nil
}

type SegmentIndexInfo struct {
	SegmentID            int64    `protobuf:"varint,1,opt,name=segmentID,proto3" json:"segmentID,omitempty"`
	FieldID              int64    `protobuf:"varint,2,opt,name=fieldID,proto3" json:"fieldID,omitempty"`
	IndexID              int64    `protobuf:"varint,3,opt,name=indexID,proto3" json:"indexID,omitempty"`
	BuildID              int64    `protobuf:"varint,4,opt,name=buildID,proto3" json:"buildID,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *SegmentIndexInfo) Reset()         { *m = SegmentIndexInfo{} }
func (m *SegmentIndexInfo) String() string { return proto.CompactTextString(m) }
func (*SegmentIndexInfo) ProtoMessage()    {}
func (*SegmentIndexInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{6}
}

func (m *SegmentIndexInfo) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_SegmentIndexInfo.Unmarshal(m, b)
}
func (m *SegmentIndexInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_SegmentIndexInfo.Marshal(b, m, deterministic)
}
func (m *SegmentIndexInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SegmentIndexInfo.Merge(m, src)
}
func (m *SegmentIndexInfo) XXX_Size() int {
	return xxx_messageInfo_SegmentIndexInfo.Size(m)
}
func (m *SegmentIndexInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_SegmentIndexInfo.DiscardUnknown(m)
}

var xxx_messageInfo_SegmentIndexInfo proto.InternalMessageInfo

func (m *SegmentIndexInfo) GetSegmentID() int64 {
	if m != nil {
		return m.SegmentID
	}
	return 0
}

func (m *SegmentIndexInfo) GetFieldID() int64 {
	if m != nil {
		return m.FieldID
	}
	return 0
}

func (m *SegmentIndexInfo) GetIndexID() int64 {
	if m != nil {
		return m.IndexID
	}
	return 0
}

func (m *SegmentIndexInfo) GetBuildID() int64 {
	if m != nil {
		return m.BuildID
	}
	return 0
}

type CollectionMeta struct {
	ID                   int64                      `protobuf:"varint,1,opt,name=ID,proto3" json:"ID,omitempty"`
	Schema               *schemapb.CollectionSchema `protobuf:"bytes,2,opt,name=schema,proto3" json:"schema,omitempty"`
	CreateTime           uint64                     `protobuf:"varint,3,opt,name=create_time,json=createTime,proto3" json:"create_time,omitempty"`
	SegmentIDs           []int64                    `protobuf:"varint,4,rep,packed,name=segmentIDs,proto3" json:"segmentIDs,omitempty"`
	PartitionTags        []string                   `protobuf:"bytes,5,rep,name=partition_tags,json=partitionTags,proto3" json:"partition_tags,omitempty"`
	PartitionIDs         []int64                    `protobuf:"varint,6,rep,packed,name=partitionIDs,proto3" json:"partitionIDs,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                   `json:"-"`
	XXX_unrecognized     []byte                     `json:"-"`
	XXX_sizecache        int32                      `json:"-"`
}

func (m *CollectionMeta) Reset()         { *m = CollectionMeta{} }
func (m *CollectionMeta) String() string { return proto.CompactTextString(m) }
func (*CollectionMeta) ProtoMessage()    {}
func (*CollectionMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{7}
}

func (m *CollectionMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_CollectionMeta.Unmarshal(m, b)
}
func (m *CollectionMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_CollectionMeta.Marshal(b, m, deterministic)
}
func (m *CollectionMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CollectionMeta.Merge(m, src)
}
func (m *CollectionMeta) XXX_Size() int {
	return xxx_messageInfo_CollectionMeta.Size(m)
}
func (m *CollectionMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_CollectionMeta.DiscardUnknown(m)
}

var xxx_messageInfo_CollectionMeta proto.InternalMessageInfo

func (m *CollectionMeta) GetID() int64 {
	if m != nil {
		return m.ID
	}
	return 0
}

func (m *CollectionMeta) GetSchema() *schemapb.CollectionSchema {
	if m != nil {
		return m.Schema
	}
	return nil
}

func (m *CollectionMeta) GetCreateTime() uint64 {
	if m != nil {
		return m.CreateTime
	}
	return 0
}

func (m *CollectionMeta) GetSegmentIDs() []int64 {
	if m != nil {
		return m.SegmentIDs
	}
	return nil
}

func (m *CollectionMeta) GetPartitionTags() []string {
	if m != nil {
		return m.PartitionTags
	}
	return nil
}

func (m *CollectionMeta) GetPartitionIDs() []int64 {
	if m != nil {
		return m.PartitionIDs
	}
	return nil
}

type FieldBinlogFiles struct {
	FieldID              int64    `protobuf:"varint,1,opt,name=fieldID,proto3" json:"fieldID,omitempty"`
	BinlogFiles          []string `protobuf:"bytes,2,rep,name=binlog_files,json=binlogFiles,proto3" json:"binlog_files,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *FieldBinlogFiles) Reset()         { *m = FieldBinlogFiles{} }
func (m *FieldBinlogFiles) String() string { return proto.CompactTextString(m) }
func (*FieldBinlogFiles) ProtoMessage()    {}
func (*FieldBinlogFiles) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{8}
}

func (m *FieldBinlogFiles) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_FieldBinlogFiles.Unmarshal(m, b)
}
func (m *FieldBinlogFiles) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_FieldBinlogFiles.Marshal(b, m, deterministic)
}
func (m *FieldBinlogFiles) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FieldBinlogFiles.Merge(m, src)
}
func (m *FieldBinlogFiles) XXX_Size() int {
	return xxx_messageInfo_FieldBinlogFiles.Size(m)
}
func (m *FieldBinlogFiles) XXX_DiscardUnknown() {
	xxx_messageInfo_FieldBinlogFiles.DiscardUnknown(m)
}

var xxx_messageInfo_FieldBinlogFiles proto.InternalMessageInfo

func (m *FieldBinlogFiles) GetFieldID() int64 {
	if m != nil {
		return m.FieldID
	}
	return 0
}

func (m *FieldBinlogFiles) GetBinlogFiles() []string {
	if m != nil {
		return m.BinlogFiles
	}
	return nil
}

type SegmentMeta struct {
	SegmentID            int64               `protobuf:"varint,1,opt,name=segmentID,proto3" json:"segmentID,omitempty"`
	CollectionID         int64               `protobuf:"varint,2,opt,name=collectionID,proto3" json:"collectionID,omitempty"`
	PartitionTag         string              `protobuf:"bytes,3,opt,name=partition_tag,json=partitionTag,proto3" json:"partition_tag,omitempty"`
	ChannelStart         int32               `protobuf:"varint,4,opt,name=channel_start,json=channelStart,proto3" json:"channel_start,omitempty"`
	ChannelEnd           int32               `protobuf:"varint,5,opt,name=channel_end,json=channelEnd,proto3" json:"channel_end,omitempty"`
	OpenTime             uint64              `protobuf:"varint,6,opt,name=open_time,json=openTime,proto3" json:"open_time,omitempty"`
	CloseTime            uint64              `protobuf:"varint,7,opt,name=close_time,json=closeTime,proto3" json:"close_time,omitempty"`
	NumRows              int64               `protobuf:"varint,8,opt,name=num_rows,json=numRows,proto3" json:"num_rows,omitempty"`
	MemSize              int64               `protobuf:"varint,9,opt,name=mem_size,json=memSize,proto3" json:"mem_size,omitempty"`
	BinlogFilePaths      []*FieldBinlogFiles `protobuf:"bytes,10,rep,name=binlog_file_paths,json=binlogFilePaths,proto3" json:"binlog_file_paths,omitempty"`
	XXX_NoUnkeyedLiteral struct{}            `json:"-"`
	XXX_unrecognized     []byte              `json:"-"`
	XXX_sizecache        int32               `json:"-"`
}

func (m *SegmentMeta) Reset()         { *m = SegmentMeta{} }
func (m *SegmentMeta) String() string { return proto.CompactTextString(m) }
func (*SegmentMeta) ProtoMessage()    {}
func (*SegmentMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{9}
}

func (m *SegmentMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_SegmentMeta.Unmarshal(m, b)
}
func (m *SegmentMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_SegmentMeta.Marshal(b, m, deterministic)
}
func (m *SegmentMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SegmentMeta.Merge(m, src)
}
func (m *SegmentMeta) XXX_Size() int {
	return xxx_messageInfo_SegmentMeta.Size(m)
}
func (m *SegmentMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_SegmentMeta.DiscardUnknown(m)
}

var xxx_messageInfo_SegmentMeta proto.InternalMessageInfo

func (m *SegmentMeta) GetSegmentID() int64 {
	if m != nil {
		return m.SegmentID
	}
	return 0
}

func (m *SegmentMeta) GetCollectionID() int64 {
	if m != nil {
		return m.CollectionID
	}
	return 0
}

func (m *SegmentMeta) GetPartitionTag() string {
	if m != nil {
		return m.PartitionTag
	}
	return ""
}

func (m *SegmentMeta) GetChannelStart() int32 {
	if m != nil {
		return m.ChannelStart
	}
	return 0
}

func (m *SegmentMeta) GetChannelEnd() int32 {
	if m != nil {
		return m.ChannelEnd
	}
	return 0
}

func (m *SegmentMeta) GetOpenTime() uint64 {
	if m != nil {
		return m.OpenTime
	}
	return 0
}

func (m *SegmentMeta) GetCloseTime() uint64 {
	if m != nil {
		return m.CloseTime
	}
	return 0
}

func (m *SegmentMeta) GetNumRows() int64 {
	if m != nil {
		return m.NumRows
	}
	return 0
}

func (m *SegmentMeta) GetMemSize() int64 {
	if m != nil {
		return m.MemSize
	}
	return 0
}

func (m *SegmentMeta) GetBinlogFilePaths() []*FieldBinlogFiles {
	if m != nil {
		return m.BinlogFilePaths
	}
	return nil
}

type FieldIndexMeta struct {
	SegmentID            int64                    `protobuf:"varint,1,opt,name=segmentID,proto3" json:"segmentID,omitempty"`
	FieldID              int64                    `protobuf:"varint,2,opt,name=fieldID,proto3" json:"fieldID,omitempty"`
	IndexID              int64                    `protobuf:"varint,3,opt,name=indexID,proto3" json:"indexID,omitempty"`
	IndexParams          []*commonpb.KeyValuePair `protobuf:"bytes,4,rep,name=index_params,json=indexParams,proto3" json:"index_params,omitempty"`
	State                commonpb.IndexState      `protobuf:"varint,5,opt,name=state,proto3,enum=milvus.proto.common.IndexState" json:"state,omitempty"`
	IndexFilePaths       []string                 `protobuf:"bytes,6,rep,name=index_file_paths,json=indexFilePaths,proto3" json:"index_file_paths,omitempty"`
	XXX_NoUnkeyedLiteral struct{}                 `json:"-"`
	XXX_unrecognized     []byte                   `json:"-"`
	XXX_sizecache        int32                    `json:"-"`
}

func (m *FieldIndexMeta) Reset()         { *m = FieldIndexMeta{} }
func (m *FieldIndexMeta) String() string { return proto.CompactTextString(m) }
func (*FieldIndexMeta) ProtoMessage()    {}
func (*FieldIndexMeta) Descriptor() ([]byte, []int) {
	return fileDescriptor_975d306d62b73e88, []int{10}
}

func (m *FieldIndexMeta) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_FieldIndexMeta.Unmarshal(m, b)
}
func (m *FieldIndexMeta) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_FieldIndexMeta.Marshal(b, m, deterministic)
}
func (m *FieldIndexMeta) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FieldIndexMeta.Merge(m, src)
}
func (m *FieldIndexMeta) XXX_Size() int {
	return xxx_messageInfo_FieldIndexMeta.Size(m)
}
func (m *FieldIndexMeta) XXX_DiscardUnknown() {
	xxx_messageInfo_FieldIndexMeta.DiscardUnknown(m)
}

var xxx_messageInfo_FieldIndexMeta proto.InternalMessageInfo

func (m *FieldIndexMeta) GetSegmentID() int64 {
	if m != nil {
		return m.SegmentID
	}
	return 0
}

func (m *FieldIndexMeta) GetFieldID() int64 {
	if m != nil {
		return m.FieldID
	}
	return 0
}

func (m *FieldIndexMeta) GetIndexID() int64 {
	if m != nil {
		return m.IndexID
	}
	return 0
}

func (m *FieldIndexMeta) GetIndexParams() []*commonpb.KeyValuePair {
	if m != nil {
		return m.IndexParams
	}
	return nil
}

func (m *FieldIndexMeta) GetState() commonpb.IndexState {
	if m != nil {
		return m.State
	}
	return commonpb.IndexState_NONE
}

func (m *FieldIndexMeta) GetIndexFilePaths() []string {
	if m != nil {
		return m.IndexFilePaths
	}
	return nil
}

func init() {
	proto.RegisterType((*TenantMeta)(nil), "milvus.proto.etcd.TenantMeta")
	proto.RegisterType((*ProxyMeta)(nil), "milvus.proto.etcd.ProxyMeta")
	proto.RegisterType((*PartitionInfo)(nil), "milvus.proto.etcd.PartitionInfo")
	proto.RegisterType((*IndexParamsInfo)(nil), "milvus.proto.etcd.IndexParamsInfo")
	proto.RegisterType((*CollectionInfo)(nil), "milvus.proto.etcd.CollectionInfo")
	proto.RegisterType((*IndexInfo)(nil), "milvus.proto.etcd.IndexInfo")
	proto.RegisterType((*SegmentIndexInfo)(nil), "milvus.proto.etcd.SegmentIndexInfo")
	proto.RegisterType((*CollectionMeta)(nil), "milvus.proto.etcd.CollectionMeta")
	proto.RegisterType((*FieldBinlogFiles)(nil), "milvus.proto.etcd.FieldBinlogFiles")
	proto.RegisterType((*SegmentMeta)(nil), "milvus.proto.etcd.SegmentMeta")
	proto.RegisterType((*FieldIndexMeta)(nil), "milvus.proto.etcd.FieldIndexMeta")
}

func init() { proto.RegisterFile("etcd_meta.proto", fileDescriptor_975d306d62b73e88) }

var fileDescriptor_975d306d62b73e88 = []byte{
	// 861 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xb4, 0x55, 0x5d, 0x8f, 0xdb, 0x44,
	0x14, 0x95, 0x93, 0x6c, 0x76, 0x7d, 0x93, 0x4d, 0x76, 0xfd, 0x64, 0x4a, 0x4b, 0x53, 0x57, 0x85,
	0x48, 0x88, 0x44, 0x5a, 0x04, 0x6f, 0x20, 0x68, 0xd3, 0x4a, 0x11, 0xa2, 0x0d, 0xce, 0x8a, 0x07,
	0x5e, 0xac, 0x89, 0x7d, 0x9b, 0x8c, 0xe4, 0x19, 0xa7, 0x9e, 0x31, 0xdd, 0xdd, 0x07, 0xc4, 0x2b,
	0x88, 0x5f, 0xc0, 0x5f, 0xe4, 0x37, 0x20, 0xa1, 0xf9, 0x88, 0x3f, 0x92, 0x08, 0xa1, 0x95, 0x78,
	0x9c, 0x73, 0xcf, 0xcc, 0xbd, 0xf7, 0x9c, 0x7b, 0x6d, 0x18, 0xa2, 0x8c, 0x93, 0x88, 0xa1, 0x24,
	0x93, 0x6d, 0x9e, 0xc9, 0xcc, 0xbb, 0x64, 0x34, 0xfd, 0xb9, 0x10, 0xe6, 0x34, 0x51, 0xd1, 0x07,
	0xfd, 0x38, 0x63, 0x2c, 0xe3, 0x06, 0x7a, 0xd0, 0x17, 0xf1, 0x06, 0x99, 0xa5, 0x07, 0x7f, 0x3a,
	0x00, 0xd7, 0xc8, 0x09, 0x97, 0xdf, 0xa3, 0x24, 0xde, 0x00, 0x5a, 0xf3, 0x99, 0xef, 0x8c, 0x9c,
	0x71, 0x3b, 0x6c, 0xcd, 0x67, 0xde, 0xc7, 0x30, 0xe4, 0x05, 0x8b, 0xde, 0x15, 0x98, 0xdf, 0x46,
	0x3c, 0x4b, 0x50, 0xf8, 0x2d, 0x1d, 0x3c, 0xe7, 0x05, 0xfb, 0x41, 0xa1, 0xaf, 0x15, 0xe8, 0x7d,
	0x0a, 0x97, 0x94, 0x0b, 0xcc, 0x65, 0x14, 0x6f, 0x08, 0xe7, 0x98, 0xce, 0x67, 0xc2, 0x6f, 0x8f,
	0xda, 0x63, 0x37, 0xbc, 0x30, 0x81, 0x17, 0x25, 0xee, 0x7d, 0x02, 0x43, 0xf3, 0x60, 0xc9, 0xf5,
	0x3b, 0x23, 0x67, 0xec, 0x86, 0x03, 0x0d, 0x97, 0xcc, 0xe0, 0x57, 0x07, 0xdc, 0x45, 0x9e, 0xdd,
	0xdc, 0x1e, 0xad, 0xed, 0x4b, 0x38, 0x25, 0x49, 0x92, 0xa3, 0x30, 0x35, 0xf5, 0xae, 0x1e, 0x4e,
	0x1a, 0xbd, 0xdb, 0xae, 0xbf, 0x35, 0x9c, 0x70, 0x47, 0x56, 0xb5, 0xe6, 0x28, 0x8a, 0xf4, 0x58,
	0xad, 0x26, 0x50, 0xd5, 0x1a, 0xdc, 0xc0, 0xf9, 0x82, 0xe4, 0x92, 0x4a, 0x9a, 0xf1, 0x39, 0x7f,
	0x9b, 0x79, 0xcf, 0x60, 0xb0, 0xdd, 0x01, 0x11, 0x27, 0x0c, 0x75, 0x45, 0x6e, 0x78, 0x5e, 0xa2,
	0xaf, 0x09, 0x43, 0x6f, 0x04, 0xbd, 0x12, 0x98, 0xcf, 0xac, 0x68, 0x75, 0xc8, 0xfb, 0x08, 0x40,
	0xe0, 0x9a, 0x21, 0x97, 0xbb, 0xfc, 0xed, 0xb0, 0x86, 0x04, 0xef, 0x60, 0x38, 0xe7, 0x09, 0xde,
	0x2c, 0x48, 0x4e, 0x98, 0xd0, 0xb9, 0x7d, 0x38, 0x7d, 0x4b, 0x53, 0x4c, 0x4a, 0x19, 0x76, 0x47,
	0x6f, 0x06, 0x7d, 0xaa, 0xc8, 0xd1, 0x56, 0xb3, 0xfd, 0xd6, 0xa8, 0x3d, 0xee, 0x5d, 0x3d, 0x39,
	0x2a, 0xc8, 0x77, 0x78, 0xfb, 0x23, 0x49, 0x0b, 0x5c, 0x10, 0x9a, 0x87, 0x3d, 0x5a, 0xe5, 0x08,
	0xfe, 0x68, 0xc1, 0xe0, 0x45, 0x96, 0xa6, 0x18, 0x97, 0xed, 0xee, 0x8b, 0xfe, 0x15, 0x74, 0xcd,
	0xfc, 0x58, 0xcd, 0x9f, 0x35, 0x53, 0xd8, 0xd9, 0xaa, 0x1e, 0x59, 0x6a, 0x20, 0xb4, 0x97, 0xbc,
	0xc7, 0xd0, 0x8b, 0x73, 0x24, 0x12, 0x23, 0x49, 0x19, 0xfa, 0xed, 0x91, 0x33, 0xee, 0x84, 0x60,
	0xa0, 0x6b, 0xca, 0xd0, 0x0b, 0xa0, 0x5f, 0x13, 0x49, 0xf8, 0x1d, 0xad, 0x4b, 0x03, 0xf3, 0x5e,
	0xee, 0x35, 0x7b, 0xa2, 0x9b, 0x0d, 0x26, 0x07, 0x93, 0x3f, 0xd9, 0x13, 0xb0, 0xd1, 0xad, 0xaa,
	0xc5, 0x3c, 0xa3, 0x5c, 0x14, 0x7e, 0x57, 0x4f, 0x00, 0x68, 0x48, 0x59, 0x28, 0x82, 0xdf, 0x1d,
	0x70, 0xf5, 0x0b, 0x5a, 0x89, 0x47, 0x00, 0x15, 0xdd, 0x9a, 0xee, 0x96, 0x6c, 0xe5, 0x8d, 0x3e,
	0x94, 0x66, 0xef, 0x8e, 0x07, 0xde, 0xb4, 0xef, 0xe5, 0xcd, 0x2f, 0x70, 0xb1, 0xb4, 0xc3, 0x51,
	0x96, 0xf4, 0x10, 0xdc, 0x72, 0x60, 0xac, 0x47, 0x15, 0x60, 0xa6, 0x05, 0xd3, 0xa4, 0xaa, 0xc8,
	0x1e, 0xeb, 0xb5, 0xb6, 0x9b, 0xb5, 0xfa, 0x70, 0xba, 0x2a, 0xa8, 0xbe, 0xd3, 0x31, 0x11, 0x7b,
	0x0c, 0xfe, 0x72, 0xea, 0xb3, 0x71, 0x74, 0x21, 0xff, 0xef, 0xd9, 0x68, 0x6e, 0x4c, 0x67, 0x7f,
	0x63, 0x9a, 0xab, 0x29, 0xc9, 0xda, 0x4c, 0x46, 0x7d, 0x35, 0xaf, 0xc9, 0x5a, 0x1c, 0x8c, 0x58,
	0xf7, 0x70, 0xc4, 0x82, 0x37, 0x70, 0xf1, 0x4a, 0x89, 0xf5, 0x9c, 0xf2, 0x34, 0x5b, 0xbf, 0xa2,
	0x29, 0x8a, 0xba, 0x9e, 0x4e, 0x53, 0xcf, 0x27, 0xd0, 0x5f, 0x69, 0x62, 0xa4, 0xf6, 0xd1, 0x6c,
	0x9f, 0x1b, 0xf6, 0x56, 0xd5, 0xe5, 0xe0, 0xef, 0x16, 0xf4, 0xac, 0x7f, 0x5a, 0xbb, 0x7f, 0xb7,
	0x2e, 0x80, 0x7e, 0x5c, 0xed, 0xe1, 0xce, 0xbf, 0x06, 0xe6, 0x3d, 0x85, 0xf3, 0x46, 0xb7, 0x5a,
	0x30, 0xb7, 0xd6, 0xc7, 0x35, 0x59, 0x2b, 0x92, 0xfd, 0xc8, 0x45, 0x42, 0x92, 0x5c, 0x6a, 0x57,
	0x4f, 0xc2, 0xbe, 0x05, 0x97, 0x0a, 0xd3, 0xc2, 0x5b, 0x12, 0xf2, 0xc4, 0x3f, 0xd1, 0x14, 0xb0,
	0xd0, 0x4b, 0x9e, 0x78, 0x1f, 0x82, 0x9b, 0x6d, 0x91, 0x1b, 0x5f, 0xba, 0xda, 0x97, 0x33, 0x05,
	0x68, 0x57, 0x1e, 0x01, 0xc4, 0x69, 0x26, 0xac, 0x6b, 0xa7, 0x3a, 0xea, 0x6a, 0x44, 0x87, 0x3f,
	0x80, 0x33, 0xf5, 0x07, 0xc9, 0xb3, 0xf7, 0xc2, 0x3f, 0x33, 0xb2, 0xf1, 0x82, 0x85, 0xd9, 0x7b,
	0xa1, 0x42, 0x0c, 0x59, 0x24, 0xe8, 0x1d, 0xfa, 0xae, 0x09, 0x31, 0x64, 0x4b, 0x7a, 0x87, 0xde,
	0x1b, 0xb8, 0xac, 0x29, 0x1a, 0x6d, 0x89, 0xdc, 0x08, 0x1f, 0xf4, 0xe2, 0x3c, 0x3d, 0xb2, 0xe7,
	0xfb, 0x5e, 0x85, 0xc3, 0x4a, 0xfb, 0x85, 0xba, 0x1b, 0xfc, 0xd6, 0x82, 0x81, 0x66, 0xe9, 0xed,
	0xf9, 0x0f, 0x16, 0xdc, 0x67, 0x7b, 0xf6, 0x37, 0xbd, 0x73, 0x9f, 0x4d, 0xf7, 0xbe, 0x80, 0x13,
	0x21, 0x89, 0x44, 0x6d, 0xc4, 0xe0, 0xea, 0xf1, 0xd1, 0xeb, 0xba, 0x8d, 0xa5, 0xa2, 0x85, 0x86,
	0xed, 0x8d, 0xe1, 0xc2, 0x24, 0xaf, 0x29, 0x66, 0xbe, 0x69, 0x03, 0x8d, 0x97, 0x5a, 0x3c, 0xff,
	0xe6, 0xa7, 0xaf, 0xd7, 0x54, 0x6e, 0x8a, 0x95, 0x7a, 0x6c, 0x7a, 0x47, 0xd3, 0x94, 0xde, 0x49,
	0x8c, 0x37, 0x53, 0x93, 0xe8, 0xb3, 0x84, 0x0a, 0x99, 0xd3, 0x55, 0x21, 0x31, 0x99, 0x52, 0x2e,
	0x31, 0xe7, 0x24, 0x9d, 0xea, 0xec, 0x53, 0xa5, 0xf6, 0x76, 0xb5, 0xea, 0xea, 0xd3, 0xe7, 0xff,
	0x04, 0x00, 0x00, 0xff, 0xff, 0x85, 0x45, 0xd7, 0x1c, 0x7e, 0x08, 0x00, 0x00,
}
