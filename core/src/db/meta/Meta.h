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

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "MetaTypes.h"
#include "db/Options.h"
#include "db/Types.h"
#include "utils/Status.h"

namespace milvus {
namespace engine {
namespace meta {

static const char* META_ENVIRONMENT = "Environment";
static const char* META_TABLES = "Tables";
static const char* META_TABLEFILES = "TableFiles";

class Meta {
    /*
 public:
    class CleanUpFilter {
     public:
        virtual bool
        IsIgnored(const SegmentSchema& schema) = 0;
    };
*/

 public:
    virtual ~Meta() = default;

    virtual Status
    CreateCollection(CollectionSchema& table_schema) = 0;

    virtual Status
    DescribeCollection(CollectionSchema& table_schema) = 0;

    virtual Status
    HasCollection(const std::string& collection_id, bool& has_or_not) = 0;

    virtual Status
    AllCollections(std::vector<CollectionSchema>& table_schema_array) = 0;

    virtual Status
    UpdateCollectionFlag(const std::string& collection_id, int64_t flag) = 0;

    virtual Status
    UpdateTableFlushLSN(const std::string& collection_id, uint64_t flush_lsn) = 0;

    virtual Status
    GetTableFlushLSN(const std::string& collection_id, uint64_t& flush_lsn) = 0;

    virtual Status
    GetTableFilesByFlushLSN(uint64_t flush_lsn, SegmentsSchema& table_files) = 0;

    virtual Status
    DropCollection(const std::string& collection_id) = 0;

    virtual Status
    DeleteTableFiles(const std::string& collection_id) = 0;

    virtual Status
    CreateCollectionFile(SegmentSchema& file_schema) = 0;

    virtual Status
    GetTableFiles(const std::string& collection_id, const std::vector<size_t>& ids, SegmentsSchema& table_files) = 0;

    virtual Status
    GetTableFilesBySegmentId(const std::string& segment_id, SegmentsSchema& table_files) = 0;

    virtual Status
    UpdateTableFile(SegmentSchema& file_schema) = 0;

    virtual Status
    UpdateTableFiles(SegmentsSchema& files) = 0;

    virtual Status
    UpdateTableFilesRowCount(SegmentsSchema& files) = 0;

    virtual Status
    UpdateTableIndex(const std::string& collection_id, const TableIndex& index) = 0;

    virtual Status
    UpdateTableFilesToIndex(const std::string& collection_id) = 0;

    virtual Status
    DescribeCollectionIndex(const std::string& collection_id, TableIndex& index) = 0;

    virtual Status
    DropCollectionIndex(const std::string& collection_id) = 0;

    virtual Status
    CreatePartition(const std::string& collection_name, const std::string& partition_name, const std::string& tag,
                    uint64_t lsn) = 0;

    virtual Status
    DropPartition(const std::string& partition_name) = 0;

    virtual Status
    ShowPartitions(const std::string& collection_name, std::vector<meta::CollectionSchema>& partition_schema_array) = 0;

    virtual Status
    GetPartitionName(const std::string& collection_name, const std::string& tag, std::string& partition_name) = 0;

    virtual Status
    FilesToSearch(const std::string& collection_id, SegmentsSchema& files) = 0;

    virtual Status
    FilesToMerge(const std::string& collection_id, SegmentsSchema& files) = 0;

    virtual Status
    FilesToIndex(SegmentsSchema&) = 0;

    virtual Status
    FilesByType(const std::string& collection_id, const std::vector<int>& file_types, SegmentsSchema& files) = 0;

    virtual Status
    FilesByID(const std::vector<size_t>& ids, SegmentsSchema& files) = 0;

    virtual Status
    Size(uint64_t& result) = 0;

    virtual Status
    Archive() = 0;

    virtual Status
    CleanUpShadowFiles() = 0;

    virtual Status
    CleanUpFilesWithTTL(uint64_t seconds /*, CleanUpFilter* filter = nullptr*/) = 0;

    virtual Status
    DropAll() = 0;

    virtual Status
    Count(const std::string& collection_id, uint64_t& result) = 0;

    virtual Status
    SetGlobalLastLSN(uint64_t lsn) = 0;

    virtual Status
    GetGlobalLastLSN(uint64_t& lsn) = 0;
};  // MetaData

using MetaPtr = std::shared_ptr<Meta>;

}  // namespace meta
}  // namespace engine
}  // namespace milvus
