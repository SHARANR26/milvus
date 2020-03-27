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

#include "server/delivery/RequestHandler.h"

#include <set>

#include "server/delivery/RequestScheduler.h"
#include "server/delivery/request/BaseRequest.h"
#include "server/delivery/request/CmdRequest.h"
#include "server/delivery/request/CompactRequest.h"
#include "server/delivery/request/CountTableRequest.h"
#include "server/delivery/request/CreateIndexRequest.h"
#include "server/delivery/request/CreatePartitionRequest.h"
#include "server/delivery/request/CreateTableRequest.h"
#include "server/delivery/request/DeleteByIDRequest.h"
#include "server/delivery/request/DescribeIndexRequest.h"
#include "server/delivery/request/DescribeTableRequest.h"
#include "server/delivery/request/DropIndexRequest.h"
#include "server/delivery/request/DropPartitionRequest.h"
#include "server/delivery/request/DropTableRequest.h"
#include "server/delivery/request/FlushRequest.h"
#include "server/delivery/request/GetVectorByIDRequest.h"
#include "server/delivery/request/GetVectorIDsRequest.h"
#include "server/delivery/request/HasTableRequest.h"
#include "server/delivery/request/InsertRequest.h"
#include "server/delivery/request/PreloadTableRequest.h"
#include "server/delivery/request/SearchByIDRequest.h"
#include "server/delivery/request/SearchRequest.h"
#include "server/delivery/request/ShowPartitionsRequest.h"
#include "server/delivery/request/ShowTableInfoRequest.h"
#include "server/delivery/request/ShowTablesRequest.h"

#include "server/delivery/hybrid_request/CreateCollectionRequest.h"
#include "server/delivery/hybrid_request/InsertEntityRequest.h"
#include "server/delivery/hybrid_request/HybridSearchRequest.h"

namespace milvus {
namespace server {

Status
RequestHandler::CreateTable(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t dimension,
                            int64_t index_file_size, int64_t metric_type) {
    BaseRequestPtr request_ptr =
        CreateTableRequest::Create(context, table_name, dimension, index_file_size, metric_type);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::HasTable(const std::shared_ptr<Context>& context, const std::string& table_name, bool& has_table) {
    BaseRequestPtr request_ptr = HasTableRequest::Create(context, table_name, has_table);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::DropTable(const std::shared_ptr<Context>& context, const std::string& table_name) {
    BaseRequestPtr request_ptr = DropTableRequest::Create(context, table_name);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::CreateIndex(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t index_type,
                            const milvus::json& json_params) {
    BaseRequestPtr request_ptr = CreateIndexRequest::Create(context, table_name, index_type, json_params);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::Insert(const std::shared_ptr<Context>& context, const std::string& table_name,
                       engine::VectorsData& vectors, const std::string& partition_tag) {
    BaseRequestPtr request_ptr = InsertRequest::Create(context, table_name, vectors, partition_tag);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::GetVectorByID(const std::shared_ptr<Context>& context, const std::string& table_name,
                              const std::vector<int64_t>& ids, engine::VectorsData& vectors) {
    BaseRequestPtr request_ptr = GetVectorByIDRequest::Create(context, table_name, ids, vectors);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::GetVectorIDs(const std::shared_ptr<Context>& context, const std::string& table_name,
                             const std::string& segment_name, std::vector<int64_t>& vector_ids) {
    BaseRequestPtr request_ptr = GetVectorIDsRequest::Create(context, table_name, segment_name, vector_ids);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::ShowTables(const std::shared_ptr<Context>& context, std::vector<std::string>& tables) {
    BaseRequestPtr request_ptr = ShowTablesRequest::Create(context, tables);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::ShowTableInfo(const std::shared_ptr<Context>& context, const std::string& table_name,
                              TableInfo& table_info) {
    BaseRequestPtr request_ptr = ShowTableInfoRequest::Create(context, table_name, table_info);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::Search(const std::shared_ptr<Context>& context, const std::string& table_name,
                       const engine::VectorsData& vectors, int64_t topk, const milvus::json& extra_params,
                       const std::vector<std::string>& partition_list, const std::vector<std::string>& file_id_list,
                       TopKQueryResult& result) {
    BaseRequestPtr request_ptr =
        SearchRequest::Create(context, table_name, vectors, topk, extra_params, partition_list, file_id_list, result);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::SearchByID(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t vector_id,
                           int64_t topk, const milvus::json& extra_params,
                           const std::vector<std::string>& partition_list, TopKQueryResult& result) {
    BaseRequestPtr request_ptr =
        SearchByIDRequest::Create(context, table_name, vector_id, topk, extra_params, partition_list, result);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::DescribeTable(const std::shared_ptr<Context>& context, const std::string& table_name,
                              TableSchema& table_schema) {
    BaseRequestPtr request_ptr = DescribeTableRequest::Create(context, table_name, table_schema);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::CountTable(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t& count) {
    BaseRequestPtr request_ptr = CountTableRequest::Create(context, table_name, count);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::Cmd(const std::shared_ptr<Context>& context, const std::string& cmd, std::string& reply) {
    BaseRequestPtr request_ptr = CmdRequest::Create(context, cmd, reply);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::DeleteByID(const std::shared_ptr<Context>& context, const std::string& table_name,
                           const std::vector<int64_t>& vector_ids) {
    BaseRequestPtr request_ptr = DeleteByIDRequest::Create(context, table_name, vector_ids);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::PreloadTable(const std::shared_ptr<Context>& context, const std::string& table_name) {
    BaseRequestPtr request_ptr = PreloadTableRequest::Create(context, table_name);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::DescribeIndex(const std::shared_ptr<Context>& context, const std::string& table_name,
                              IndexParam& param) {
    BaseRequestPtr request_ptr = DescribeIndexRequest::Create(context, table_name, param);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::DropIndex(const std::shared_ptr<Context>& context, const std::string& table_name) {
    BaseRequestPtr request_ptr = DropIndexRequest::Create(context, table_name);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::CreatePartition(const std::shared_ptr<Context>& context, const std::string& table_name,
                                const std::string& tag) {
    BaseRequestPtr request_ptr = CreatePartitionRequest::Create(context, table_name, tag);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::ShowPartitions(const std::shared_ptr<Context>& context, const std::string& table_name,
                               std::vector<PartitionParam>& partitions) {
    BaseRequestPtr request_ptr = ShowPartitionsRequest::Create(context, table_name, partitions);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::DropPartition(const std::shared_ptr<Context>& context, const std::string& table_name,
                              const std::string& tag) {
    BaseRequestPtr request_ptr = DropPartitionRequest::Create(context, table_name, tag);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::Flush(const std::shared_ptr<Context>& context, const std::vector<std::string>& table_names) {
    BaseRequestPtr request_ptr = FlushRequest::Create(context, table_names);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

Status
RequestHandler::Compact(const std::shared_ptr<Context>& context, const std::string& table_name) {
    BaseRequestPtr request_ptr = CompactRequest::Create(context, table_name);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

/*******************************************New Interface*********************************************/

Status
RequestHandler::CreateCollection(const std::shared_ptr<Context>& context,
                 const std::string& collection_name,
                 std::vector<std::pair<std::string, engine::meta::hybrid::DataType>>& field_types,
                 std::vector<std::pair<std::string, uint64_t>>& vector_dimensions,
                 std::vector<std::pair<std::string, std::string>>& field_extra_params) {
    BaseRequestPtr request_ptr =
        CreateCollectionRequest::Create(context, collection_name, field_types, vector_dimensions, field_extra_params);

    RequestScheduler::ExecRequest(request_ptr);
    return request_ptr->status();
}

Status
RequestHandler::HasCollection(const std::shared_ptr<Context>& context, std::string& collection_name, bool& has_collection) {

}

Status
RequestHandler::InsertEntity(const std::shared_ptr<Context>& context,
                             const std::string& collection_name,
                             const std::string& partition_tag,
                             std::vector<std::string>& field_name_array,
                             std::vector<std::vector<std::string>>& field_values,
                             std::vector<engine::VectorsData>& vector_data) {
    BaseRequestPtr request_ptr =
        InsertEntityRequest::Create(context,
                                    collection_name,
                                    partition_tag,
                                    field_name_array,
                                    field_values,
                                    vector_data);

    RequestScheduler::ExecRequest(request_ptr);
    return request_ptr->status();
}

Status
RequestHandler::HybridSearch(const std::shared_ptr<Context>& context,
                                    const std::string& collection_name,
                                    std::vector<std::string>& partition_list,
                                    milvus::query::GeneralQueryPtr& general_query,
                                    milvus::server::HybridQueryResult& result) {
    BaseRequestPtr request_ptr =
        HybridSearchRequest::Create(context, collection_name, partition_list, general_query, result);
    RequestScheduler::ExecRequest(request_ptr);

    return request_ptr->status();
}

}  // namespace server
}  // namespace milvus
