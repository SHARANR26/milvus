// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "server/delivery/request/FlushRequest.h"

#include <memory>
#include <unordered_map>

#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

namespace milvus {
namespace server {

FlushRequest::FlushRequest(const std::shared_ptr<milvus::server::Context>& context,
                           const std::vector<std::string>& collection_names)
    : BaseRequest(context, BaseRequest::kFlush), collection_names_(collection_names) {
}

BaseRequestPtr
FlushRequest::Create(const std::shared_ptr<milvus::server::Context>& context,
                     const std::vector<std::string>& collection_names) {
    return std::shared_ptr<BaseRequest>(new FlushRequest(context, collection_names));
}

Status
FlushRequest::OnExecute() {
    std::string hdr = "FlushRequest flush collections: ";
    for (auto& name : collection_names_) {
        hdr += name;
        hdr += ", ";
    }

    TimeRecorderAuto rc(hdr);
    Status status = Status::OK();
    LOG_SERVER_DEBUG_ << hdr;

    // flush all collections
    if (collection_names_.empty()) {
        return DBWrapper::DB()->Flush();
    }

    // flush specified collections
    for (auto& name : collection_names_) {
        bool exist = false;
        auto status = DBWrapper::DB()->HasCollection(name, exist);
        if (!exist) {
            return Status(SERVER_COLLECTION_NOT_EXIST, CollectionNotExistMsg(name));
        }

        STATUS_CHECK(DBWrapper::DB()->Flush(name));
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
