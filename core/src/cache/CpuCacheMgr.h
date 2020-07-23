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

#include <memory>
#include <string>

#include "cache/CacheMgr.h"
#include "cache/DataObj.h"
#include "config/ConfigMgr.h"

namespace milvus {
namespace cache {

class CpuCacheMgr : public CacheMgr<DataObjPtr>, public ConfigObserver {
 private:
    CpuCacheMgr();

    ~CpuCacheMgr();

 public:
    // TODO(myh): use smart pointer instead
    static CpuCacheMgr*
    GetInstance();

    DataObjPtr
    GetIndex(const std::string& key);

 public:
    void
    ConfigUpdate(const std::string& name) override;
};

}  // namespace cache
}  // namespace milvus
