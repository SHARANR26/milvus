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

#include "CacheMgr.h"
#include "DataObj.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "config/handler/GpuResourceConfigHandler.h"

namespace milvus {
namespace cache {

#ifdef MILVUS_GPU_VERSION
class GpuCacheMgr;
using GpuCacheMgrPtr = std::shared_ptr<GpuCacheMgr>;

class GpuCacheMgr : public CacheMgr<DataObjPtr>, public server::GpuResourceConfigHandler {
 public:
    GpuCacheMgr();

    ~GpuCacheMgr();

    static GpuCacheMgr*
    GetInstance(uint64_t gpu_id);

    DataObjPtr
    GetIndex(const std::string& key);

    void
    InsertItem(const std::string& key, const DataObjPtr& data);

 protected:
    void
    OnGpuCacheCapacityChanged(int64_t capacity) override;

 private:
    bool gpu_enable_ = true;
    std::string identity_;
    static std::mutex mutex_;
    static std::unordered_map<uint64_t, GpuCacheMgrPtr> instance_;
};
#endif

}  // namespace cache
}  // namespace milvus
