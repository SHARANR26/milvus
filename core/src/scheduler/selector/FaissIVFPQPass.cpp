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
#ifdef MILVUS_GPU_VERSION
#include "scheduler/selector/FaissIVFPQPass.h"
#include "cache/GpuCacheMgr.h"
#include "config/ServerConfig.h"
#include "faiss/gpu/utils/DeviceUtils.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "scheduler/SchedInst.h"
#include "scheduler/Utils.h"
#include "scheduler/task/SearchTask.h"
#include "scheduler/tasklabel/SpecResLabel.h"
#include "utils/Log.h"

#include <fiu/fiu-local.h>

namespace milvus {
namespace scheduler {

FaissIVFPQPass::FaissIVFPQPass() {
    ConfigMgr::GetInstance().Attach("gpu.gpu_search_threshold", this);
}

FaissIVFPQPass::~FaissIVFPQPass() {
    ConfigMgr::GetInstance().Detach("gpu.gpu_search_threshold", this);
}

void
FaissIVFPQPass::Init() {
#ifdef MILVUS_GPU_VERSION
<<<<<<< HEAD
    gpu_enable_ = config.gpu.enable();
    threshold_ = config.gpu.gpu_search_threshold();
    search_gpus_ = ParseGPUDevices(config.gpu.search_devices());
=======
    server::Config& config = server::Config::GetInstance();
    Status s = config.GetGpuResourceConfigGpuSearchThreshold(threshold_);
    if (!s.ok()) {
        threshold_ = std::numeric_limits<int32_t>::max();
    }
    s = config.GetGpuResourceConfigSearchResources(search_gpus_);
    if (!s.ok()) {
        throw std::exception();
    }

    SetIdentity("FaissIVFPQPass");
    AddGpuEnableListener();
    AddGpuSearchThresholdListener();
    AddGpuSearchResourcesListener();
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
#endif
}

bool
FaissIVFPQPass::Run(const TaskPtr& task) {
    if (task->Type() != TaskType::SearchTask) {
        return false;
    }

    auto search_task = std::static_pointer_cast<SearchTask>(task);
    if (search_task->IndexType() != knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
        return false;
    }

    ResourcePtr res_ptr;
    if (!gpu_enable_) {
        LOG_SERVER_DEBUG_ << LogOut("FaissIVFPQPass: gpu disable, specify cpu to search!");
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
    } else if (search_task->nq() < threshold_) {
        LOG_SERVER_DEBUG_ << LogOut("FaissIVFPQPass: nq < gpu_search_threshold, specify cpu to search!");
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
<<<<<<< HEAD
    } else if (search_task->ExtraParam()[knowhere::IndexParams::nprobe].get<int64_t>() >
               faiss::gpu::getMaxKSelection()) {
        LOG_SERVER_DEBUG_ << LogOut("FaissIVFFlatPass: nprobe > gpu_max_nprobe_threshold, specify cpu to search!");
=======
    } else if (search_job->nq() < (uint64_t)threshold_) {
        LOG_SERVER_DEBUG_ << LogOut("[%s][%d] FaissIVFPQPass: nq < gpu_search_threshold, specify cpu to search!",
                                    "search", 0);
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
    } else {
        LOG_SERVER_DEBUG_ << LogOut("FaissIVFPQPass: nq >= gpu_search_threshold, specify gpu %d to search!",
                                    search_gpus_[idx_]);
        res_ptr = ResMgrInst::GetInstance()->GetResource(ResourceType::GPU, search_gpus_[idx_]);
        idx_ = (idx_ + 1) % search_gpus_.size();
    }
    auto label = std::make_shared<SpecResLabel>(res_ptr);
    task->label() = label;
    return true;
}

void
FaissIVFPQPass::ConfigUpdate(const std::string& name) {
    threshold_ = config.gpu.gpu_search_threshold();
    search_gpus_ = ParseGPUDevices(config.gpu.search_devices());
}

}  // namespace scheduler
}  // namespace milvus
#endif
