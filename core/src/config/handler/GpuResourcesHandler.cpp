
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
#include "config/handler/GpuResourcesHandler.h"

namespace milvus {
namespace server {

GpuResourcesHandler::GpuResourcesHandler() {
    server::Config& config = server::Config::GetInstance();
    config.GetGpuResourceConfigEnable(gpu_enable_);
}

GpuResourcesHandler::~GpuResourcesHandler() {
    RemoveGpuEnableListener();
}

//////////////////////////////////////////////////////////////
void
GpuResourcesHandler::OnGpuEnableChanged(bool enable) {
    gpu_enable_ = enable;
}

void
GpuResourcesHandler::AddGpuEnableListener() {
    SERVER_LOG_WARNING << "[" + identity_ + "] AddGpuEnableListener";
    auto& config = server::Config::GetInstance();

    server::ConfigCallBackF lambda = [this](const std::string& value) -> Status {
        SERVER_LOG_WARNING << "[" + identity_ + "] Call GpuEnableListener";
        auto& config = server::Config::GetInstance();
        bool enable;
        auto status = config.GetGpuResourceConfigEnable(enable);
        if (status.ok()) {
            OnGpuEnableChanged(enable);
        }

        return status;
    };
    config.RegisterCallBack(server::CONFIG_GPU_RESOURCE, server::CONFIG_GPU_RESOURCE_ENABLE, identity_, lambda);
}

void
GpuResourcesHandler::RemoveGpuEnableListener() {
    SERVER_LOG_WARNING << "[" + identity_ + "] RemoveGpuEnableListener";
    server::Config& config = server::Config::GetInstance();
    config.CancelCallBack(server::CONFIG_GPU_RESOURCE, server::CONFIG_GPU_RESOURCE_ENABLE, identity_);
}

}  // namespace server
}  // namespace milvus
#endif
