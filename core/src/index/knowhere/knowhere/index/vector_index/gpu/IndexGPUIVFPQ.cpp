// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <string>

#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/index_factory.h>

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexIVFPQ.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/gpu/IndexGPUIVFPQ.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

namespace milvus {
namespace knowhere {

void
GPUIVFPQ::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)
    gpu_id_ = config[knowhere::meta::DEVICEID];

    auto gpu_res = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_);
    if (gpu_res != nullptr) {
        ResScope rs(gpu_res, gpu_id_, true);
        auto device_index =
            new faiss::gpu::GpuIndexIVFPQ(gpu_res->faiss_res.get(), dim, config[IndexParams::nlist].get<int64_t>(),
                                          config[IndexParams::m], config[IndexParams::nbits],
                                          GetMetricType(config[Metric::TYPE].get<std::string>()));  // IP not support
<<<<<<< HEAD
        device_index->train(rows, reinterpret_cast<const float*>(p_data));
=======
        device_index->train(rows, (float*)p_data);
>>>>>>> af8ea3cc1f1816f42e94a395ab9286dfceb9ceda

        index_.reset(device_index);
        res_ = gpu_res;
    } else {
        KNOWHERE_THROW_MSG("Build IVFPQ can't get gpu resource");
    }
}

VecIndexPtr
GPUIVFPQ::CopyGpuToCpu(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);

    faiss::Index* device_index = index_.get();
    faiss::Index* host_index = faiss::gpu::index_gpu_to_cpu(device_index);

    std::shared_ptr<faiss::Index> new_index;
    new_index.reset(host_index);
    return std::make_shared<IVFPQ>(new_index);
}

std::shared_ptr<faiss::IVFSearchParameters>
GPUIVFPQ::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFPQSearchParameters>();
    params->nprobe = config[IndexParams::nprobe];
    // params->scan_table_threshold = config["scan_table_threhold"]
    // params->polysemous_ht = config["polysemous_ht"]
    // params->max_codes = config["max_codes"]

    return params;
}

}  // namespace knowhere
}  // namespace milvus
