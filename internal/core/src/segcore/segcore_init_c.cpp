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

#include "config/ConfigKnowhere.h"
#include "log/Log.h"
#include "segcore/SegcoreConfig.h"
#include "segcore/segcore_init_c.h"

namespace milvus::segcore {
extern "C" void
SegcoreInit() {
    milvus::config::KnowhereInitImpl();
#if defined(EMBEDDED_MILVUS)
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    // Disable all logs for embedded milvus.
    defaultConf.set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Warning, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Error, el::ConfigurationType::Enabled, "false");
    defaultConf.set(el::Level::Fatal, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureLogger("default", defaultConf);
#endif
}

// TODO merge small index config into one config map, including enable/disable small_index
extern "C" void
SegcoreSetChunkRows(const int64_t value) {
    milvus::segcore::SegcoreConfig& config = milvus::segcore::SegcoreConfig::default_config();
    config.set_chunk_rows(value);
}

extern "C" void
SegcoreSetNlist(const int64_t value) {
    milvus::segcore::SegcoreConfig& config = milvus::segcore::SegcoreConfig::default_config();
    config.set_nlist(value);
}

extern "C" void
SegcoreSetNprobe(const int64_t value) {
    milvus::segcore::SegcoreConfig& config = milvus::segcore::SegcoreConfig::default_config();
    config.set_nprobe(value);
}

// return value must be freed by the caller
extern "C" char*
SegcoreSetSimdType(const char* value) {
    LOG_SEGCORE_DEBUG_ << "set config simd_type: " << value;
    auto real_type = milvus::config::KnowhereSetSimdType(value);
    char* ret = reinterpret_cast<char*>(malloc(real_type.length() + 1));
    memcpy(ret, real_type.c_str(), real_type.length());
    ret[real_type.length()] = 0;
    return ret;
}

extern "C" void
SegcoreSetIndexSliceSize(const int64_t value) {
    milvus::config::KnowhereSetIndexSliceSize(value);
    LOG_SEGCORE_DEBUG_ << "set config index slice size: " << value;
}

}  // namespace milvus::segcore
