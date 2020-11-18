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

#include <prometheus/registry.h>

namespace milvus {

class Prometheus {
 public:
    prometheus::Registry&
    registry() {
        return *registry_;
    }

 private:
    std::shared_ptr<prometheus::Registry> registry_ = std::make_shared<prometheus::Registry>();
};

extern Prometheus prometheus;

}  // namespace milvus
