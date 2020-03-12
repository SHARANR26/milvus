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

#pragma once

#include <memory>

#include "cache/DataObj.h"
#include "knowhere/common/BinarySet.h"
#include "knowhere/common/Config.h"

namespace knowhere {

class Index : public milvus::cache::DataObj {
 public:
    virtual BinarySet
    Serialize(const Config& config = Config()) = 0;

    virtual void
    Load(const BinarySet&) = 0;

    int64_t
    Size() override {
        return size_;
    }

    void
    set_size(const int64_t& size) {
        size_ = size;
    }

 protected:
    int64_t size_ = -1;
};

using IndexPtr = std::shared_ptr<Index>;

// todo: remove from knowhere
class ToIndexData : public milvus::cache::DataObj {
 public:
    explicit ToIndexData(int64_t size) : size_(size) {
    }

    int64_t
    Size() override {
        return size_;
    }

 private:
    int64_t size_ = 0;
};

}  // namespace knowhere
