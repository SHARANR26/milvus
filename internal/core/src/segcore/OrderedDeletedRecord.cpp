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

#include "segcore/OrderedDeletedRecord.h"

namespace milvus::segcore {

void
OrderedDeletedRecord::generate_unsafe() {
    AssertInfo(!switched_, "generate deleted record more than once");

    auto size = buffer_.size();
    std::vector<std::tuple<Timestamp, PkType>> ordering;
    ordering.reserve(size);
    for (const auto& kv : buffer_) {
        ordering.push_back(std::make_tuple(kv.second, kv.first));
    }
    std::sort(ordering.begin(), ordering.end());
    std::vector<PkType> sort_pks(size);
    std::vector<Timestamp> sort_timestamps(size);
    for (int i = 0; i < size; i++) {
        auto [t, pk] = ordering[i];
        sort_timestamps[i] = t;
        sort_pks[i] = pk;
    }

    final_record_.push(sort_pks, sort_timestamps.data());
    buffer_.clear();
    switched_ = true;
}

DeletedRecord&
OrderedDeletedRecord::get_or_generate() {
    {
        std::shared_lock<std::shared_mutex> lck(mtx_);
        if (switched_) {
            return final_record_;
        }
    }

    std::unique_lock<std::shared_mutex> lck(mtx_);
    if (switched_) {
        // already generated by other thread.
        return final_record_;
    }

    generate_unsafe();

    return final_record_;
}

int64_t
OrderedDeletedRecord::get_deleted_count() const {
    std::shared_lock<std::shared_mutex> lck(mtx_);
    if (switched_) {
        return final_record_.size();
    }
    return buffer_.size();
}

void
OrderedDeletedRecord::load(const std::vector<PkType>& pks,
                           const Timestamp* timestamps) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    AssertInfo(!switched_,
               "[should not happen] load delete record after search");
    load_unsafe(pks, timestamps);
}

void
OrderedDeletedRecord::push(const std::vector<PkType>& pks,
                           const Timestamp* timestamps) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    if (!switched_) {
        load_unsafe(pks, timestamps);
    } else {
        push_unsafe(pks, timestamps);
    }
}

void
OrderedDeletedRecord::load_unsafe(const std::vector<PkType>& pks,
                                  const Timestamp* timestamps) {
    // TODO: what if len(pks) != len(timestamps)?
    auto len = pks.size();
    for (size_t i = 0; i < len; i++) {
        auto pk = pks[i];
        auto timestamp = timestamps[i];

        auto iter = buffer_.find(pk);
        if (iter == buffer_.end()) {
            buffer_[pk] = timestamp;
        } else {
            // we need the newest record anyway.
            iter->second = timestamp > iter->second ? timestamp : iter->second;
        }
    }
}

void
OrderedDeletedRecord::push_unsafe(const std::vector<PkType>& pks,
                                  const Timestamp* timestamps) {
    final_record_.push(pks, timestamps);
}

}  // namespace milvus::segcore
