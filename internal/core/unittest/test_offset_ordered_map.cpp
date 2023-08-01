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

#include <gtest/gtest.h>
#include <random>
#include "segcore/InsertRecord.h"

using namespace milvus;
using namespace milvus::segcore;

template <typename T>
class TypedOffsetOrderedMapTest : public testing::Test {
 public:
    void
    SetUp() override {
        er = std::default_random_engine(42);
    }

    void
    TearDown() override {
    }

 protected:
    void
    insert(T pk) {
        map_.insert(pk, offset_++);
        data_.push_back(pk);
        std::sort(data_.begin(), data_.end());
    }

    std::vector<T>
    random_generate(int num) {
        std::vector<T> res;
        for (int i = 0; i < num; i++) {
            if constexpr (std::is_same_v<std::string, T>) {
                res.push_back(std::to_string(er()));
            } else {
                res.push_back(static_cast<T>(er()));
            }
        }
        return res;
    }

 protected:
    int64_t offset_ = 0;
    std::vector<T> data_;
    milvus::segcore::OffsetOrderedMap<T> map_;
    std::default_random_engine er;
};

using TypeOfPks = testing::Types<int64_t, std::string>;
TYPED_TEST_CASE_P(TypedOffsetOrderedMapTest);

TYPED_TEST_P(TypedOffsetOrderedMapTest, find_first) {
    std::vector<int64_t> offsets;

    // no data.
    offsets = this->map_.find_first(Unlimited, {});
    ASSERT_EQ(0, offsets.size());

    // insert 10 entities.
    int num = 10;
    auto data = this->random_generate(num);
    for (const auto& x : data) {
        this->insert(x);
    }

    // all is satisfied.
    BitsetType all(num);
    all.set();
    offsets = this->map_.find_first(num / 2, all);
    ASSERT_EQ(num / 2, offsets.size());
    for (int i = 1; i < offsets.size(); i++) {
        ASSERT_TRUE(data[offsets[i - 1]] <= data[offsets[i]]);
    }
    offsets = this->map_.find_first(Unlimited, all);
    ASSERT_EQ(num, offsets.size());
    for (int i = 1; i < offsets.size(); i++) {
        ASSERT_TRUE(data[offsets[i - 1]] <= data[offsets[i]]);
    }

    // none is satisfied.
    BitsetType none(num);
    none.reset();
    offsets = this->map_.find_first(num / 2, none);
    ASSERT_EQ(0, offsets.size());
    offsets = this->map_.find_first(NoLimit, none);
    ASSERT_EQ(0, offsets.size());
}

REGISTER_TYPED_TEST_CASE_P(TypedOffsetOrderedMapTest, find_first);
INSTANTIATE_TYPED_TEST_CASE_P(Prefix, TypedOffsetOrderedMapTest, TypeOfPks);
