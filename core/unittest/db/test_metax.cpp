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

#include <iostream>
#include <memory>

#include "db/metax/MetaAdapter.h"
#include "db/metax/MetaFactory.h"
#include "db/metax/MetaProxy.h"
#include "db/metax/MetaResField.h"

#include "db/snapshot/Resources.h"

// unittest folder
#include "db/utils.h"

template <typename F>
using MetaResField = milvus::engine::metax::MetaResField<F>;

template <typename Base, typename Equal, typename Derived>
using is_decay_base_of_and_equal_of = milvus::engine::metax::is_decay_base_of_and_equal_of<Base, Equal, Derived>;

TEST_F(MetaxTest, HelperTest) {
    auto proxy = std::make_shared<milvus::engine::metax::MetaProxy>();

    auto adapter = std::make_shared<milvus::engine::metax::MetaAdapter>(proxy);

    auto collection = std::make_shared<Collection>("metax_test_c1");
    auto status = adapter->Insert(collection);
    ASSERT_TRUE(status.ok()) << status.ToString();
}

TEST_F(MetaxTest, TraitsTest) {
    auto ff = MetaResField<IdField>();

    std::cout << is_decay_base_of_and_equal_of<MetaResField<IdField>::FType, IdField, Collection>::value << std::endl;
    std::cout << is_decay_base_of_and_equal_of<MetaResField<MappingsField>::FType, MappingsField, Collection>::value << std::endl;
//    auto collection = std::make_shared<Collection>("a");
//    std::string b = bool(milvus::engine::meta::is_shared_ptr<decltype(collection)>::value) ? "True" : "False";
//    std::cout << "std::make_shared<Collection>(\"a\") is shared_ptr: "
//              << b << std::endl;
}
