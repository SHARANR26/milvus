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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "unittest/utils.h"

int
main(int argc, char** argv) {
    std::cout << "Start index ut" << std::endl;
    signal(SIGILL, handle_signal);
    signal(SIGSEGV, handle_signal);
    signal(SIGABRT, handle_signal);
    signal(SIGFPE, handle_signal);
    signal(SIGTERM, handle_signal);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
