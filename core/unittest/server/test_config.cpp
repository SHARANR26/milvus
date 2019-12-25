// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>
#include <cmath>
#include "config/YamlConfigMgr.h"
#include "server/Config.h"
#include "server/utils.h"
#include "utils/CommonUtil.h"
#include "utils/StringHelpFunctions.h"
#include "utils/ValidationUtil.h"

#include <limits>

namespace {

static constexpr uint64_t KB = 1024;
static constexpr uint64_t MB = KB * 1024;
static constexpr uint64_t GB = MB * 1024;

}  // namespace

namespace ms = milvus::server;

TEST_F(ConfigTest, CONFIG_TEST) {
    milvus::server::ConfigMgr* config_mgr = milvus::server::YamlConfigMgr::GetInstance();

    milvus::Status s = config_mgr->LoadConfigFile("");
    ASSERT_FALSE(s.ok());

    std::string config_path(CONFIG_PATH);
    s = config_mgr->LoadConfigFile(config_path + INVALID_CONFIG_FILE);
    ASSERT_FALSE(s.ok());

    s = config_mgr->LoadConfigFile(config_path + VALID_CONFIG_FILE);
    ASSERT_TRUE(s.ok());

    config_mgr->Print();
    config_mgr->DumpString();

    milvus::server::ConfigNode& root_config = config_mgr->GetRootNode();
    milvus::server::ConfigNode& server_config = root_config.GetChild("server_config");
    milvus::server::ConfigNode& db_config = root_config.GetChild("db_config");
    milvus::server::ConfigNode& metric_config = root_config.GetChild("metric_config");
    milvus::server::ConfigNode& cache_config = root_config.GetChild("cache_config");
    milvus::server::ConfigNode invalid_config = root_config.GetChild("invalid_config");

    const auto& im_config_mgr = *static_cast<milvus::server::YamlConfigMgr*>(config_mgr);
    const milvus::server::ConfigNode& im_root_config = im_config_mgr.GetRootNode();
    const milvus::server::ConfigNode& im_invalid_config = im_root_config.GetChild("invalid_config");
    const milvus::server::ConfigNode& im_not_exit_config = im_root_config.GetChild("not_exit_config");
    ASSERT_EQ(im_not_exit_config.GetConfig().size(), 0);
    ASSERT_EQ(im_root_config.DumpString(), root_config.DumpString());
    ASSERT_EQ(im_invalid_config.DumpString(), invalid_config.DumpString());

    auto valus = invalid_config.GetSequence("not_exist");
    float ff = invalid_config.GetFloatValue("not_exist", 3.0);
    ASSERT_EQ(ff, 3.0);
    double not_exit_double = server_config.GetDoubleValue("not_exit", 3.0);
    ASSERT_EQ(not_exit_double, 3.0);
    int64_t not_exit_int64 = server_config.GetInt64Value("not_exit", 3);
    ASSERT_EQ(not_exit_int64, 3);
    int64_t not_exit_int32 = server_config.GetInt32Value("not_exit", 3);
    ASSERT_EQ(not_exit_int32, 3);
    bool not_exit_bool = server_config.GetBoolValue("not_exit", false);
    ASSERT_FALSE(not_exit_bool);

    std::string address = server_config.GetValue("address");
    ASSERT_TRUE(!address.empty());
    int64_t port = server_config.GetInt64Value("port");
    ASSERT_NE(port, 0);

    server_config.SetValue("float_test", "2.5");
    double dbl = server_config.GetDoubleValue("float_test");
    ASSERT_LE(std::fabs(dbl - 2.5), std::numeric_limits<double>::epsilon());
    float flt = server_config.GetFloatValue("float_test");
    ASSERT_LE(std::fabs(flt - 2.5), std::numeric_limits<float>::epsilon());

    server_config.SetValue("bool_test", "true");
    bool blt = server_config.GetBoolValue("bool_test");
    ASSERT_TRUE(blt);

    server_config.SetValue("int_test", "34");
    int32_t it32 = server_config.GetInt32Value("int_test");
    ASSERT_EQ(it32, 34);
    int64_t it64 = server_config.GetInt64Value("int_test");
    ASSERT_EQ(it64, 34);

    milvus::server::ConfigNode fake;
    server_config.AddChild("fake", fake);
    fake = server_config.GetChild("fake");
    milvus::server::ConfigNodeArr arr;
    server_config.GetChildren(arr);
    ASSERT_EQ(arr.size(), 1UL);

    server_config.ClearChildren();
    auto children = server_config.GetChildren();
    ASSERT_TRUE(children.empty());

    server_config.ClearConfig();
    auto configs = server_config.GetConfig();
    ASSERT_TRUE(configs.empty());

    server_config.AddSequenceItem("seq", "aaa");
    server_config.AddSequenceItem("seq", "bbb");
    auto seq = server_config.GetSequence("seq");
    ASSERT_EQ(seq.size(), 2UL);

    server_config.SetValue("fake", "fake");
    server_config.AddChild("fake", fake);
    milvus::server::ConfigNode combine;
    combine.Combine(server_config);

    combine.PrintAll();
    std::string all = combine.DumpString();
    ASSERT_TRUE(!all.empty());

    server_config.ClearSequences();
    auto seqs = server_config.GetSequences();
    ASSERT_TRUE(seqs.empty());
}

TEST_F(ConfigTest, SERVER_CONFIG_VALID_TEST) {
    std::string config_path(CONFIG_PATH);
    milvus::server::Config& config = milvus::server::Config::GetInstance();
    milvus::Status s;
    std::string str_val;
    int64_t int64_val;
    float float_val;
    bool bool_val;

    /* server config */
    std::string server_addr = "192.168.1.155";
    s = config.SetServerConfigAddress(server_addr);
    ASSERT_TRUE(s.ok());
    s = config.GetServerConfigAddress(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == server_addr);

    std::string server_port = "12345";
    s = config.SetServerConfigPort(server_port);
    ASSERT_TRUE(s.ok());
    s = config.GetServerConfigPort(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == server_port);

    std::string server_mode = "cluster_readonly";
    s = config.SetServerConfigDeployMode(server_mode);
    ASSERT_TRUE(s.ok());
    s = config.GetServerConfigDeployMode(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == server_mode);

    std::string server_time_zone = "UTC+6";
    s = config.SetServerConfigTimeZone(server_time_zone);
    ASSERT_TRUE(s.ok());
    s = config.GetServerConfigTimeZone(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == server_time_zone);

    /* db config */
    std::string db_primary_path = "/home/zilliz";
    s = config.SetDBConfigPrimaryPath(db_primary_path);
    ASSERT_TRUE(s.ok());
    s = config.GetDBConfigPrimaryPath(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == db_primary_path);

    std::string db_secondary_path = "/home/zilliz";
    s = config.SetDBConfigSecondaryPath(db_secondary_path);
    ASSERT_TRUE(s.ok());
    s = config.GetDBConfigSecondaryPath(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == db_secondary_path);

    std::string db_backend_url = "mysql://root:123456@127.0.0.1:19530/milvus";
    s = config.SetDBConfigBackendUrl(db_backend_url);
    ASSERT_TRUE(s.ok());
    s = config.GetDBConfigBackendUrl(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == db_backend_url);

    int64_t db_archive_disk_threshold = 100;
    s = config.SetDBConfigArchiveDiskThreshold(std::to_string(db_archive_disk_threshold));
    ASSERT_TRUE(s.ok());
    s = config.GetDBConfigArchiveDiskThreshold(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == db_archive_disk_threshold);

    int64_t db_archive_days_threshold = 365;
    s = config.SetDBConfigArchiveDaysThreshold(std::to_string(db_archive_days_threshold));
    ASSERT_TRUE(s.ok());
    s = config.GetDBConfigArchiveDaysThreshold(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == db_archive_days_threshold);

    int64_t db_insert_buffer_size = 2;
    s = config.SetDBConfigInsertBufferSize(std::to_string(db_insert_buffer_size));
    ASSERT_TRUE(s.ok());
    s = config.GetDBConfigInsertBufferSize(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == db_insert_buffer_size);

    /* metric config */
    bool metric_enable_monitor = false;
    s = config.SetMetricConfigEnableMonitor(std::to_string(metric_enable_monitor));
    ASSERT_TRUE(s.ok());
    s = config.GetMetricConfigEnableMonitor(bool_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(bool_val == metric_enable_monitor);

    std::string metric_collector = "prometheus";
    s = config.SetMetricConfigCollector(metric_collector);
    ASSERT_TRUE(s.ok());
    s = config.GetMetricConfigCollector(str_val);
    ASSERT_TRUE(str_val == metric_collector);

    std::string metric_prometheus_port = "2222";
    s = config.SetMetricConfigPrometheusPort(metric_prometheus_port);
    ASSERT_TRUE(s.ok());
    s = config.GetMetricConfigPrometheusPort(str_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(str_val == metric_prometheus_port);

    /* cache config */
    int64_t cache_cpu_cache_capacity = 1;
    s = config.SetCacheConfigCpuCacheCapacity(std::to_string(cache_cpu_cache_capacity));
    ASSERT_TRUE(s.ok());
    s = config.GetCacheConfigCpuCacheCapacity(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == cache_cpu_cache_capacity);

    float cache_cpu_cache_threshold = 0.1;
    s = config.SetCacheConfigCpuCacheThreshold(std::to_string(cache_cpu_cache_threshold));
    ASSERT_TRUE(s.ok());
    s = config.GetCacheConfigCpuCacheThreshold(float_val);
    ASSERT_TRUE(float_val == cache_cpu_cache_threshold);

    bool cache_insert_data = true;
    s = config.SetCacheConfigCacheInsertData(std::to_string(cache_insert_data));
    ASSERT_TRUE(s.ok());
    s = config.GetCacheConfigCacheInsertData(bool_val);
    ASSERT_TRUE(bool_val == cache_insert_data);

    /* engine config */
    int64_t engine_use_blas_threshold = 50;
    s = config.SetEngineConfigUseBlasThreshold(std::to_string(engine_use_blas_threshold));
    ASSERT_TRUE(s.ok());
    s = config.GetEngineConfigUseBlasThreshold(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == engine_use_blas_threshold);

    int64_t engine_omp_thread_num = 1;
    s = config.SetEngineConfigOmpThreadNum(std::to_string(engine_omp_thread_num));
    ASSERT_TRUE(s.ok());
    s = config.GetEngineConfigOmpThreadNum(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == engine_omp_thread_num);

#ifdef MILVUS_GPU_VERSION
    int64_t engine_gpu_search_threshold = 800;
    s = config.SetEngineConfigGpuSearchThreshold(std::to_string(engine_gpu_search_threshold));
    ASSERT_TRUE(s.ok());
    s = config.GetEngineConfigGpuSearchThreshold(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == engine_gpu_search_threshold);

    /* gpu resource config */
    bool resource_enable_gpu = true;
    s = config.SetGpuResourceConfigEnable(std::to_string(resource_enable_gpu));
    ASSERT_TRUE(s.ok());
    s = config.GetGpuResourceConfigEnable(bool_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(bool_val == resource_enable_gpu);

    int64_t gpu_cache_capacity = 1;
    s = config.SetGpuResourceConfigCacheCapacity(std::to_string(gpu_cache_capacity));
    ASSERT_TRUE(s.ok());
    s = config.GetGpuResourceConfigCacheCapacity(int64_val);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(int64_val == gpu_cache_capacity);

    float gpu_cache_threshold = 0.2;
    s = config.SetGpuResourceConfigCacheThreshold(std::to_string(gpu_cache_threshold));
    ASSERT_TRUE(s.ok());
    s = config.GetGpuResourceConfigCacheThreshold(float_val);
    ASSERT_TRUE(float_val == gpu_cache_threshold);

    std::vector<std::string> search_resources = {"gpu0"};
    std::vector<int64_t> search_res_vec;
    std::string search_res_str;
    milvus::server::StringHelpFunctions::MergeStringWithDelimeter(
        search_resources, milvus::server::CONFIG_GPU_RESOURCE_DELIMITER, search_res_str);
    s = config.SetGpuResourceConfigSearchResources(search_res_str);
    ASSERT_TRUE(s.ok());
    s = config.GetGpuResourceConfigSearchResources(search_res_vec);
    ASSERT_TRUE(s.ok());
    for (size_t i = 0; i < search_resources.size(); i++) {
        ASSERT_TRUE(std::stoll(search_resources[i].substr(3)) == search_res_vec[i]);
    }

    std::vector<std::string> build_index_resources = {"gpu0"};
    std::vector<int64_t> build_index_res_vec;
    std::string build_index_res_str;
    milvus::server::StringHelpFunctions::MergeStringWithDelimeter(
        build_index_resources, milvus::server::CONFIG_GPU_RESOURCE_DELIMITER, build_index_res_str);
    s = config.SetGpuResourceConfigBuildIndexResources(build_index_res_str);
    ASSERT_TRUE(s.ok());
    s = config.GetGpuResourceConfigBuildIndexResources(build_index_res_vec);
    ASSERT_TRUE(s.ok());
    for (size_t i = 0; i < build_index_resources.size(); i++) {
        ASSERT_TRUE(std::stoll(build_index_resources[i].substr(3)) == build_index_res_vec[i]);
    }
#endif
}

std::string
gen_get_command(const std::string& parent_node, const std::string& child_node) {
    std::string cmd = "get_config " + parent_node + ms::CONFIG_NODE_DELIMITER + child_node;
    return cmd;
}

std::string
gen_set_command(const std::string& parent_node, const std::string& child_node, const std::string& value) {
    std::string cmd = "set_config " + parent_node + ms::CONFIG_NODE_DELIMITER + child_node + " " + value;
    return cmd;
}

TEST_F(ConfigTest, SERVER_CONFIG_CLI_TEST) {
    std::string config_path(CONFIG_PATH);
    milvus::server::Config& config = milvus::server::Config::GetInstance();
    milvus::Status s;

    std::string get_cmd, set_cmd;
    std::string result, dummy;

    s = config.ProcessConfigCli(result, "get_config *");
    ASSERT_TRUE(s.ok());

    /* server config */
    std::string server_addr = "192.168.1.155";
    get_cmd = gen_get_command(ms::CONFIG_SERVER, ms::CONFIG_SERVER_ADDRESS);
    set_cmd = gen_set_command(ms::CONFIG_SERVER, ms::CONFIG_SERVER_ADDRESS, server_addr);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_FALSE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());

    /* db config */
    std::string db_primary_path = "/home/zilliz";
    get_cmd = gen_get_command(ms::CONFIG_DB, ms::CONFIG_DB_PRIMARY_PATH);
    set_cmd = gen_set_command(ms::CONFIG_DB, ms::CONFIG_DB_PRIMARY_PATH, db_primary_path);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_FALSE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());

    /* metric config */
    std::string metric_enable_monitor = "false";
    get_cmd = gen_get_command(ms::CONFIG_METRIC, ms::CONFIG_METRIC_ENABLE_MONITOR);
    set_cmd = gen_set_command(ms::CONFIG_METRIC, ms::CONFIG_METRIC_ENABLE_MONITOR, metric_enable_monitor);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_FALSE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());

    /* cache config */
    std::string cache_cpu_cache_capacity = "1";
    get_cmd = gen_get_command(ms::CONFIG_CACHE, ms::CONFIG_CACHE_CPU_CACHE_CAPACITY);
    set_cmd = gen_set_command(ms::CONFIG_CACHE, ms::CONFIG_CACHE_CPU_CACHE_CAPACITY, cache_cpu_cache_capacity);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == cache_cpu_cache_capacity);

    std::string cache_cpu_cache_threshold = "0.1";
    get_cmd = gen_get_command(ms::CONFIG_CACHE, ms::CONFIG_CACHE_CPU_CACHE_THRESHOLD);
    set_cmd = gen_set_command(ms::CONFIG_CACHE, ms::CONFIG_CACHE_CPU_CACHE_THRESHOLD, cache_cpu_cache_threshold);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(result == cache_cpu_cache_threshold);

    std::string cache_insert_data = "true";
    get_cmd = gen_get_command(ms::CONFIG_CACHE, ms::CONFIG_CACHE_CACHE_INSERT_DATA);
    set_cmd = gen_set_command(ms::CONFIG_CACHE, ms::CONFIG_CACHE_CACHE_INSERT_DATA, cache_insert_data);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(result == cache_insert_data);

    /* engine config */
    std::string engine_use_blas_threshold = "50";
    get_cmd = gen_get_command(ms::CONFIG_ENGINE, ms::CONFIG_ENGINE_USE_BLAS_THRESHOLD);
    set_cmd = gen_set_command(ms::CONFIG_ENGINE, ms::CONFIG_ENGINE_USE_BLAS_THRESHOLD, engine_use_blas_threshold);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == engine_use_blas_threshold);

    std::string engine_omp_thread_num = "1";
    get_cmd = gen_get_command(ms::CONFIG_ENGINE, ms::CONFIG_ENGINE_OMP_THREAD_NUM);
    set_cmd = gen_set_command(ms::CONFIG_ENGINE, ms::CONFIG_ENGINE_OMP_THREAD_NUM, engine_omp_thread_num);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == engine_omp_thread_num);

#ifdef MILVUS_GPU_VERSION
    std::string engine_gpu_search_threshold = "800";
    get_cmd = gen_get_command(ms::CONFIG_ENGINE, ms::CONFIG_ENGINE_GPU_SEARCH_THRESHOLD);
    set_cmd = gen_set_command(ms::CONFIG_ENGINE, ms::CONFIG_ENGINE_GPU_SEARCH_THRESHOLD, engine_gpu_search_threshold);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == engine_gpu_search_threshold);

    /* gpu resource config */
    std::string resource_enable_gpu = "true";
    get_cmd = gen_get_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_ENABLE);
    set_cmd = gen_set_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_ENABLE, resource_enable_gpu);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == resource_enable_gpu);

    std::string gpu_cache_capacity = "1";
    get_cmd = gen_get_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_CACHE_CAPACITY);
    set_cmd = gen_set_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_CACHE_CAPACITY, gpu_cache_capacity);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == gpu_cache_capacity);

    std::string gpu_cache_threshold = "0.2";
    get_cmd = gen_get_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_CACHE_THRESHOLD);
    set_cmd = gen_set_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_CACHE_THRESHOLD, gpu_cache_threshold);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(result == gpu_cache_threshold);

    std::string search_resources = "gpu0";
    get_cmd = gen_get_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_SEARCH_RESOURCES);
    set_cmd = gen_set_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_SEARCH_RESOURCES, search_resources);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == search_resources);

    std::string build_index_resources = "gpu0";
    get_cmd = gen_get_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES);
    set_cmd =
        gen_set_command(ms::CONFIG_GPU_RESOURCE, ms::CONFIG_GPU_RESOURCE_BUILD_INDEX_RESOURCES, build_index_resources);
    s = config.ProcessConfigCli(dummy, set_cmd);
    ASSERT_TRUE(s.ok());
    s = config.ProcessConfigCli(result, get_cmd);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(result == build_index_resources);
#endif
}

TEST_F(ConfigTest, SERVER_CONFIG_INVALID_TEST) {
    std::string config_path(CONFIG_PATH);
    milvus::server::Config& config = milvus::server::Config::GetInstance();
    milvus::Status s;

    s = config.LoadConfigFile("");
    ASSERT_FALSE(s.ok());

    s = config.LoadConfigFile(config_path + INVALID_CONFIG_FILE);
    ASSERT_FALSE(s.ok());
    s = config.LoadConfigFile(config_path + "dummy.yaml");
    ASSERT_FALSE(s.ok());

    /* server config */
    s = config.SetServerConfigAddress("0.0.0");
    ASSERT_FALSE(s.ok());
    s = config.SetServerConfigAddress("0.0.0.256");
    ASSERT_FALSE(s.ok());

    s = config.SetServerConfigPort("a");
    ASSERT_FALSE(s.ok());
    s = config.SetServerConfigPort("99999");
    ASSERT_FALSE(s.ok());

    s = config.SetServerConfigDeployMode("cluster");
    ASSERT_FALSE(s.ok());

    s = config.SetServerConfigTimeZone("GM");
    ASSERT_FALSE(s.ok());
    s = config.SetServerConfigTimeZone("GMT8");
    ASSERT_FALSE(s.ok());
    s = config.SetServerConfigTimeZone("UTCA");
    ASSERT_FALSE(s.ok());

    /* db config */
    s = config.SetDBConfigPrimaryPath("");
    ASSERT_FALSE(s.ok());

    //    s = config.SetDBConfigSecondaryPath("");
    //    ASSERT_FALSE(s.ok());

    s = config.SetDBConfigBackendUrl("http://www.google.com");
    ASSERT_FALSE(s.ok());
    s = config.SetDBConfigBackendUrl("sqlite://:@:");
    ASSERT_FALSE(s.ok());
    s = config.SetDBConfigBackendUrl("mysql://root:123456@127.0.0.1/milvus");
    ASSERT_FALSE(s.ok());

    s = config.SetDBConfigArchiveDiskThreshold("0x10");
    ASSERT_FALSE(s.ok());

    s = config.SetDBConfigArchiveDaysThreshold("0x10");
    ASSERT_FALSE(s.ok());

    s = config.SetDBConfigInsertBufferSize("a");
    ASSERT_FALSE(s.ok());
    s = config.SetDBConfigInsertBufferSize("0");
    ASSERT_FALSE(s.ok());
    s = config.SetDBConfigInsertBufferSize("2048");
    ASSERT_FALSE(s.ok());

    /* metric config */
    s = config.SetMetricConfigEnableMonitor("Y");
    ASSERT_FALSE(s.ok());

    s = config.SetMetricConfigCollector("zilliz");
    ASSERT_FALSE(s.ok());

    s = config.SetMetricConfigPrometheusPort("0xff");
    ASSERT_FALSE(s.ok());

    /* cache config */
    s = config.SetCacheConfigCpuCacheCapacity("a");
    ASSERT_FALSE(s.ok());
    s = config.SetCacheConfigCpuCacheCapacity("0");
    ASSERT_FALSE(s.ok());
    s = config.SetCacheConfigCpuCacheCapacity("2048");
    ASSERT_FALSE(s.ok());

    s = config.SetCacheConfigCpuCacheThreshold("a");
    ASSERT_FALSE(s.ok());
    s = config.SetCacheConfigCpuCacheThreshold("1.0");
    ASSERT_FALSE(s.ok());

    s = config.SetCacheConfigCacheInsertData("N");
    ASSERT_FALSE(s.ok());

    /* engine config */
    s = config.SetEngineConfigUseBlasThreshold("0xff");
    ASSERT_FALSE(s.ok());

    s = config.SetEngineConfigOmpThreadNum("a");
    ASSERT_FALSE(s.ok());
    s = config.SetEngineConfigOmpThreadNum("10000");
    ASSERT_FALSE(s.ok());

#ifdef MILVUS_GPU_VERSION
    s = config.SetEngineConfigGpuSearchThreshold("-1");
    ASSERT_FALSE(s.ok());

    /* gpu resource config */
    s = config.SetGpuResourceConfigEnable("ok");
    ASSERT_FALSE(s.ok());

    s = config.SetGpuResourceConfigCacheCapacity("a");
    ASSERT_FALSE(s.ok());
    s = config.SetGpuResourceConfigCacheCapacity("128");
    ASSERT_FALSE(s.ok());

    s = config.SetGpuResourceConfigCacheThreshold("a");
    ASSERT_FALSE(s.ok());
    s = config.SetGpuResourceConfigCacheThreshold("1.0");
    ASSERT_FALSE(s.ok());

    s = config.SetGpuResourceConfigSearchResources("gpu10");
    ASSERT_FALSE(s.ok());

    s = config.SetGpuResourceConfigBuildIndexResources("gup2");
    ASSERT_FALSE(s.ok());
    s = config.SetGpuResourceConfigBuildIndexResources("gpu16");
    ASSERT_FALSE(s.ok());
#endif
}

TEST_F(ConfigTest, SERVER_CONFIG_TEST) {
    std::string config_path(CONFIG_PATH);
    milvus::server::Config& config = milvus::server::Config::GetInstance();
    milvus::Status s = config.LoadConfigFile(config_path + VALID_CONFIG_FILE);
    ASSERT_TRUE(s.ok());

    s = config.ValidateConfig();
    ASSERT_TRUE(s.ok());

    std::string config_json_str;
    config.GetConfigJsonStr(config_json_str);
    std::cout << config_json_str << std::endl;

    s = config.ResetDefaultConfig();
    ASSERT_TRUE(s.ok());
}
