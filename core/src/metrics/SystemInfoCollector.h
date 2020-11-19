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

#include <mutex>
#include <thread>

#include "metrics/Prometheus.h"

#define GAUGE(name, description)                  \
    prometheus::Family<prometheus::Gauge>& name = \
        prometheus::BuildGauge().Name(cluster_name + #name).Help(description).Register(prometheus.registry());

namespace milvus {

class SystemInfoCollector {
 public:
    static SystemInfoCollector&
    GetInstance() {
        static SystemInfoCollector instance;
        return instance;
    }

    void
    Start();

    void
    Stop();

 private:
    SystemInfoCollector() = default;

    void
    collector_function();

 private:
    bool running_ = false;
    std::mutex mutex_;

    std::thread collector_thread_;

    /* metrics */
    template <typename T>
    using Family = prometheus::Family<T>;
    using Gauge = prometheus::Gauge;

    /* cpu_utilization_ratio */
    Family<Gauge>& cpu_utilization_ratio_family_ = prometheus::BuildGauge()
                                                       .Name("cpu_utilization_ratio")
                                                       .Help("cpu_utilization_ratio")
                                                       .Register(prometheus.registry());
    Gauge& cpu_utilization_ratio_ = cpu_utilization_ratio_family_.Add({});

    Family<Gauge>& cpu_tempearature_family_ =
        prometheus::BuildGauge().Name("cpu_temperature").Help("cpu_temperature").Register(prometheus.registry());
    Gauge& cpu_temperature_ = cpu_tempearature_family_.Add({});

    Family<Gauge>& mem_usage_family_ =
        prometheus::BuildGauge().Name("mem_usage").Help("mem_usage").Register(prometheus.registry());
    Gauge& mem_usage_ = mem_usage_family_.Add({});

    Family<Gauge>& mem_total_family_ =
        prometheus::BuildGauge().Name("mem_total").Help("mem_total").Register(prometheus.registry());
    Gauge& mem_total_ = mem_total_family_.Add({});

    Family<Gauge>& mem_available_family_ =
        prometheus::BuildGauge().Name("mem_available").Help("mem_available").Register(prometheus.registry());
    Gauge& mem_available_ = mem_available_family_.Add({});

    Family<Gauge>& network_in_octets_family_ =
        prometheus::BuildGauge().Name("network_in_octets").Help("network_in_octets").Register(prometheus.registry());
    Gauge& network_in_octets_ = network_in_octets_family_.Add({});

    Family<Gauge>& network_out_octets_family_ =
        prometheus::BuildGauge().Name("network_out_octets").Help("network_out_octets").Register(prometheus.registry());
    Gauge& network_out_octets_ = network_out_octets_family_.Add({});
};

}  // namespace milvus
