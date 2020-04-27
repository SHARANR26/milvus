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

#include "utils/LogUtil.h"

#include <libgen.h>
#include <cctype>
#include <string>

#include <yaml-cpp/yaml.h>

#include "config/Config.h"
#include "utils/Log.h"

namespace milvus {
namespace server {

namespace {
static int global_idx = 0;
static int debug_idx = 0;
static int warning_idx = 0;
static int trace_idx = 0;
static int error_idx = 0;
static int fatal_idx = 0;
}  // namespace

// TODO(yzb) : change the easylogging library to get the log level from parameter rather than filename
void
RolloutHandler(const char* filename, std::size_t size, el::Level level) {
    char* dirc = strdup(filename);
    char* basec = strdup(filename);
    char* dir = dirname(dirc);
    char* base = basename(basec);

    std::string s(base);
    std::string list[] = {"\\", " ", "\'", "\"", "*", "\?", "{", "}", ";", "<",
                          ">",  "|", "^",  "&",  "$", "#",  "!", "`", "~"};
    std::string::size_type position;
    for (auto substr : list) {
        position = 0;
        while ((position = s.find_first_of(substr, position)) != std::string::npos) {
            s.insert(position, "\\");
            position += 2;
        }
    }
    int ret;
    std::string m(std::string(dir) + "/" + s);
    s = m;
    switch (level) {
        case el::Level::Debug:
            s.append("." + std::to_string(++debug_idx));
            ret = rename(m.c_str(), s.c_str());
            break;
        case el::Level::Warning:
            s.append("." + std::to_string(++warning_idx));
            ret = rename(m.c_str(), s.c_str());
            break;
        case el::Level::Trace:
            s.append("." + std::to_string(++trace_idx));
            ret = rename(m.c_str(), s.c_str());
            break;
        case el::Level::Error:
            s.append("." + std::to_string(++error_idx));
            ret = rename(m.c_str(), s.c_str());
            break;
        case el::Level::Fatal:
            s.append("." + std::to_string(++fatal_idx));
            ret = rename(m.c_str(), s.c_str());
            break;
        default:
            s.append("." + std::to_string(++global_idx));
            ret = rename(m.c_str(), s.c_str());
            break;
    }
}

Status
InitLog(const std::string& logs_path) {
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.setGlobally(el::ConfigurationType::Format, "[%datetime][%level]%msg");
    defaultConf.setGlobally(el::ConfigurationType::ToFile, "true");
    defaultConf.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
    defaultConf.setGlobally(el::ConfigurationType::SubsecondPrecision, "3");
    defaultConf.setGlobally(el::ConfigurationType::PerformanceTracking, "false");
    defaultConf.setGlobally(el::ConfigurationType::MaxLogFileSize,"209715200");

    std::string global_log_path = logs_path + "milvus-%datetime{%y-%M-%d-%H:%m}-global.log";
    defaultConf.set(el::Level::Global, el::ConfigurationType::Filename, global_log_path.c_str());
    defaultConf.set(el::Level::Global, el::ConfigurationType::Enabled, "true");

    std::string info_log_path = logs_path + "milvus-%datetime{%y-%M-%d-%H:%m}-info.log";
    defaultConf.set(el::Level::Info, el::ConfigurationType::Filename, info_log_path.c_str());
    defaultConf.set(el::Level::Info, el::ConfigurationType::Enabled, "true");

    std::string debug_log_path = logs_path + "milvus-%datetime{%y-%M-%d-%H:%m}-debug.log";
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Filename, debug_log_path.c_str());
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Enabled, "true");

    std::string warning_log_path = logs_path + "milvus-%datetime{%y-%M-%d-%H:%m}-warning.log";
    defaultConf.set(el::Level::Warning, el::ConfigurationType::Filename, warning_log_path.c_str());
    defaultConf.set(el::Level::Warning, el::ConfigurationType::Enabled, "true");

    std::string trace_log_path = logs_path + "milvus-%datetime{%y-%M-%d-%H:%m}-trace.log";
    defaultConf.set(el::Level::Trace, el::ConfigurationType::Filename, trace_log_path.c_str());
    defaultConf.set(el::Level::Trace, el::ConfigurationType::Enabled, "true");

    std::string error_log_path = logs_path + "milvus-%datetime{%y-%M-%d-%H:%m}-error.log";
    defaultConf.set(el::Level::Error, el::ConfigurationType::Filename, error_log_path.c_str());
    defaultConf.set(el::Level::Error, el::ConfigurationType::Enabled, "true");

    std::string fatal_log_path = logs_path + "milvus-%datetime{%y-%M-%d-%H:%m}-fatal.log";
    defaultConf.set(el::Level::Fatal, el::ConfigurationType::Filename, fatal_log_path.c_str());
    defaultConf.set(el::Level::Fatal, el::ConfigurationType::Enabled, "true");

    el::Loggers::reconfigureLogger("default", defaultConf);

    el::Loggers::addFlag(el::LoggingFlag::StrictLogFileSizeCheck);
    el::Helpers::installPreRollOutCallback(RolloutHandler);
    el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);

    return Status::OK();
}

void
LogConfigInFile(const std::string& path) {
    // TODO(yhz): Check if file exists
    auto node = YAML::LoadFile(path);
    YAML::Emitter out;
    out << node;
    LOG_SERVER_INFO_ << "\n\n"
                     << std::string(15, '*') << "Config in file" << std::string(15, '*') << "\n\n"
                     << out.c_str();
}

void
LogConfigInMem() {
    auto& config = Config::GetInstance();
    std::string config_str;
    config.GetConfigJsonStr(config_str, 3);
    LOG_SERVER_INFO_ << "\n\n"
                     << std::string(15, '*') << "Config in memory" << std::string(15, '*') << "\n\n"
                     << config_str;
}

void
LogCpuInfo() {
    /*CPU information*/
    std::fstream fcpu("/proc/cpuinfo", std::ios::in);
    if (!fcpu.is_open()) {
        LOG_SERVER_WARNING_ << "Cannot obtain CPU information. Open file /proc/cpuinfo fail: " << strerror(errno);
        return;
    }
    std::stringstream cpu_info_ss;
    cpu_info_ss << fcpu.rdbuf();
    fcpu.close();
    std::string cpu_info = cpu_info_ss.str();

    auto processor_pos = cpu_info.rfind("processor");
    if (std::string::npos == processor_pos) {
        LOG_SERVER_WARNING_ << "Cannot obtain CPU information. No sub string \'processor\'";
        return;
    }

    auto sub_str = cpu_info.substr(processor_pos);
    LOG_SERVER_INFO_ << "\n\n" << std::string(15, '*') << "CPU" << std::string(15, '*') << "\n\n" << sub_str;
}

}  // namespace server
}  // namespace milvus
