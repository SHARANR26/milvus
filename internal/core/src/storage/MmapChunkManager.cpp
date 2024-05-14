// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "storage/MmapChunkManager.h"
#include "storage/LocalChunkManagerSingleton.h"
#include <fstream>
#include <sys/mman.h>
#include <unistd.h>
#include "stdio.h"
#include <fcntl.h>
#include "log/Log.h"

namespace milvus::storage {
namespace {
static constexpr int kMmapDefaultProt = PROT_WRITE | PROT_READ;
static constexpr int kMmapDefaultFlags = MAP_SHARED;
};  // namespace

// todo(cqy): After confirming the append parallelism of multiple fields, adjust the lock granularity.

MmapBlock::MmapBlock(const std::string& file_name,
                     const uint64_t file_size,
                     BlockType type)
    : file_name_(file_name), file_size_(file_size), block_type_(type) {
    allocated_size_.fetch_add(file_size);
    // create tmp file
    int fd = open(file_name_.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        LOG_ERROR("Failed to open mmap tmp file");
        return;
    }
    // append file size to 'file_size'
    if (lseek(fd, file_size - 1, SEEK_SET) == -1) {
        LOG_ERROR("Failed to seek mmap tmp file");
        return;
    }
    if (write(fd, "", 1) == -1) {
        LOG_ERROR("Failed to write mmap tmp file");
        return;
    }
    // memory mmaping
    addr_ = static_cast<char*>(
        mmap(nullptr, file_size, kMmapDefaultProt, kMmapDefaultFlags, fd, 0));
    if (addr_ == MAP_FAILED) {
        LOG_ERROR("Failed to mmap");
        return;
    }
    offset_.store(0);
}

MmapBlock::~MmapBlock() {
    if (addr_ != nullptr) {
        munmap(addr_, file_size_);
    }
    if (std::ifstream(file_name_.c_str()).good()) {
        std::remove(file_name_.c_str());
    }
    allocated_size_.fetch_sub(file_size_);
}

void*
MmapBlock::Get(const uint64_t size, ErrorCode& error_code) {
    if (file_size_ - offset_.load() < size) {
        error_code = ErrorCode::MemAllocateSizeNotMatch;
        return nullptr;
    } else {
        error_code = ErrorCode::Success;
        return (void*)(addr_ + offset_.fetch_add(size));
    }
}

MmapBlockPtr
MmapBlocksHandler::AllocateFixSizeBlock(ErrorCode& error_code) {
    if (fix_size_blocks_cache_.size() != 0) {
        // return a mmap_block in fix_size_blocks_cache_
        auto block = std::move(fix_size_blocks_cache_.front());
        std::cout << "find a block in cache " << block.get() << std::endl;
        fix_size_blocks_cache_.pop();
        error_code = ErrorCode::Success;
        return std::move(block);
    } else {
        // if space not enough for create a new block, clear cache and check again
        if (fix_mmap_file_size_ + Size() > max_disk_limit_) {
            error_code = ErrorCode::MmapSpaceInsufficient;
            return nullptr;
        }
        std::cout << "generate a new block " << std::endl;
        error_code = ErrorCode::Success;
        return std::move(std::make_unique<MmapBlock>(
            GetMmapFilePath(), GetFixFileSize(), MmapBlock::BlockType::Fixed));
    }
}

MmapBlockPtr
MmapBlocksHandler::AllocateLargeBlock(const uint64_t size,
                                      ErrorCode& error_code) {
    if (fix_mmap_file_size_ + Capacity() > max_disk_limit_) {
        ClearCache();
    }
    if (fix_mmap_file_size_ + Size() > max_disk_limit_) {
        error_code = ErrorCode::MmapSpaceInsufficient;
        return nullptr;
    }
    error_code = ErrorCode::Success;
    return std::move(std::make_unique<MmapBlock>(
        GetMmapFilePath(), size, MmapBlock::BlockType::Variable));
}

void
MmapBlocksHandler::Deallocate(MmapBlockPtr&& block) {
    if (block->GetType() == MmapBlock::BlockType::Fixed) {
        // store the block in cache
        block->Clear();
        fix_size_blocks_cache_.push(std::move(block));
        uint64_t max_cache_size =
            uint64_t(cache_threshold * (float)max_disk_limit_);
        if (fix_size_blocks_cache_.size() * fix_mmap_file_size_ >
            max_cache_size) {
            FitCache(max_cache_size);
        }
    } else {
        // release the mmap
        block = nullptr;
    }
}

void
MmapBlocksHandler::ClearCache() {
    while (!fix_size_blocks_cache_.empty()) {
        fix_size_blocks_cache_.pop();
    }
}

void
MmapBlocksHandler::FitCache(const uint64_t size) {
    while (fix_size_blocks_cache_.size() * fix_mmap_file_size_ > size) {
        fix_size_blocks_cache_.pop();
    }
}

MmapChunkManager::MmapChunkManager(const std::string prefix,
                                   const uint64_t max_limit,
                                   const uint64_t file_size)
    : blocks_handler_(max_limit, file_size, prefix), mmap_file_prefix_(prefix) {
    auto cm =
        storage::LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    AssertInfo(cm != nullptr,
               "Fail to get LocalChunkManager, LocalChunkManagerPtr is null");
    if (cm->Exist(prefix)) {
        cm->RemoveDir(prefix);
    }
    cm->CreateDir(prefix);
}

MmapChunkManager::~MmapChunkManager() {
    auto cm =
        storage::LocalChunkManagerSingleton::GetInstance().GetChunkManager();
    if (cm->Exist(mmap_file_prefix_)) {
        cm->RemoveDir(mmap_file_prefix_);
    }
}

void
MmapChunkManager::Register(const uint64_t key) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    if (HasKey(key)) {
        LOG_WARN("key has exist in growing mmap manager");
        return;
    }
    blocks_table_.emplace(key, std::vector<MmapBlockPtr>());
    return;
}

void
MmapChunkManager::UnRegister(const uint64_t key) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    if (blocks_table_.find(key) != blocks_table_.end()) {
        auto& blocks = blocks_table_[key];
        for (auto i = 0; i < blocks.size(); i++) {
            blocks_handler_.Deallocate(std::move(blocks[i]));
        }
        blocks_table_.erase(key);
    }
}

inline bool
MmapChunkManager::HasKey(const uint64_t key) {
    return (blocks_table_.find(key) != blocks_table_.end());
}

void*
MmapChunkManager::Allocate(const uint64_t key,
                           const uint64_t size,
                           ErrorCode& error_code) {
    std::unique_lock<std::shared_mutex> lck(mtx_);
    if (!HasKey(key)) {
        error_code = ErrorCode::MemAllocateFailed;
        return nullptr;
    }
    if (size < blocks_handler_.GetFixFileSize()) {
        // find a place to fix in
        for (auto block_id = 0; block_id < blocks_table_[key].size();
             block_id++) {
            auto addr = blocks_table_[key][block_id]->Get(size, error_code);
            if (error_code == ErrorCode::Success) {
                return addr;
            }
        }
        // create a new block
        auto new_block = blocks_handler_.AllocateFixSizeBlock(error_code);
        if (error_code != ErrorCode::Success && new_block != nullptr) {
            return nullptr;
        }
        auto addr = new_block->Get(size, error_code);
        blocks_table_[key].emplace_back(std::move(new_block));
        return addr;
    } else {
        auto new_block = blocks_handler_.AllocateLargeBlock(size, error_code);
        if (error_code != ErrorCode::Success && new_block != nullptr) {
            return nullptr;
        }
        auto addr = new_block->Get(size, error_code);
        blocks_table_[key].emplace_back(std::move(new_block));
        return addr;
    }
}
}  // namespace milvus::storage