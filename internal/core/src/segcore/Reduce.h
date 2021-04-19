#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>

#include "utils/Status.h"

namespace milvus::segcore {
Status
merge_into(int64_t num_queries,
           int64_t topk,
           float* distances,
           int64_t* uids,
           const float* new_distances,
           const int64_t* new_uids);
}  // namespace milvus::segcore
