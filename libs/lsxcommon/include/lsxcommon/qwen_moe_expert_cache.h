#ifndef LSXCOMMON_QWEN_MOE_EXPERT_CACHE_H
#define LSXCOMMON_QWEN_MOE_EXPERT_CACHE_H

// Compatibility surface for the unified expert device cache
// (lsxcommon/expert_device_cache.h). The LRU logic lives once in
// gpu::ExpertDeviceCache; these aliases and delegating helpers keep the
// Qwen-MoE call sites stable.

#include "lsxcommon/expert_device_cache.h"

namespace lsxcommon {
namespace qwen_moe {

using QwenMoeExpertDeviceCache = gpu::ExpertDeviceCache;

inline void QwenMoeExpertCacheTouch(QwenMoeExpertDeviceCache &c, int slot)
{
    c.Touch(slot);
}
inline int QwenMoeExpertCacheLookup(QwenMoeExpertDeviceCache &c, int flat)
{
    return c.Lookup(flat);
}
inline int QwenMoeExpertCacheInsert(QwenMoeExpertDeviceCache &c, int flat)
{
    return c.Insert(flat);
}

} // namespace qwen_moe
} // namespace lsxcommon

#endif // LSXCOMMON_QWEN_MOE_EXPERT_CACHE_H
