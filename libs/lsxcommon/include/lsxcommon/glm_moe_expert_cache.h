#ifndef LSXCOMMON_GLM_EXPERT_CACHE_H
#define LSXCOMMON_GLM_EXPERT_CACHE_H

// Compatibility surface for the unified expert device cache
// (lsxcommon/expert_device_cache.h). The LRU logic lives once in
// gpu::ExpertDeviceCache; these aliases and delegating helpers keep the GLM
// call sites stable.

#include "lsxcommon/expert_device_cache.h"

namespace lsxcommon {
namespace glm {

using GlmExpertDeviceCache = gpu::ExpertDeviceCache;

inline void CacheTouch(GlmExpertDeviceCache &c, int slot) { c.Touch(slot); }
inline int CacheLookup(GlmExpertDeviceCache &c, int flat)
{
    return c.Lookup(flat);
}
inline int CacheInsert(GlmExpertDeviceCache &c, int flat)
{
    return c.Insert(flat);
}

} // namespace glm
} // namespace lsxcommon

#endif // LSXCOMMON_GLM_EXPERT_CACHE_H
