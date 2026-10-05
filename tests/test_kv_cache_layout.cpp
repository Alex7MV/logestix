// L1: the shared KV-cache geometry layer.
#include <catch2/catch_test_macros.hpp>
#include <set>

#include "lsxcommon/lsx_kv_cache.h"

using lsxcommon::KvCacheGeometry;
using lsxcommon::KvCacheLayout;

TEST_CASE("Make produces the documented strides", "[kv_layout][fast]")
{
    // Xing's real shape: 32 heads, K row 192 (nope 128 + rope 64), V row 128.
    const auto hp =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 32, 192, 40);
    CHECK(hp.pos_stride == 192);
    CHECK(hp.head_stride == 64 * 192);

    const auto pp =
        KvCacheGeometry::Make(KvCacheLayout::kPosMajor, 64, 32, 192, 40);
    CHECK(pp.pos_stride == 32 * 192);
    CHECK(pp.head_stride == 192);
}

TEST_CASE("LayerStride and Elements are layout-invariant", "[kv_layout][fast]")
{
    // The invariant that makes the swap size-neutral: only the decomposition of a
    // layer's block into head and position changes, never its size.
    const auto hp =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 128, 32, 192, 40);
    const auto pp =
        KvCacheGeometry::Make(KvCacheLayout::kPosMajor, 128, 32, 192, 40);
    CHECK(hp.LayerStride() == pp.LayerStride());
    CHECK(hp.Elements() == pp.Elements());
    CHECK(hp.LayerStride() == 32 * 128 * 192);
    CHECK(hp.Elements() == 40 * 32 * 128 * 192);
}

TEST_CASE("layouts are address bijections over head x position",
          "[kv_layout][fast]")
{
    // The strongest available property: both layouts cover the same set of
    // offsets. No position is aliased or skipped - which is what licenses the
    // bit-exact parity claim.
    const int64_t cap = 16, heads = 4, dim = 3, layers = 2;
    const auto hp =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, cap, heads, dim, layers);
    const auto pp =
        KvCacheGeometry::Make(KvCacheLayout::kPosMajor, cap, heads, dim, layers);

    std::set<int64_t> hs, ps;
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t p = 0; p < cap; ++p)
                for (int64_t d = 0; d < dim; ++d) {
                    hs.insert(hp.Offset(l, h, p, d));
                    ps.insert(pp.Offset(l, h, p, d));
                }
    CHECK(hs.size() == static_cast<size_t>(layers * heads * cap * dim));
    CHECK(hs == ps);
}

TEST_CASE("K and V of different widths do not overlap", "[kv_layout][fast]")
{
    // Xing: kQkHeadDim = 192, kVHeadDim = 128. Two independent geometries.
    const auto k =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 32, 192, 40);
    const auto v =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 32, 128, 40);
    CHECK(k.Elements() != v.Elements());
    CHECK(k.Offset(0, 0, 0) == 0);
    CHECK(v.Offset(0, 0, 0) == 0);
    // The last K element must stay inside K, not reach into V's range.
    CHECK(k.Offset(39, 31, 63, 191) == k.Elements() - 1);
    CHECK(v.Offset(39, 31, 63, 127) == v.Elements() - 1);
}

TEST_CASE("MLA latent stays contiguous within a head", "[kv_layout][fast]")
{
    // kv_lora 512 + rope 64 in one stored row: no interleaving.
    const int64_t lora = 512, rope = 64;
    const auto g = KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 32, 8,
                                         lora + rope, 4);
    const int64_t base = g.Offset(0, 3, 7, 0);
    for (int64_t d = 0; d < lora + rope; ++d)
        CHECK(g.Offset(0, 3, 7, d) == base + d);
    CHECK(g.Offset(0, 3, 8, 0) == base + lora + rope);
}

TEST_CASE("degenerate geometry yields zero elements", "[kv_layout][fast]")
{
    const auto z =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 0, 32, 192, 40);
    CHECK(z.Elements() == 0);
    const auto h =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 0, 192, 40);
    CHECK(h.Elements() == 0);
}

TEST_CASE("offsets are int64-wide", "[kv_layout][fast]")
{
    // 32 * 32768 * 192 * 40 = 8.05G elements: twice past int32. A wrapped index
    // under head-major yields plausible-looking data, so the width is pinned.
    const auto g =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 32768, 32, 192, 40);
    CHECK(g.Elements() == 8053063680LL);
    CHECK(g.Offset(39, 31, 32767, 191) == g.Elements() - 1);
}

TEST_CASE("a write at pos == capacity would cross into the next head",
          "[kv_layout][fast]")
{
    // Documents WHY the pos < capacity guard must survive the layout change: at
    // pos == capacity a pos-major write lands past the whole layer, but a
    // head-major write lands squarely inside the NEXT head's data. Both are
    // wrong; the second is quieter.
    const auto g =
        KvCacheGeometry::Make(KvCacheLayout::kHeadMajor, 64, 32, 192, 40);
    const int64_t overflow = g.Offset(0, 3, 64, 0);
    const int64_t head4 = g.Offset(0, 4, 0, 0);
    CHECK(overflow == head4);
    CHECK(overflow < g.Elements());
}

#include "lsxcommon/model.h"

TEST_CASE("default config is maximum acceleration", "[kv_layout][fast]")
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = "unused";
    CHECK(cfg.kv_cache_layout == lsxcommon::KvCacheLayout::kHeadMajor);
}
