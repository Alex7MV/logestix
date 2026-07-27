# Loader Round-Trip Test Design

## Purpose
Verify `lsxcommon::Loader` can successfully read `.arrow` expert files produced by `ExpertSchema`/`SaveExpertToArrow`, with bit-exact data integrity.

## Approach
Programmatically generate a `PackedExpert` with known data, round-trip through file I/O (`ExpertSchema::MakeBatch` → `WriteRecordBatchToFile` → `Loader::LoadExpertFromArrow`), and compare all fields.

## Test Cases

### 1. `Loader round-trip with synthetic PackedExpert`

**Input data:**
| Field | Content | Size |
|---|---|---|
| `w1_packed` | `[0x00..0xFF] × 4` | 1024 bytes |
| `w2_packed` | `[0xFF..0x00] × 4` | 1024 bytes |
| `w3_packed` | alternating `0xAA, 0xBB` | 512 bytes |
| `w1_scales` | `[0.5f, 1.0f, 1.5f, 2.0f]` | 4 floats |
| `w2_scales` | `[0.25f, 0.75f]` | 2 floats |
| `w3_scales` | `[1.0f]` | 1 float |
| `w*_shape` | `{}` (empty — not validated by loader) | — |

**Steps:**
1. Build `PackedExpert` from above
2. `ExpertSchema::MakeBatch(expert)` → `RecordBatch`; assert `ValidateFull().ok()`
3. `WriteRecordBatchToFile(batch, temp_file)`
4. `Loader::LoadExpertFromArrow(temp_file)` → `LoadedExpert`
5. Compare:
   - `memcmp(w1_weight, w1_packed)` across `w1_weight_size`, same for w2/w3
   - `w1_scale[i] == w1_scales[i]` element-by-element, same for w2/w3
   - size/count fields match
6. Delete temp file

### 2. `Load expert with empty fields`

**Input:** `w1_packed` empty, `w1_scales` empty, others normal.

**Asserts:** `w1_weight_size == 0`, `w1_scale_count == 0`, load succeeds without error.

## File Structure

**New test source:** `tests/test_loader.cpp`
**New test target:** `test_loader` in `tests/CMakeLists.txt`

## Build Changes (`tests/CMakeLists.txt`)

```cmake
add_executable(test_loader test_loader.cpp)
target_link_libraries(test_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_loader>"
    )
    add_custom_command(TARGET test_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_loader>"
    )
endif()

add_test(NAME test_loader COMMAND test_loader)
```

## Error Handling
- All Arrow operations checked via `ASSERT_TRUE(status.ok())` / `REQUIRE(result.ok())`
- Temp files cleaned up even on failure (RAII helper or defer)
