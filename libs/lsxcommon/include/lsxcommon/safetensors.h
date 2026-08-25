#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_begin;
    int64_t data_end;

    int64_t numel() const;
    int64_t byte_size() const;
};

struct SafetensorsModel {
    struct Shard {
        std::string filename;
        int64_t file_size;
        std::vector<TensorInfo> tensors;
    };
    std::vector<Shard> shards;
};

SafetensorsModel::Shard read_safetensors_file(const std::string &path);
SafetensorsModel scan_directory(const std::string &path);

void print_shard(const SafetensorsModel::Shard &shard);
void print_model(const SafetensorsModel &model);

} // namespace lsxcommon
