// Guard for the required-field registry (libs/lsxcommon/index_fields.h).
//
// The audit that produced this file found that a textual audit of the index is
// unreliable: keys are read through local lambdas and written through five
// different helpers, so each pattern-based scan reported a different answer, and
// the last three reports each contained false positives. This guard therefore
// does three NARROW, exactly-checkable things instead of trying to be exhaustive:
//
//   1. `layout` is emitted by EVERY converter that writes an index. That is the
//      field whose predecessor (`format`) rotted to 11/18 dead.
//   2. Every key registered as required is mentioned by the converter that owns
//      it. Broad literal search on purpose — `write_array`, `bool_array`,
//      `scalar` and a streamed `<<` all count, because the point is "the owning
//      converter knows this key", not "it uses one particular helper".
//   3. Every hard-failure read shape in a loader resolves to a registered key.
//      This one fails CLOSED: if a new required read appears and is not
//      registered, the test fails and a human adds the row.
//
// What it deliberately does NOT do: prove that every written key is read. That
// is not a correctness property (a key may exist for tooling), and the audit
// showed the question cannot be answered reliably by scanning anyway.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/index_fields.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<fs::path> Sources(const char *sub)
{
    std::vector<fs::path> out;
    std::error_code ec;
    const fs::path root = fs::path(LSX_SOURCE_DIR) / sub;
    for (const auto &e : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        const auto ext = e.path().extension().string();
        if (ext == ".cpp" || ext == ".cu") out.push_back(e.path());
    }
    return out;
}

// A converter writes an index when it opens model.index.json for output.
// Does this FILE write a model.index.json?
//
// The text heuristic (mentions the filename + has an ofstream) is what caught
// the 18 rotting `format` tags, so it stays. It needs one exclusion: a MODEL
// file reads the index and may legitimately mention it in prose while opening
// ofstreams for something else entirely (inkling's model writes logits dumps) —
// flagging that file demanded that a comment stop naming the file it talks
// about, which is the wrong trade. Converters, repack tools and every other
// index writer carry no `_model.cpp` suffix.
bool WritesAnIndex(const fs::path &path, const std::string &text)
{
    if (path.filename().string().size() > 10 &&
        path.filename().string().compare(path.filename().string().size() - 10,
                                         10, "_model.cpp") == 0)
        return false;
    return text.find("model.index.json") != std::string::npos &&
           text.find("ofstream") != std::string::npos;
}

// A loader is a file that parses an index and is not a converter or a golden
// reader.
bool ParsesAnIndex(const fs::path &p, const std::string &text)
{
    const auto n = p.filename().string();
    if (n.find("convert") != std::string::npos) return false;
    if (n.find("selfcheck") != std::string::npos) return false;
    return text.find("model.index.json") != std::string::npos;
}

} // namespace

TEST_CASE("every converter writes the layout generation", "[index_fields]") {
    // The field whose predecessor rotted: 11 of 18 `format` tags were written
    // and never read. `layout` is universal by construction instead.
    std::vector<std::string> missing;
    for (const auto &p : Sources("libs")) {
        const auto text = ReadFile(p);
        if (!WritesAnIndex(p, text)) continue;
        if (text.find("\\\"layout\\\":") == std::string::npos)
            missing.push_back(fs::relative(p, LSX_SOURCE_DIR).string());
    }
    for (const auto &m : missing) INFO(m);
    CHECK(missing.empty());
}

TEST_CASE("every required key is emitted by the converter that owns it",
          "[index_fields]") {
    int checked = 0;
    for (const auto &row : lsxcommon::kRequiredIndexFields) {
        // "" = universal: covered by the layout test above.
        if (row.owner_file == nullptr || *row.owner_file == '\0') continue;
        std::error_code ec;
        const fs::path owner =
            fs::path(LSX_SOURCE_DIR) / "libs" / "lsxcommon" / "src" / row.owner_file;
        INFO(row.key << " <- " << row.owner_file);
        REQUIRE(fs::is_regular_file(owner, ec));
        const auto text = ReadFile(owner);
        // Two spellings: a JSON fragment inside a C++ string literal is
        // `\"total_experts\"`, while a helper call is `"total_experts"`. Both
        // mean the owning converter knows the key.
        const std::string plain = std::string("\"") + row.key + "\"";
        const std::string escaped = std::string("\\\"") + row.key + "\\\"";
        CHECK((text.find(plain) != std::string::npos ||
               text.find(escaped) != std::string::npos));
        ++checked;
    }
    // A registry whose rows all point at a missing owner would pass vacuously.
    CHECK(checked >= 5);
}

TEST_CASE("the registry is internally consistent", "[index_fields]") {
    std::set<std::string> keys;
    for (const auto &row : lsxcommon::kRequiredIndexFields) {
        REQUIRE(row.key != nullptr);
        REQUIRE(*row.key != '\0');
        // A required key with no explanation is one nobody can safely remove.
        REQUIRE(row.why != nullptr);
        CHECK(std::string(row.why).size() > 20);
        CHECK(keys.insert(row.key).second);   // no duplicate rows
    }
    CHECK(keys.size() == lsxcommon::kRequiredIndexFieldCount);
}

TEST_CASE("every hard-failure read in a loader is registered", "[index_fields]") {
    // Two shapes only, both verified against the tree by hand:
    //   if (!ReadIntArray(doc, "k", ...) || ...)  -> error
    //   if (doc["k"].get_...() != simdjson::SUCCESS) { ...; return <error> }
    // This fails CLOSED: an unregistered required read is a triage event, not a
    // silent pass.
    std::set<std::string> registered;
    for (const auto &row : lsxcommon::kRequiredIndexFields)
        registered.insert(row.key);

    std::vector<std::string> unknown;
    for (const auto &p : Sources("libs")) {
        const auto text = ReadFile(p);
        if (!ParsesAnIndex(p, text)) continue;

        // Shape 1: negated helper inside a condition that returns an error.
        for (size_t at = text.find("!ReadIntArray("); at != std::string::npos;
             at = text.find("!ReadIntArray(", at + 1)) {
            const auto q1 = text.find('"', at);
            const auto q2 = text.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos) continue;
            const std::string key = text.substr(q1 + 1, q2 - q1 - 1);
            if (!registered.count(key))
                unknown.push_back(fs::relative(p, LSX_SOURCE_DIR).string() + ": " +
                                  key);
        }

        // Shape 2: direct subscript whose failure is followed by `return`.
        for (size_t at = text.find("[\""); at != std::string::npos;
             at = text.find("[\"", at + 1)) {
            const auto q2 = text.find("\"]", at + 2);
            if (q2 == std::string::npos) continue;
            const std::string key = text.substr(at + 2, q2 - at - 2);
            const auto line_end = text.find('\n', q2);
            if (line_end == std::string::npos) continue;
            const auto line = text.substr(line_end > at ? at : 0,
                                          line_end - (line_end > at ? at : 0));
            if (line.find("!= simdjson::SUCCESS") == std::string::npos) continue;
            // A defaulted read assigns instead of returning; only a `return`
            // within the next 200 characters makes the key required.
            const auto tail = text.substr(line_end, 200);
            if (tail.find("return") == std::string::npos) continue;
            if (tail.find("= 0") != std::string::npos) continue;
            if (!registered.count(key))
                unknown.push_back(fs::relative(p, LSX_SOURCE_DIR).string() + ": " +
                                  key);
        }
    }
    for (const auto &u : unknown) INFO(u);
    CHECK(unknown.empty());
}
