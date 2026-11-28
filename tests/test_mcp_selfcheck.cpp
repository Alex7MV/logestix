// Host-only unit tests for the MCP protocol-payload gate.
//
// No GPU, no weights, no model: the harness's contribution is the CRITERION —
// extract the payload, canonicalize it, and compare documents rather than
// bytes — so that is what these assertions pin. The real-model gates
// (test_<family>_mcp_selfcheck) exercise the other half.
//
// Why the criterion is document equality and not byte equality is in
// docs/superpowers/specs/2026-11-28-mcp-selfcheck-design.md §3: an MCP payload
// is fixed by JSON-RPC 2.0, which says nothing about key order, so a byte
// comparison would measure whether the model guessed the template's layout.
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <lsxcommon/mcp_selfcheck.h>

namespace {

constexpr const char *kToolCall =
    "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":{\"name\":"
    "\"fetch_secure_ledger\",\"arguments\":{\"node_id\":\"NX-5090\"}},\"id\":1}";
constexpr const char *kError =
    "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32602,\"message\":"
    "\"Invalid params\"},\"id\":99}";

// The expected payload is itself JSON, so it must be JSON-escaped when it is
// embedded as a string value — an unescaped quote here silently makes the
// golden unparseable, which is what this helper first shipped with.
std::string Esc(std::string_view s)
{
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

std::string Golden(std::string_view expected, std::string_view kind,
                   std::string_view anchor)
{
    return std::string("{\"format\":\"logestix-mcp-golden\",\"version\":1,") +
           "\"model_type\":\"unit-model\",\"prompts\":[{\"id\":\"MCP-TEST-01\"," +
           "\"kind\":\"" + std::string(kind) + "\",\"anchor\":\"" +
           std::string(anchor) + "\",\"input_ids\":[1,2,3]," +
           "\"max_new_tokens\":64,\"expected\":\"" + Esc(expected) +
           "\",\"raw\":\"\"}]}";
}

} // namespace

TEST_CASE("mcp: a request payload is extracted out of prose",
          "[mcp][unit]")
{
    const std::string raw = std::string(
        "Sure. Here is the call you asked for:\n```json\n") +
        "{\n  \"jsonrpc\": \"2.0\",\n  \"method\": \"tools/call\",\n" +
        "  \"params\": {\"name\": \"fetch_secure_ledger\",\n" +
        "             \"arguments\": {\"node_id\": \"NX-5090\"}},\n" +
        "  \"id\": 1\n}\n```\nThe ledger node is NX-5090.";
    const std::string got = lsxcommon::mcp::ExtractMcpAnswer(
        lsxcommon::mcp::McpKind::kToolCall, "tools/call", raw);
    CHECK(got == kToolCall);
}

TEST_CASE("mcp: a stray object before the payload does not swallow it",
          "[mcp][unit]")
{
    // The old extractor took the FIRST balanced object and then failed the
    // anchor test, so a correct payload behind a one-key example read as an
    // empty extraction. The scan keeps looking.
    const std::string raw = std::string("{\"scratch\":true} then: ") + kToolCall;
    const std::string got = lsxcommon::mcp::ExtractMcpAnswer(
        lsxcommon::mcp::McpKind::kToolCall, "tools/call", raw);
    CHECK(got == kToolCall);
}

TEST_CASE("mcp: the reasoning chain does not shadow the payload",
          "[mcp][unit]")
{
    // A reasoning model writes an illustrative REQUEST while thinking and then
    // answers. The error case must not adopt the illustration: the shape filter
    // and the last-</think> cut both refuse it.
    const std::string raw =
        "I could build {\"jsonrpc\":\"2.0\",\"method\":\"tools/call\","
        "\"params\":{},\"id\":1} but the task asks for an error.\n"
        "</think>\n" + std::string(kError);
    const std::string got = lsxcommon::mcp::ExtractMcpAnswer(
        lsxcommon::mcp::McpKind::kErrorResponse, "Invalid params", raw);
    CHECK(got == kError);
}

TEST_CASE("mcp: no payload means an empty extraction, not a guess",
          "[mcp][unit]")
{
    const std::string raw = "I cannot construct that payload.";
    CHECK(lsxcommon::mcp::ExtractMcpAnswer(lsxcommon::mcp::McpKind::kToolCall,
                                           "tools/call", raw).empty());
    // A payload without the anchor is not the answer to this prompt.
    CHECK(lsxcommon::mcp::ExtractMcpAnswer(lsxcommon::mcp::McpKind::kToolCall,
                                           "resources/list", kToolCall).empty());
    // A request is not an error response, even with an anchor-ish body.
    CHECK(lsxcommon::mcp::ExtractMcpAnswer(
              lsxcommon::mcp::McpKind::kErrorResponse, "jsonrpc", kToolCall)
              .empty());
}

TEST_CASE("mcp: key order and whitespace are insignificant", "[mcp][unit]")
{
    // The criterion, stated as an assertion: JSON-RPC does not order keys.
    const std::string reordered =
        "{ \"id\": 1,\n  \"params\": {\"arguments\": {\"node_id\": \"NX-5090\"},"
        " \"name\": \"fetch_secure_ledger\"},\n  \"method\": \"tools/call\","
        "  \"jsonrpc\": \"2.0\" }";
    CHECK(lsxcommon::mcp::JsonDocumentDiff(kToolCall, reordered).empty());

    std::string canon_expected;
    std::string canon_got;
    std::string err;
    REQUIRE(lsxcommon::mcp::CanonicalizeJson(kToolCall, &canon_expected, &err));
    REQUIRE(lsxcommon::mcp::CanonicalizeJson(reordered, &canon_got, &err));
    CHECK(canon_expected == canon_got);
    // Canonical form is key-sorted and unspaced, so the verdict line is stable
    // whatever the model emitted.
    CHECK(canon_expected.rfind("{\"id\":1,\"jsonrpc\":\"2.0\"", 0) == 0);
}

TEST_CASE("mcp: a real protocol difference is reported, and named",
          "[mcp][unit]")
{
    SECTION("wrong error code")
    {
        const std::string got =
            "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32601,\"message\":"
            "\"Invalid params\"},\"id\":99}";
        const std::string why = lsxcommon::mcp::JsonDocumentDiff(kError, got);
        CHECK_FALSE(why.empty());
        // The verdict must say WHICH field, or a reader cannot tell a protocol
        // mistake from a missing payload.
        CHECK(why.find("error.code") != std::string::npos);
        CHECK(why.find("-32601") != std::string::npos);
    }
    SECTION("wrong session id")
    {
        const std::string got =
            "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32602,\"message\":"
            "\"Invalid params\"},\"id\":98}";
        const std::string why = lsxcommon::mcp::JsonDocumentDiff(kError, got);
        CHECK(why.find("id") != std::string::npos);
    }
    SECTION("missing argument")
    {
        const std::string got =
            "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":"
            "{\"name\":\"fetch_secure_ledger\",\"arguments\":{}},\"id\":1}";
        const std::string why = lsxcommon::mcp::JsonDocumentDiff(kToolCall, got);
        CHECK(why.find("node_id") != std::string::npos);
    }
    SECTION("extra key is not silently tolerated")
    {
        const std::string got =
            "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"params\":"
            "{\"name\":\"fetch_secure_ledger\",\"arguments\":"
            "{\"node_id\":\"NX-5090\"}},\"id\":1,\"note\":\"hi\"}";
        const std::string why = lsxcommon::mcp::JsonDocumentDiff(kToolCall, got);
        CHECK(why.find("note") != std::string::npos);
    }
    SECTION("the protocol version is not negotiable")
    {
        const std::string got =
            "{\"jsonrpc\":\"1.0\",\"method\":\"tools/call\",\"params\":"
            "{\"name\":\"fetch_secure_ledger\",\"arguments\":"
            "{\"node_id\":\"NX-5090\"}},\"id\":1}";
        CHECK(lsxcommon::mcp::JsonDocumentDiff(kToolCall, got)
                  .find("jsonrpc") != std::string::npos);
    }
}

TEST_CASE("mcp: canonicalization refuses what is not JSON", "[mcp][unit]")
{
    std::string out;
    std::string err;
    CHECK_FALSE(lsxcommon::mcp::CanonicalizeJson("not json", &out, &err));
    CHECK_FALSE(err.empty());
    CHECK(out.empty());
}

TEST_CASE("mcp: the golden carries a protocol message or is rejected",
          "[mcp][unit]")
{
    lsxcommon::mcp::McpGolden golden;
    std::string error;

    SECTION("a valid golden parses")
    {
        REQUIRE(lsxcommon::mcp::ParseMcpGolden(
            Golden(kToolCall, "tool_call", "tools/call"), golden, &error));
        REQUIRE(golden.cases.size() == 1);
        CHECK(golden.cases[0].id == "MCP-TEST-01");
        CHECK(golden.cases[0].anchor == "tools/call");
        CHECK(golden.cases[0].input_ids == std::vector<int32_t>{1, 2, 3});
        CHECK(golden.cases[0].kind == lsxcommon::mcp::McpKind::kToolCall);
    }
    SECTION("jsonrpc is not \"2.0\"")
    {
        const std::string bad =
            Golden("{\"jsonrpc\":\"1.0\",\"method\":\"tools/call\",\"id\":1}",
                   "tool_call", "tools/call");
        CHECK_FALSE(lsxcommon::mcp::ParseMcpGolden(bad, golden, &error));
        CHECK(error.find("JSON-RPC 2.0") != std::string::npos);
    }
    SECTION("a method and an error together are not a protocol message")
    {
        const std::string bad = Golden(
            "{\"jsonrpc\":\"2.0\",\"method\":\"tools/call\",\"error\":"
            "{\"code\":-32602,\"message\":\"x\"},\"id\":1}",
            "tool_call", "tools/call");
        CHECK_FALSE(lsxcommon::mcp::ParseMcpGolden(bad, golden, &error));
    }
    SECTION("an unknown kind is an error, never a silent tool_call")
    {
        CHECK_FALSE(lsxcommon::mcp::ParseMcpGolden(
            Golden(kToolCall, "tool_calls", "tools/call"), golden, &error));
        CHECK(error.find("unknown kind") != std::string::npos);
    }
}
