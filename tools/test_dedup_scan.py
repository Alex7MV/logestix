"""Unit tests for tools/dedup_scan.py — the duplication-residue instrument.

Host-only, no CUDA, no repo weights: synthetic fixtures in a temp directory.
Run: python3 tools/test_dedup_scan.py  (or via ctest test_dedup_scan).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN = ROOT / "tools" / "dedup_scan.py"

KERNEL_A = """\
// family a
__global__ void cast_kernel(const float *__restrict__ src,
                            __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}
"""

# Same body, different name/comments/whitespace: must group after normalization
# only when the NAME is irrelevant — bodies are compared, names are not.
KERNEL_B = """\
/* family b, another comment */
__global__ void totally_other_name(const float *__restrict__ src,
                                   __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)  return;
    dst[i] = __float2half(src[i]);
}
"""

KERNEL_DIFF = """\
__global__ void cast_kernel(const float *__restrict__ src,
                            __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i] * 2.0f);
}
"""

KERNEL_TINY = """\
__global__ void tiny_kernel(float *x) { x[0] = 1.0f; }
"""

HOST_BODY = """\
cudaError_t UploadBytes(const void *src, void *dst, size_t n, cudaStream_t s)
{
    if (n == 0) return cudaSuccess;
    if (src == nullptr) return cudaErrorInvalidValue;
    if (dst == nullptr) return cudaErrorInvalidValue;
    cudaError_t e = cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) return e;
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    return cudaSuccess;
}
"""


class DedupScanTest(unittest.TestCase):
    def run_scan(self, files):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel, text in files.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            out = root / "report.json"
            proc = subprocess.run(
                [sys.executable, str(SCAN), "--root", str(root),
                 "--json", str(out)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return json.loads(out.read_text())

    def test_identical_bodies_group_across_files(self):
        r = self.run_scan({
            "libs/fam_a/kernels.cu": KERNEL_A,
            "libs/fam_b/kernels.cu": KERNEL_B,
        })
        self.assertEqual(r["kernel_dup_groups"], 1)
        g = r["kernel_groups"][0]
        self.assertEqual(g["copies"], 2)
        self.assertEqual(g["distinct_files"], 2)
        self.assertGreaterEqual(g["lines"], 6)

    def test_differing_bodies_do_not_group(self):
        r = self.run_scan({
            "libs/fam_a/kernels.cu": KERNEL_A,
            "libs/fam_b/kernels.cu": KERNEL_DIFF,
        })
        self.assertEqual(r["kernel_dup_groups"], 0)
        self.assertEqual(r["kernel_wasted_lines"], 0)

    def test_tiny_kernels_below_min_lines_are_ignored(self):
        r = self.run_scan({
            "libs/fam_a/kernels.cu": KERNEL_TINY,
            "libs/fam_b/kernels.cu": KERNEL_TINY,
        })
        self.assertEqual(r["kernel_dup_groups"], 0)

    def test_same_file_copies_are_not_cross_file_duplication(self):
        r = self.run_scan({
            "libs/fam_a/kernels.cu": KERNEL_A + "\n" + KERNEL_B,
        })
        # two copies in ONE file: a group, but distinct_files == 1
        self.assertEqual(r["kernel_dup_groups"], 1)
        self.assertEqual(r["kernel_groups"][0]["distinct_files"], 1)

    def test_host_functions_group_across_files(self):
        r = self.run_scan({
            "libs/fam_a/load.cpp": HOST_BODY,
            "libs/fam_b/load.cpp": HOST_BODY.replace("UploadBytes", "UploadOther"),
        })
        self.assertEqual(r["host_dup_groups"], 1)
        self.assertGreaterEqual(r["host_wasted_lines"], 12)

    def test_comments_and_whitespace_do_not_split_groups(self):
        noisy = "// noise\n" + KERNEL_A.replace("    ", "\t") + "\n/* end */\n"
        r = self.run_scan({
            "libs/fam_a/kernels.cu": KERNEL_A,
            "libs/fam_b/kernels.cu": noisy,
        })
        self.assertEqual(r["kernel_dup_groups"], 1)


if __name__ == "__main__":
    unittest.main()