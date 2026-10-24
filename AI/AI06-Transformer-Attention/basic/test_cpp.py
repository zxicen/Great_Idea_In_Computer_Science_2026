"""C++ 副本的验收：python -m unittest -v test_cpp   （在 basic/ 目录下运行，需要 g++）

- attention_basic.cpp：运行内置自检，退出码必须为 0；
- oj_solution.cpp：与 oj_solution.py、NumPy 实现在随机输入上逐元素对照。
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from attention_basic import attention_forward

HERE = Path(__file__).resolve().parent
CXX = shutil.which("g++") or shutil.which("clang++")


def compile_cpp(src, out):
    subprocess.run([CXX, "-std=c++17", "-O2", "-Wall", "-Wextra", "-o", str(out), str(HERE / src)],
                   check=True, capture_output=True, text=True)


def oj_input(n, dm, dk, dv, causal, X, Wq, Wk, Wv):
    head = f"{n} {dm} {dk} {dv} {int(causal)}\n"
    return head + "\n".join(" ".join(repr(float(v)) for v in M.ravel()) for M in (X, Wq, Wk, Wv)) + "\n"


@unittest.skipIf(CXX is None, "未找到 C++ 编译器")
class CppCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.selftest = Path(cls.tmp.name) / "attention_basic"
        cls.oj = Path(cls.tmp.name) / "oj_solution"
        compile_cpp("attention_basic.cpp", cls.selftest)
        compile_cpp("oj_solution.cpp", cls.oj)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_oj(self, inp):
        p = subprocess.run([str(self.oj)], input=inp, capture_output=True, text=True, check=True)
        return p.stdout

    def test_selftest_passes(self):
        p = subprocess.run([str(self.selftest)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertNotIn("[FAIL]", p.stdout)

    def test_sample_matches_python_program(self):
        for causal in (0, 1):
            inp = f"2 2 2 2 {causal}\n1 0 0 1\n1 0 0 1\n1 0 0 1\n1 0 0 1\n"
            py = subprocess.run(["python3", str(HERE / "oj_solution.py")], input=inp, capture_output=True,
                                text=True, check=True).stdout
            cpp = np.array(self.run_oj(inp).split(), dtype=float)
            np.testing.assert_allclose(cpp, np.array(py.split(), dtype=float), rtol=1e-12, atol=1e-12)
            self.assertEqual(cpp.size, 24)

    def test_matches_numpy_on_random_inputs(self):
        rng = np.random.default_rng(11)
        for n, dm, dk, dv in [(1, 1, 1, 1), (3, 4, 3, 2), (7, 5, 6, 4), (32, 32, 32, 32), (10, 32, 1, 32)]:
            for causal in (False, True):
                X, Wq, Wk, Wv = (rng.normal(size=s) for s in ((n, dm), (dm, dk), (dm, dk), (dm, dv)))
                r = attention_forward(X, Wq, Wk, Wv, causal)
                expected = np.concatenate([r[k].ravel() for k in ("Q", "K", "V", "S", "A", "O")])
                actual = np.array(self.run_oj(oj_input(n, dm, dk, dv, causal, X, Wq, Wk, Wv)).split(), dtype=float)
                self.assertEqual(actual.shape, expected.shape)
                # 与 OJ 说明一致的容差：绝对 1e-6 + 相对 2e-5
                self.assertTrue((np.abs(actual - expected) <= 1e-6 + 2e-5 * np.abs(expected)).all(),
                                f"n={n} dm={dm} dk={dk} dv={dv} causal={causal}")

    def test_large_scores_do_not_overflow(self):
        rng = np.random.default_rng(4)
        X, Wq, Wk, Wv = (rng.normal(size=(4, 4)) for _ in range(4))
        out = np.array(self.run_oj(oj_input(4, 4, 4, 4, False, X * 300, Wq, Wk, Wv)).split(), dtype=float)
        self.assertTrue(np.isfinite(out).all())

    def test_output_format_one_row_per_line(self):
        out = self.run_oj("2 2 2 2 1\n1 0 0 1\n1 0 0 1\n1 0 0 1\n1 0 0 1\n")
        lines = out.strip().split("\n")
        self.assertEqual(len(lines), 12)  # 6 个矩阵 × 2 行
        self.assertTrue(all(len(line.split()) == 2 for line in lines))
        self.assertNotIn("-0 ", out + " ")  # -0.0 已规整为 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
