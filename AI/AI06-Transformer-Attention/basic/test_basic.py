"""Basic 验收测试：python -m unittest -v test_basic   （在 basic/ 目录下运行）"""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from attention_basic import (attention_forward, causal_mask, check_properties, max_prefix_change,
                             softmax_rows)

HERE = Path(__file__).resolve().parent
TOL = 1e-6
I2 = np.eye(2)


def rand_case(rng, n, d_model, d_k, d_v):
    return (rng.normal(size=(n, d_model)), rng.normal(size=(d_model, d_k)),
            rng.normal(size=(d_model, d_k)), rng.normal(size=(d_model, d_v)))


class HandCheck(unittest.TestCase):
    """Q5：X=W_Q=W_K=W_V=I2，d_k=2 的手算结果。"""

    def test_unmasked(self):
        r = attention_forward(I2, I2, I2, I2)
        a = np.exp(1 / np.sqrt(2)) / (np.exp(1 / np.sqrt(2)) + 1)  # 手算：e^{1/√2}/(e^{1/√2}+1)
        np.testing.assert_allclose(r["A"], [[0.669762, 0.330238], [0.330238, 0.669762]], atol=TOL)
        np.testing.assert_allclose(r["A"], [[a, 1 - a], [1 - a, a]], atol=1e-12)
        np.testing.assert_allclose(r["O"], r["A"], atol=1e-12)  # V 是单位阵，O = A

    def test_causal(self):
        r = attention_forward(I2, I2, I2, I2, causal=True)
        np.testing.assert_allclose(r["A"], [[1.0, 0.0], [0.330238, 0.669762]], atol=TOL)
        np.testing.assert_allclose(r["O"], r["A"], atol=1e-12)

    def test_scores_are_unmasked(self):
        r = attention_forward(I2, I2, I2, I2, causal=True)
        np.testing.assert_allclose(r["S"], np.eye(2) / np.sqrt(2), atol=1e-12)


class Properties(unittest.TestCase):
    def test_random_shapes(self):
        rng = np.random.default_rng(0)
        for n, dm, dk, dv in [(1, 1, 1, 1), (2, 3, 2, 4), (5, 8, 4, 3), (16, 32, 32, 32), (32, 7, 5, 1)]:
            for causal in (False, True):
                r = attention_forward(*rand_case(rng, n, dm, dk, dv), causal=causal)
                bad = [name for name, ok in check_properties(r, causal) if not ok]
                self.assertEqual(bad, [], f"n={n} causal={causal}")

    def test_single_position_output_equals_v(self):
        rng = np.random.default_rng(1)
        for causal in (False, True):
            X, Wq, Wk, Wv = rand_case(rng, 1, 5, 3, 4)
            r = attention_forward(X, Wq, Wk, Wv, causal=causal)
            np.testing.assert_allclose(r["A"], [[1.0]], atol=1e-12)
            np.testing.assert_allclose(r["O"], r["V"], atol=1e-12)

    def test_mask_is_strict_upper_triangle(self):
        m = causal_mask(4)
        self.assertTrue(np.isneginf(m[np.triu_indices(4, k=1)]).all())
        self.assertTrue((m[np.tril_indices(4)] == 0).all())  # 含对角线


class Causality(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.args = rand_case(self.rng, 6, 8, 4, 5)

    def test_future_change_does_not_affect_past_when_causal(self):
        X, Wq, Wk, Wv = self.args
        for k in range(1, 6):
            diff = max_prefix_change(X, Wq, Wk, Wv, True, k, self.rng.normal(size=(6 - k, 8)) * 10)
            self.assertLessEqual(diff, 1e-12, f"修改第 {k} 行及之后")

    def test_future_change_affects_past_without_mask(self):
        X, Wq, Wk, Wv = self.args
        diff = max_prefix_change(X, Wq, Wk, Wv, False, 3, self.rng.normal(size=(3, 8)) * 10)
        self.assertGreater(diff, 1e-3)

    def test_permutation_equivariance_without_mask(self):
        X, Wq, Wk, Wv = self.args
        order = self.rng.permutation(6)
        O1 = attention_forward(X, Wq, Wk, Wv)["O"]
        O2 = attention_forward(X[order], Wq, Wk, Wv)["O"]
        np.testing.assert_allclose(O1[order], O2, atol=1e-12)


class Numerics(unittest.TestCase):
    def test_softmax_stable_for_large_scores(self):
        S = np.array([[1000.0, 1001.0], [-1000.0, -1001.0]])
        A = softmax_rows(S)
        self.assertTrue(np.isfinite(A).all())
        np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(A, softmax_rows(S - 5), atol=1e-12)  # 整体平移不变

    def test_zeroing_after_softmax_breaks_row_sum(self):
        A = softmax_rows(np.eye(2) / np.sqrt(2))
        wrong = np.tril(A)  # 先 softmax 再把未来权重置零
        self.assertNotAlmostEqual(wrong.sum(axis=1)[0], 1.0, places=3)
        right = attention_forward(I2, I2, I2, I2, causal=True)["A"]
        np.testing.assert_allclose(right.sum(axis=1), 1.0, atol=1e-12)


class InputValidation(unittest.TestCase):
    def test_bad_shapes_raise(self):
        ok = np.ones((2, 3)), np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 4))
        attention_forward(*ok)  # 合法输入不应报错
        bad = [
            (np.ones((0, 3)),) + ok[1:],                               # 空序列
            (np.ones((2, 3)), np.ones((4, 2)), ok[2], ok[3]),          # d_model 对不上
            (ok[0], ok[1], np.ones((3, 5)), ok[3]),                    # Q、K 维数不同
            (np.ones(3),) + ok[1:],                                    # 非二维
        ]
        for args in bad:
            with self.assertRaises(ValueError):
                attention_forward(*args)


def oj_expected(r, causal):
    """OJ 要求输出 Q、K、V、S、A、O；因果时 S 为加掩码后的分数（j>i 处为 -inf）。"""
    S = r["S"].copy()
    if causal:
        S[np.triu_indices_from(S, k=1)] = -np.inf
    return np.concatenate([M.ravel() for M in (r["Q"], r["K"], r["V"], S, r["A"], r["O"])])


def oj_close(actual, expected):
    """OJ 容差：绝对 1e-6 + 相对 2e-5；-inf 必须位置相同。"""
    finite = np.isfinite(expected)
    return ((np.isinf(actual) == ~finite).all() and (actual[~finite] == expected[~finite]).all()
            and (np.abs(actual[finite] - expected[finite]) <= 1e-6 + 2e-5 * np.abs(expected[finite])).all())


def run_oj(inp):
    p = subprocess.run([sys.executable, str(HERE / "oj_solution.py")], input=inp, capture_output=True,
                       text=True, check=True)
    return np.array([float(x) for x in p.stdout.split()])


def oj_input(n, dm, dk, dv, causal, X, Wq, Wk, Wv):
    head = f"{n} {dm} {dk} {dv} {int(causal)}\n"
    return head + "\n".join(" ".join(repr(float(v)) for v in M.ravel()) for M in (X, Wq, Wk, Wv)) + "\n"


class OnlineJudgeProgram(unittest.TestCase):
    def test_q5_example(self):
        inp = "2 2 2 2 0\n1 0 0 1\n1 0 0 1\n1 0 0 1\n1 0 0 1\n"
        out = run_oj(inp)
        self.assertEqual(out.size, 24)  # Q、K、V、S、A、O 各 2×2
        A = out[16:20].reshape(2, 2)
        np.testing.assert_allclose(A, [[0.669762, 0.330238], [0.330238, 0.669762]], atol=TOL)

    def test_q5_example_causal(self):
        out = run_oj("2 2 2 2 1\n1 0 0 1\n1 0 0 1\n1 0 0 1\n1 0 0 1\n")
        np.testing.assert_allclose(out[16:20].reshape(2, 2), [[1, 0], [0.330238, 0.669762]], atol=TOL)
        self.assertEqual(out[13], -np.inf)  # 因果时输出的 S 在 j>i 处为 -inf
        self.assertTrue(np.isfinite(np.delete(out, 13)).all())

    def test_matches_numpy_on_random_inputs(self):
        rng = np.random.default_rng(3)
        for n, dm, dk, dv in [(1, 1, 1, 1), (3, 4, 3, 2), (7, 5, 6, 4), (32, 32, 32, 32), (10, 32, 1, 32)]:
            for causal in (False, True):
                X, Wq, Wk, Wv = rand_case(rng, n, dm, dk, dv)
                r = attention_forward(X, Wq, Wk, Wv, causal)
                expected = oj_expected(r, causal)
                actual = run_oj(oj_input(n, dm, dk, dv, causal, X, Wq, Wk, Wv))
                self.assertEqual(actual.shape, expected.shape)
                # 与 OJ 说明一致的容差：绝对 1e-6 + 相对 2e-5
                self.assertTrue(oj_close(actual, expected),
                                f"n={n} dm={dm} dk={dk} dv={dv} causal={causal}")

    def test_large_scores_do_not_overflow(self):
        rng = np.random.default_rng(4)
        X, Wq, Wk, Wv = rand_case(rng, 4, 4, 4, 4)
        X = X * 300
        out = run_oj(oj_input(4, 4, 4, 4, False, X, Wq, Wk, Wv))
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
