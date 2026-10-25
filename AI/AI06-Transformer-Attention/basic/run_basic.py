"""Basic 实验：打印矩阵日志、做核对实验、画权重热力图。

    python -m pip install -r requirements.txt
    python run_basic.py

输出：output/matrix_log.txt（矩阵日志）、output/attention_heatmap.png（权重热力图）。
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from attention_basic import attention_forward, check_properties, max_prefix_change, scaled_scores, softmax_rows  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
TOL = 1e-6
np.set_printoptions(precision=6, suppress=True, linewidth=120)


def show(name, M):
    print(f"{name}  shape = {M.shape}")
    print(np.array2string(M, precision=6, suppress_small=True))
    print()


def show_all(result):
    for name in ("Q", "K", "V", "S", "A", "O"):
        show(name, result[name])
    print("每行权重之和:", np.round(result["A"].sum(axis=1), 6))
    print()


def report_checks(result, causal):
    for name, ok in check_properties(result, causal, TOL):
        print(f"  [{'通过' if ok else '失败'}] {name}")
    print()


def experiments():
    results = {}
    I2 = np.eye(2)

    print("=" * 70)
    print("实验 1：Q5 手算例子（X=W_Q=W_K=W_V=I2，d_k=2）")
    print("=" * 70)
    expected = {
        False: np.array([[0.669762, 0.330238], [0.330238, 0.669762]]),
        True: np.array([[1.0, 0.0], [0.330238, 0.669762]]),
    }
    for causal in (False, True):
        print(f"--- {'因果掩码' if causal else '无掩码'} ---")
        r = attention_forward(I2, I2, I2, I2, causal=causal)
        show_all(r)
        ok = np.allclose(r["A"], expected[causal], atol=TOL)
        print(f"  [{'通过' if ok else '失败'}] A 与手算值一致（容差 {TOL}）")
        print(f"  [{'通过' if np.allclose(r['O'], r['A']) else '失败'}] V 为单位阵，O = A")
        report_checks(r, causal)
        results[("q5", causal)] = r

    print("=" * 70)
    print("实验 2：三个位置的小例子（固定随机种子 0；n=3，d_model=4，d_k=3，d_v=2）")
    print("=" * 70)
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(3, 4)), 2)
    W_Q = np.round(rng.normal(size=(4, 3)), 2)
    W_K = np.round(rng.normal(size=(4, 3)), 2)
    W_V = np.round(rng.normal(size=(4, 2)), 2)
    show("X", X)
    show("W_Q", W_Q)
    show("W_K", W_K)
    show("W_V", W_V)
    for causal in (False, True):
        print(f"--- {'因果掩码' if causal else '无掩码'} ---")
        r = attention_forward(X, W_Q, W_K, W_V, causal=causal)
        show_all(r)
        report_checks(r, causal)
        results[("three", causal)] = r
    three = (X, W_Q, W_K, W_V)

    print("=" * 70)
    print("实验 3：只有一个位置时 O = V")
    print("=" * 70)
    X1 = np.array([[2.0, 3.0]])
    for causal in (False, True):
        r = attention_forward(X1, I2, I2, I2, causal=causal)
        print(f"causal={causal}: A = {r['A'].tolist()}, O = {r['O'].tolist()}, V = {r['V'].tolist()}")
        print(f"  [{'通过' if np.allclose(r['O'], r['V'], atol=TOL) else '失败'}] O = V")
    print()

    print("=" * 70)
    print("实验 4：只修改未来输入，较早位置的输出是否变化（用实验 2 的 X，改第 3 行）")
    print("=" * 70)
    new_row = np.array([[9.0, -9.0, 9.0, -9.0]])
    for causal in (True, False):
        diff = max_prefix_change(X, W_Q, W_K, W_V, causal, 2, new_row)
        verdict = "前两行输出不变" if diff <= TOL else "前两行输出发生变化"
        expect = "（符合预期）" if (causal and diff <= TOL) or (not causal and diff > TOL) else "（不符合预期！）"
        print(f"  causal={causal}: 前两行输出最大绝对变化 = {diff:.3e} → {verdict} {expect}")
    print()

    print("=" * 70)
    print("实验 5：无掩码时打乱输入顺序，输出按同样方式重排（用实验 2 的数据）")
    print("=" * 70)
    order = [2, 0, 1]
    O1 = attention_forward(X, W_Q, W_K, W_V)["O"]
    O2 = attention_forward(X[order], W_Q, W_K, W_V)["O"]
    print(f"  最大绝对误差 |O1[order] - O2| = {np.abs(O1[order] - O2).max():.3e}")
    print(f"  [{'通过' if np.allclose(O1[order], O2, atol=TOL) else '失败'}] 输出按同样方式重排")
    print()

    print("=" * 70)
    print("实验 6：掩码必须在 softmax 之前（Q5 例子）")
    print("=" * 70)
    A_unmasked = results[("q5", False)]["A"]
    wrong = np.tril(A_unmasked)
    show("先 softmax 再把未来权重置零（错误做法）", wrong)
    print("  行和:", np.round(wrong.sum(axis=1), 6), "← 第一行不再是 1")
    print("  正确做法（softmax 前加 -inf）的行和:", np.round(results[("q5", True)]["A"].sum(axis=1), 6))
    print()

    print("=" * 70)
    print("实验 7：为什么要除以 sqrt(d_k)（n=8，Q、K 元素独立服从 N(0,1)，每个 d_k 重复 200 次，种子 0）")
    print("=" * 70)
    scaling_experiment()
    return results, three


def scaling_experiment(n=8, trials=200, dims=(1, 4, 16, 64, 256, 1024)):
    rng = np.random.default_rng(0)
    print(f"{'d_k':>5} | {'S 标准差(不缩放)':>16} | {'S 标准差(缩放)':>14} | "
          f"{'行最大权重(不缩放)':>18} | {'行最大权重(缩放)':>16} | {'最大权重>0.99 的行占比(不缩放)':>24}")
    for d in dims:
        raw_std, sc_std, raw_max, sc_max, raw_sat = [], [], [], [], []
        for _ in range(trials):
            Q = rng.normal(size=(n, d))
            K = rng.normal(size=(n, d))
            raw = Q @ K.T
            scaled = scaled_scores(Q, K)
            a_raw, a_sc = softmax_rows(raw), softmax_rows(scaled)
            raw_std.append(raw.std())
            sc_std.append(scaled.std())
            raw_max.append(a_raw.max(axis=1).mean())
            sc_max.append(a_sc.max(axis=1).mean())
            raw_sat.append((a_raw.max(axis=1) > 0.99).mean())
        print(f"{d:>5} | {np.mean(raw_std):>16.3f} | {np.mean(sc_std):>14.3f} | "
              f"{np.mean(raw_max):>18.3f} | {np.mean(sc_max):>16.3f} | {np.mean(raw_sat):>24.3f}")
    print("  缩放后 S 的标准差始终约为 1，权重分布与 d_k 无关；不缩放时标准差约为 sqrt(d_k)，"
          "softmax 趋于只看最大值（one-hot），权重对输入的变化几乎不敏感。")
    print()


def draw(ax, M, title, causal=None):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    n = M.shape[0]
    ax.set_xticks(range(n), [f"j={j + 1}" for j in range(n)])
    ax.set_yticks(range(n), [f"i={i + 1}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.6 else "black", fontsize=9)
    return im


def heatmaps(results, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panels = [
        (("q5", False), "Q5: no mask"), (("q5", True), "Q5: causal mask"),
        (("three", False), "3 positions: no mask"), (("three", True), "3 positions: causal mask"),
    ]
    im = None
    for ax, (key, title) in zip(axes.ravel(), panels):
        im = draw(ax, results[key]["A"], title)
        ax.set_xlabel("key position j")
        ax.set_ylabel("query position i")
    fig.colorbar(im, ax=axes, shrink=0.8, label="attention weight")
    fig.suptitle("Single-head scaled dot-product attention weights A (rows sum to 1)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        results, _ = experiments()
    log = buf.getvalue()
    (OUT / "matrix_log.txt").write_text(log, encoding="utf-8")
    sys.stdout.write(log)
    heatmaps(results, OUT / "attention_heatmap.png")
    print(f"已写入 {OUT / 'matrix_log.txt'} 与 {OUT / 'attention_heatmap.png'}")
    if "失败" in log or "不符合预期" in log:
        sys.exit(1)


if __name__ == "__main__":
    main()
