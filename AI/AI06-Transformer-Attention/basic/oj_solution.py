"""OJ 3280（Transformer Basic 单头注意力）提交程序：仅用标准库，从标准输入读、向标准输出写。

输入：第一行 n d_model d_k d_v causal；随后依次为 X、W_Q、W_K、W_V（均按行优先）。
输出：依次为 Q、K、V、S、A、O 的全部元素（行优先，每行一个矩阵行）。
S = Q K^T / sqrt(d_k) 为未加掩码的缩放分数；A = softmax(S + mask)，掩码在 softmax 之前加上。
"""
import math
import sys


def read_matrix(vals, pos, rows, cols):
    m = [vals[pos + i * cols: pos + (i + 1) * cols] for i in range(rows)]
    return m, pos + rows * cols


def matmul(a, b):
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def softmax_row(row):
    # 先减去该行最大值；被屏蔽位置为 -inf，exp 后恰好为 0
    mx = max(row)
    e = [math.exp(x - mx) for x in row]
    total = sum(e)
    return [x / total for x in e]


def main():
    data = sys.stdin.read().split()
    n, d_model, d_k, d_v, causal = (int(x) for x in data[:5])
    vals = [float(x) for x in data[5:]]
    pos = 0
    X, pos = read_matrix(vals, pos, n, d_model)
    W_Q, pos = read_matrix(vals, pos, d_model, d_k)
    W_K, pos = read_matrix(vals, pos, d_model, d_k)
    W_V, pos = read_matrix(vals, pos, d_model, d_v)

    Q, K, V = matmul(X, W_Q), matmul(X, W_K), matmul(X, W_V)
    scale = math.sqrt(d_k)
    S = [[x / scale for x in row] for row in matmul(Q, [list(c) for c in zip(*K)])]

    neg_inf = float("-inf")
    A = []
    for i in range(n):
        masked = [S[i][j] + (neg_inf if causal and j > i else 0.0) for j in range(n)]
        A.append(softmax_row(masked))
    O = matmul(A, V)

    out = []
    for M in (Q, K, V, S, A, O):
        for row in M:
            out.append(" ".join(format(x + 0.0, ".12g") for x in row))  # x + 0.0 把 -0.0 规整为 0
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
