"""AI6 Basic：单头缩放点积注意力的前向计算（NumPy）。

    Q = X W_Q，K = X W_K，V = X W_V
    S = Q K^T / sqrt(d_k)
    A = softmax(S + mask)，按行计算
    O = A V

只借用矩阵乘法；投影、缩放、掩码、逐行归一化和加权求和都在这里自行组织。
"""
import numpy as np


def check_shapes(X, W_Q, W_K, W_V):
    """检查输入形状与非空序列，返回转成 float64 的 (X, W_Q, W_K, W_V)。"""
    X, W_Q, W_K, W_V = (np.asarray(a, dtype=np.float64) for a in (X, W_Q, W_K, W_V))
    for name, a in (("X", X), ("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V)):
        if a.ndim != 2:
            raise ValueError(f"{name} 必须是二维矩阵，当前形状 {a.shape}")
        if a.size == 0:
            raise ValueError(f"{name} 不能为空（形状 {a.shape}）：序列和各维度都至少为 1")
    d_model = X.shape[1]
    for name, W in (("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V)):
        if W.shape[0] != d_model:
            raise ValueError(f"{name} 的行数 {W.shape[0]} 必须等于 X 的列数 d_model={d_model}")
    if W_Q.shape[1] != W_K.shape[1]:
        raise ValueError(f"Q、K 的维数必须相同：d_k(W_Q)={W_Q.shape[1]}，d_k(W_K)={W_K.shape[1]}")
    return X, W_Q, W_K, W_V


def project(X, W_Q, W_K, W_V):
    """线性投影：X (n×d_model) → Q (n×d_k)、K (n×d_k)、V (n×d_v)。"""
    X, W_Q, W_K, W_V = check_shapes(X, W_Q, W_K, W_V)
    return X @ W_Q, X @ W_K, X @ W_V


def scaled_scores(Q, K):
    """S = Q K^T / sqrt(d_k)，形状 n×n。"""
    return Q @ K.T / np.sqrt(Q.shape[1])


def causal_mask(n):
    """加性掩码：允许位置为 0，严格上三角（j > i）为 -inf，保留对角线。"""
    mask = np.zeros((n, n))
    mask[np.triu_indices(n, k=1)] = -np.inf
    return mask


def softmax_rows(S):
    """数值稳定的逐行 softmax：先减每行最大值，再求指数并按行归一化。"""
    shifted = S - S.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def attention_forward(X, W_Q, W_K, W_V, causal=False):
    """单头注意力前向，返回 dict：Q, K, V, S（未加掩码的分数）, A, O。

    掩码在 softmax 之前加到 S 上；因此不会出现“先 softmax 再置零”导致行和不为 1。
    """
    Q, K, V = project(X, W_Q, W_K, W_V)
    S = scaled_scores(Q, K)
    masked = S + causal_mask(S.shape[0]) if causal else S
    A = softmax_rows(masked)
    return {"Q": Q, "K": K, "V": V, "S": S, "A": A, "O": A @ V}


def check_properties(result, causal, tol=1e-6):
    """检查行和、非负性、被屏蔽位置与各矩阵形状，返回 [(名称, 是否通过), ...]。"""
    Q, K, V, S, A, O = (result[k] for k in ("Q", "K", "V", "S", "A", "O"))
    n, d_k, d_v = Q.shape[0], Q.shape[1], V.shape[1]
    checks = [
        ("形状：Q、K 为 n×d_k，V 为 n×d_v", Q.shape == K.shape == (n, d_k) and V.shape == (n, d_v)),
        ("形状：S、A 为 n×n，O 为 n×d_v", S.shape == A.shape == (n, n) and O.shape == (n, d_v)),
        ("行和为 1", bool(np.allclose(A.sum(axis=1), 1.0, atol=tol))),
        ("权重非负", bool((A >= 0).all())),
    ]
    if causal:
        upper = A[np.triu_indices(n, k=1)]
        checks.append(("被屏蔽的未来位置权重为 0", bool((upper == 0).all())))
        checks.append(("对角线保留（每行至少一个可见位置）", bool((np.diag(A) > 0).all())))
    return checks


def max_prefix_change(X, W_Q, W_K, W_V, causal, first_changed, new_rows):
    """把 X 中从 first_changed 起的行替换成 new_rows，返回前 first_changed 行输出的最大绝对变化。"""
    X = np.array(X, dtype=np.float64)
    before = attention_forward(X, W_Q, W_K, W_V, causal)["O"]
    X2 = X.copy()
    X2[first_changed:] = new_rows
    after = attention_forward(X2, W_Q, W_K, W_V, causal)["O"]
    return float(np.abs(before[:first_changed] - after[:first_changed]).max())
