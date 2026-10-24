// OJ 3280（Transformer Basic 单头注意力）提交程序，C++17 单文件，与 oj_solution.py 输出一致。
//
// 输入：第一行 n d_model d_k d_v causal；随后依次为 X、W_Q、W_K、W_V（均按行优先）。
// 输出：依次为 Q、K、V、S、A、O 的全部元素（行优先，每行一个矩阵行）。
// S = Q K^T / sqrt(d_k)；因果掩码（causal=1）在 softmax 之前令 S_ij = -inf (j > i)，
// 输出的 S 即为加掩码后的分数（被屏蔽位置输出 -inf），A = softmax(S)，被屏蔽位置恰好为 0。
//
//     g++ -std=c++17 -O2 -o oj_solution oj_solution.cpp && ./oj_solution < input.txt
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

static bool read_matrix(Matrix& m, int r, int c) {
    m.assign(r, std::vector<double>(c));
    for (auto& row : m)
        for (double& x : row)
            if (std::scanf("%lf", &x) != 1) return false;
    return true;
}

static Matrix matmul(const Matrix& a, const Matrix& b) {
    const size_t n = a.size(), k = b.size(), m = b.empty() ? 0 : b[0].size();
    Matrix c(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < k; ++t) s += a[i][t] * b[t][j];
            c[i][j] = s;
        }
    return c;
}

static Matrix transpose(const Matrix& a) {
    Matrix t(a.empty() ? 0 : a[0].size(), std::vector<double>(a.size()));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[i].size(); ++j) t[j][i] = a[i][j];
    return t;
}

static void print_matrix(const Matrix& m) {
    for (const auto& row : m) {
        for (size_t j = 0; j < row.size(); ++j) std::printf(j ? " %.12g" : "%.12g", row[j] + 0.0);  // + 0.0 把 -0.0 规整为 0
        std::printf("\n");
    }
}

int main() {
    int n, d_model, d_k, d_v, causal;
    if (std::scanf("%d %d %d %d %d", &n, &d_model, &d_k, &d_v, &causal) != 5) return 1;
    Matrix X, W_Q, W_K, W_V;
    if (!read_matrix(X, n, d_model) || !read_matrix(W_Q, d_model, d_k) || !read_matrix(W_K, d_model, d_k) ||
        !read_matrix(W_V, d_model, d_v))
        return 1;

    const Matrix Q = matmul(X, W_Q), K = matmul(X, W_K), V = matmul(X, W_V);
    Matrix S = matmul(Q, transpose(K));
    const double scale = std::sqrt(static_cast<double>(d_k));
    for (auto& row : S)
        for (double& x : row) x /= scale;

    const double neg_inf = -std::numeric_limits<double>::infinity();
    if (causal)
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) S[i][j] = neg_inf;

    Matrix A = S;
    for (auto& row : A) {
        const double mx = *std::max_element(row.begin(), row.end());  // 先减去该行最大值，对角线保证 mx 有限
        double total = 0.0;
        for (double& x : row) {
            x = std::exp(x - mx);  // 被屏蔽位置为 -inf，exp 后恰好为 0
            total += x;
        }
        for (double& x : row) x /= total;
    }
    const Matrix O = matmul(A, V);

    for (const Matrix& m : {std::cref(Q), std::cref(K), std::cref(V), std::cref(S), std::cref(A), std::cref(O)})
        print_matrix(m);
    return 0;
}
