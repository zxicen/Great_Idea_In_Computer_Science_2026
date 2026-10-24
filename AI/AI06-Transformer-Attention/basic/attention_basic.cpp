// AI6 Basic：单头缩放点积注意力的前向计算（C++17），与 attention_basic.py 一一对应。
//
//     Q = X W_Q，K = X W_K，V = X W_V
//     S = Q K^T / sqrt(d_k)
//     A = softmax(S + mask)，按行计算
//     O = A V
//
// 编译并运行自检：
//     g++ -std=c++17 -O2 -Wall -Wextra -o attention_basic attention_basic.cpp && ./attention_basic
// 所有检查通过时退出码为 0，否则为 1。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

struct Result {
    Matrix Q, K, V, S, A, O;  // S 为未加掩码的分数
};

static size_t rows(const Matrix& m) { return m.size(); }
static size_t cols(const Matrix& m) { return m.empty() ? 0 : m[0].size(); }

// 检查输入为非空的规整二维矩阵，且形状互相匹配；不合法时抛出 std::invalid_argument。
void check_shapes(const Matrix& X, const Matrix& W_Q, const Matrix& W_K, const Matrix& W_V) {
    const std::pair<const char*, const Matrix*> named[] = {{"X", &X}, {"W_Q", &W_Q}, {"W_K", &W_K}, {"W_V", &W_V}};
    for (const auto& [name, m] : named) {
        if (m->empty() || m->front().empty())
            throw std::invalid_argument(std::string(name) + " 不能为空：序列和各维度都至少为 1");
        for (const auto& row : *m)
            if (row.size() != m->front().size())
                throw std::invalid_argument(std::string(name) + " 必须是规整的二维矩阵");
    }
    const size_t d_model = cols(X);
    for (const auto& [name, m] : {named[1], named[2], named[3]})
        if (rows(*m) != d_model)
            throw std::invalid_argument(std::string(name) + " 的行数必须等于 X 的列数 d_model");
    if (cols(W_Q) != cols(W_K)) throw std::invalid_argument("Q、K 的维数必须相同：d_k(W_Q) != d_k(W_K)");
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    const size_t n = rows(a), k = cols(a), m = cols(b);
    Matrix c(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < k; ++t) s += a[i][t] * b[t][j];
            c[i][j] = s;
        }
    return c;
}

Matrix transpose(const Matrix& a) {
    Matrix t(cols(a), std::vector<double>(rows(a)));
    for (size_t i = 0; i < rows(a); ++i)
        for (size_t j = 0; j < cols(a); ++j) t[j][i] = a[i][j];
    return t;
}

// 线性投影：X (n×d_model) → Q (n×d_k)、K (n×d_k)、V (n×d_v)。
void project(const Matrix& X, const Matrix& W_Q, const Matrix& W_K, const Matrix& W_V, Matrix& Q, Matrix& K,
             Matrix& V) {
    check_shapes(X, W_Q, W_K, W_V);
    Q = matmul(X, W_Q);
    K = matmul(X, W_K);
    V = matmul(X, W_V);
}

// S = Q K^T / sqrt(d_k)，形状 n×n。
Matrix scaled_scores(const Matrix& Q, const Matrix& K) {
    Matrix S = matmul(Q, transpose(K));
    const double scale = std::sqrt(static_cast<double>(cols(Q)));
    for (auto& row : S)
        for (double& x : row) x /= scale;
    return S;
}

// 加性掩码：允许位置为 0，严格上三角（j > i）为 -inf，保留对角线。
Matrix causal_mask(size_t n) {
    Matrix mask(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) mask[i][j] = NEG_INF;
    return mask;
}

// 数值稳定的逐行 softmax：先减每行最大值，再求指数并按行归一化。
Matrix softmax_rows(const Matrix& S) {
    Matrix A = S;
    for (auto& row : A) {
        const double mx = *std::max_element(row.begin(), row.end());
        double total = 0.0;
        for (double& x : row) {
            x = std::exp(x - mx);  // 被屏蔽位置为 -inf，exp 后恰好为 0
            total += x;
        }
        for (double& x : row) x /= total;
    }
    return A;
}

// 单头注意力前向。掩码在 softmax 之前加到 S 上，因此不会出现“先 softmax 再置零”导致行和不为 1。
Result attention_forward(const Matrix& X, const Matrix& W_Q, const Matrix& W_K, const Matrix& W_V,
                         bool causal = false) {
    Result r;
    project(X, W_Q, W_K, W_V, r.Q, r.K, r.V);
    r.S = scaled_scores(r.Q, r.K);
    Matrix masked = r.S;
    if (causal) {
        const Matrix mask = causal_mask(rows(masked));
        for (size_t i = 0; i < rows(masked); ++i)
            for (size_t j = 0; j < cols(masked); ++j) masked[i][j] += mask[i][j];
    }
    r.A = softmax_rows(masked);
    r.O = matmul(r.A, r.V);
    return r;
}

// 检查行和、非负性、被屏蔽位置与各矩阵形状，返回 (名称, 是否通过) 列表。
std::vector<std::pair<std::string, bool>> check_properties(const Result& r, bool causal, double tol = 1e-6) {
    const size_t n = rows(r.Q), d_k = cols(r.Q), d_v = cols(r.V);
    std::vector<std::pair<std::string, bool>> checks;
    checks.emplace_back("形状：Q、K 为 n×d_k，V 为 n×d_v",
                        rows(r.K) == n && cols(r.K) == d_k && rows(r.V) == n);
    checks.emplace_back("形状：S、A 为 n×n，O 为 n×d_v",
                        rows(r.S) == n && cols(r.S) == n && rows(r.A) == n && cols(r.A) == n &&
                            rows(r.O) == n && cols(r.O) == d_v);
    bool sums = true, nonneg = true;
    for (const auto& row : r.A) {
        sums = sums && std::fabs(std::accumulate(row.begin(), row.end(), 0.0) - 1.0) <= tol;
        for (double x : row) nonneg = nonneg && x >= 0.0;
    }
    checks.emplace_back("行和为 1", sums);
    checks.emplace_back("权重非负", nonneg);
    if (causal) {
        bool zero = true, diag = true;
        for (size_t i = 0; i < n; ++i) {
            diag = diag && r.A[i][i] > 0.0;
            for (size_t j = i + 1; j < n; ++j) zero = zero && r.A[i][j] == 0.0;
        }
        checks.emplace_back("被屏蔽的未来位置权重为 0", zero);
        checks.emplace_back("对角线保留（每行至少一个可见位置）", diag);
    }
    return checks;
}

// 把 X 中从 first_changed 起的行替换成 new_rows，返回前 first_changed 行输出的最大绝对变化。
double max_prefix_change(Matrix X, const Matrix& W_Q, const Matrix& W_K, const Matrix& W_V, bool causal,
                         size_t first_changed, const Matrix& new_rows) {
    const Matrix before = attention_forward(X, W_Q, W_K, W_V, causal).O;
    for (size_t i = first_changed; i < rows(X); ++i) X[i] = new_rows[i - first_changed];
    const Matrix after = attention_forward(X, W_Q, W_K, W_V, causal).O;
    double worst = 0.0;
    for (size_t i = 0; i < first_changed; ++i)
        for (size_t j = 0; j < cols(before); ++j) worst = std::max(worst, std::fabs(before[i][j] - after[i][j]));
    return worst;
}

// ---------------------------------------------------------------- 自检

static int failures = 0;

static void expect(bool ok, const std::string& name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) ++failures;
}

static bool close(const Matrix& a, const Matrix& b, double tol) {
    if (rows(a) != rows(b) || cols(a) != cols(b)) return false;
    for (size_t i = 0; i < rows(a); ++i)
        for (size_t j = 0; j < cols(a); ++j)
            if (!(std::fabs(a[i][j] - b[i][j]) <= tol)) return false;
    return true;
}

static Matrix random_matrix(std::mt19937& rng, size_t r, size_t c, double scale = 1.0) {
    std::normal_distribution<double> dist(0.0, scale);
    Matrix m(r, std::vector<double>(c));
    for (auto& row : m)
        for (double& x : row) x = dist(rng);
    return m;
}

static bool all_pass(const std::vector<std::pair<std::string, bool>>& checks) {
    return std::all_of(checks.begin(), checks.end(), [](const auto& c) { return c.second; });
}

int main() {
    const Matrix I2 = {{1, 0}, {0, 1}};
    const double a = std::exp(1 / std::sqrt(2.0)) / (std::exp(1 / std::sqrt(2.0)) + 1);  // 手算：e^{1/√2}/(e^{1/√2}+1)

    // Q5：X = W_Q = W_K = W_V = I2，d_k = 2
    const Result plain = attention_forward(I2, I2, I2, I2, false);
    expect(close(plain.A, {{0.669762, 0.330238}, {0.330238, 0.669762}}, 1e-6), "Q5 无掩码：A 与手算一致");
    expect(close(plain.A, {{a, 1 - a}, {1 - a, a}}, 1e-12), "Q5 无掩码：A 等于解析值 e^{1/√2}/(e^{1/√2}+1)");
    expect(close(plain.O, plain.A, 1e-12), "Q5 无掩码：V 为单位阵，O = A");

    const Result masked = attention_forward(I2, I2, I2, I2, true);
    expect(close(masked.A, {{1.0, 0.0}, {0.330238, 0.669762}}, 1e-6), "Q5 因果掩码：A 与手算一致");
    expect(close(masked.O, masked.A, 1e-12), "Q5 因果掩码：O = A");
    expect(close(masked.S, {{1 / std::sqrt(2.0), 0}, {0, 1 / std::sqrt(2.0)}}, 1e-12), "S 为未加掩码的缩放分数");

    // 单个位置：softmax 只有一项，权重为 1，O = V
    const Matrix X1 = {{0.3, -1.2, 0.8}};
    std::mt19937 rng(20261024);
    const Matrix Wq1 = random_matrix(rng, 3, 2), Wk1 = random_matrix(rng, 3, 2), Wv1 = random_matrix(rng, 3, 4);
    const Result one = attention_forward(X1, Wq1, Wk1, Wv1, false);
    expect(one.A[0][0] == 1.0 && close(one.O, one.V, 1e-12), "单个位置：A = [1]，O = V");

    // 随机输入的基本性质（含因果/非因果，序列长度 1…32）
    bool props_ok = true;
    for (size_t n : {1u, 2u, 3u, 7u, 16u, 32u})
        for (bool causal : {false, true}) {
            const Matrix X = random_matrix(rng, n, 5), Wq = random_matrix(rng, 5, 4), Wk = random_matrix(rng, 5, 4),
                         Wv = random_matrix(rng, 5, 3);
            props_ok = props_ok && all_pass(check_properties(attention_forward(X, Wq, Wk, Wv, causal), causal));
        }
    expect(props_ok, "随机输入：形状、行和为 1、非负、被屏蔽位置为 0");

    // 因果性：只改未来的输入，较早位置的输出完全不变；无掩码时会变
    const Matrix X6 = random_matrix(rng, 6, 4), Wq = random_matrix(rng, 4, 3), Wk = random_matrix(rng, 4, 3),
                 Wv = random_matrix(rng, 4, 5), fresh = random_matrix(rng, 3, 4, 5.0);
    expect(max_prefix_change(X6, Wq, Wk, Wv, true, 3, fresh) == 0.0, "因果掩码：修改未来输入，前缀输出不变");
    expect(max_prefix_change(X6, Wq, Wk, Wv, false, 3, fresh) > 1e-3, "无掩码：修改未来输入，前缀输出会变化");

    // 置换等变性（无掩码）：打乱输入行，输出同样打乱
    std::vector<size_t> perm(6);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    Matrix Xp(6);
    for (size_t i = 0; i < 6; ++i) Xp[i] = X6[perm[i]];
    const Matrix O = attention_forward(X6, Wq, Wk, Wv).O, Op = attention_forward(Xp, Wq, Wk, Wv).O;
    Matrix expected(6);
    for (size_t i = 0; i < 6; ++i) expected[i] = O[perm[i]];
    expect(close(Op, expected, 1e-12), "无掩码：输入行置换，输出同样置换");

    // 数值稳定：分数量级 ±1000 时不溢出
    const Matrix big = {{1000.0, -1000.0}, {-1000.0, 1000.0}};
    const Matrix sm = softmax_rows(big);
    bool finite = true;
    for (const auto& row : sm)
        for (double x : row) finite = finite && std::isfinite(x);
    expect(finite && close(sm, {{1, 0}, {0, 1}}, 1e-12), "softmax 对 ±1000 的分数数值稳定");

    // 掩码必须在 softmax 之前：先 softmax 再置零，行和不再为 1
    Matrix wrong = plain.A;
    wrong[0][1] = 0.0;
    expect(std::fabs(wrong[0][0] + wrong[0][1] - 1.0) > 0.3 &&
               std::fabs(masked.A[0][0] + masked.A[0][1] - 1.0) < 1e-12,
           "先 softmax 再置零会破坏行和；softmax 前加掩码则行和为 1");

    // 输入校验
    auto throws = [](const Matrix& x, const Matrix& wq, const Matrix& wk, const Matrix& wv) {
        try {
            attention_forward(x, wq, wk, wv);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const Matrix ones23(2, std::vector<double>(3, 1)), ones32(3, std::vector<double>(2, 1)),
        ones34(3, std::vector<double>(4, 1));
    expect(!throws(ones23, ones32, ones32, ones34), "合法输入不报错");
    expect(throws({}, ones32, ones32, ones34), "空序列报错");
    expect(throws(ones23, Matrix(4, std::vector<double>(2, 1)), ones32, ones34), "d_model 对不上报错");
    expect(throws(ones23, ones32, Matrix(3, std::vector<double>(5, 1)), ones34), "Q、K 维数不同报错");
    expect(throws({{1, 2, 3}, {1, 2}}, ones32, ones32, ones34), "非规整矩阵报错");

    std::printf("\n%s\n", failures == 0 ? "全部通过" : "存在失败项");
    return failures == 0 ? 0 : 1;
}
