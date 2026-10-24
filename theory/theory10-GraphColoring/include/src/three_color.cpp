// 图的三染色判定（Basic）：朴素回溯 —— C++17 单文件，与 three_color.py 的输出逐字节一致。
//
// 输入：  n m  然后 m 行 "u v"（无重边；若含自环则直接输出 UNSAT / NODES 0）
// 输出：  SAT / 染色方案（第 i 个数为顶点 i-1 的颜色）/ NODES x ；不可染色时只有 UNSAT / NODES x
//
// 搜索规则完全固定：顶点顺序 0..n-1，颜色顺序 0,1,2，只与已染色邻居判冲突，找到第一个完整方案即停；
// NODES = 进入 dfs 的总次数（dfs(0) 的首次调用与 v == n 的成功终止调用都计入）。
//
//     g++ -std=c++17 -O2 -o three_color three_color.cpp && ./three_color < input.txt
#include <cstdio>
#include <optional>
#include <vector>

constexpr int NUM_COLORS = 3;
constexpr int UNCOLORED = -1;  // 表示“这个顶点还没有颜色”

namespace {
int n;
std::vector<std::vector<int>> adj;
std::vector<int> coloring;
long long nodes = 0;

bool dfs(int v) {
    ++nodes;
    if (v == n) return true;
    for (int c = 0; c < NUM_COLORS; ++c) {
        bool conflict = false;
        for (int u : adj[v])
            if (coloring[u] == c) {
                conflict = true;
                break;
            }
        if (conflict) continue;
        coloring[v] = c;
        if (dfs(v + 1)) return true;
        coloring[v] = UNCOLORED;  // 撤销，否则旧颜色会污染之后所有分支
    }
    return false;
}
}  // namespace

int main() {
    int m;
    if (std::scanf("%d %d", &n, &m) != 2) return 0;
    adj.assign(n, {});
    for (int i = 0; i < m; ++i) {
        int u, v;
        if (std::scanf("%d %d", &u, &v) != 2) return 1;
        if (u == v) {
            // 自环永远无法染色；这一步在搜索之前完成，没有任何递归调用，因此 NODES 0
            std::printf("UNSAT\nNODES 0\n");
            return 0;
        }
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    coloring.assign(n, UNCOLORED);
    if (dfs(0)) {
        std::printf("SAT\n");
        for (int i = 0; i < n; ++i) std::printf(i ? " %d" : "%d", coloring[i]);
        std::printf("\nNODES %lld\n", nodes);
    } else {
        std::printf("UNSAT\nNODES %lld\n", nodes);
    }
    return 0;
}
