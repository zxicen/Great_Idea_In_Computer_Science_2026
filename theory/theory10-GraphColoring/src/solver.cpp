// solver.py 的 C++ 版本：任务一 two_color（二分图判定）与任务二 three_color_backtrack（朴素回溯三染色）。
// Advance 的 three_color_mrv 与 Python 版一样未实现。
//
// 编译：g++ -std=c++17 -O2 -Wall -Wextra -o solver src/solver.cpp
// 用法：./solver data/triangle.txt      或      ./solver < data/triangle.txt
// 图文件格式与 src/graph.py 相同：# 开头为注释；首行 "n m"；随后每行一条无向边 "u v"（顶点编号 0..n-1）。
//
// 输出（可染色时第二行是染色方案，不可染色时没有）：
//     two_color: SAT|UNSAT
//     c0 c1 ... c(n-1)
//     three_color_backtrack: SAT|UNSAT
//     c0 c1 ... c(n-1)
//     NODES x
#include <algorithm>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

constexpr int NUM_COLORS = 3;
constexpr int UNCOLORED = -1;  // 未染色的哨兵值

using Adj = std::vector<std::vector<int>>;  // adj[v]：v 的邻居（已去重，自环会出现 v 自己）
using Coloring = std::optional<std::vector<int>>;

// 任务一：判定图能否二染色，能则返回染色方案。
// 二染色不需要回溯：起点颜色一旦定下，整个连通分量的颜色就被唯一决定，没有“选择”，也就没有“选错了重来”。
Coloring two_color(int n, const Adj& adj) {
    std::vector<int> coloring(n, UNCOLORED);
    for (int start = 0; start < n; ++start) {
        if (coloring[start] != UNCOLORED) continue;
        coloring[start] = 0;  // 每个连通分量任选起点染 0
        std::deque<int> queue{start};
        while (!queue.empty()) {
            const int u = queue.front();
            queue.pop_front();
            for (int v : adj[u]) {
                if (coloring[v] == UNCOLORED) {
                    coloring[v] = 1 - coloring[u];
                    queue.push_back(v);
                } else if (coloring[v] == coloring[u]) {
                    return std::nullopt;  // 同色相邻（含自环 u == v）：存在奇环
                }
            }
        }
    }
    return coloring;
}

namespace {
struct Backtrack {
    int n;
    const Adj& adj;
    long long& nodes;
    std::vector<int> coloring;

    bool dfs(int v) {
        ++nodes;  // 每进入一次递归计一个节点，v == n 的成功终止调用也计入
        if (v == n) return true;
        for (int c = 0; c < NUM_COLORS; ++c) {
            // 只与已染色的邻居判冲突
            const bool conflict =
                std::any_of(adj[v].begin(), adj[v].end(), [&](int u) { return coloring[u] == c; });
            if (conflict) continue;
            coloring[v] = c;
            if (dfs(v + 1)) return true;
            coloring[v] = UNCOLORED;  // 回溯：撤销当前尝试
        }
        return false;
    }
};
}  // namespace

// 任务二：按固定顺序 0..n-1、颜色 0,1,2 的朴素回溯；nodes 累计递归调用次数。
Coloring three_color_backtrack(int n, const Adj& adj, long long& nodes) {
    // 自环无法合法染色；冲突检查只看已染色邻居，必须在搜索前单独处理
    for (int v = 0; v < n; ++v)
        if (std::binary_search(adj[v].begin(), adj[v].end(), v)) return std::nullopt;
    Backtrack bt{n, adj, nodes, std::vector<int>(n, UNCOLORED)};
    if (bt.dfs(0)) return bt.coloring;
    return std::nullopt;
}

// 读取 graph.py 格式的图；格式错误时返回 false。
bool load_graph(std::istream& in, int& n, Adj& adj) {
    std::string line;
    bool have_header = false;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream ss(line);
        if (!have_header) {
            long long m;
            if (!(ss >> n >> m) || n < 0) return false;
            adj.assign(n, {});
            have_header = true;
        } else {
            int u, v;
            if (!(ss >> u >> v) || u < 0 || v < 0 || u >= n || v >= n) return false;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
    for (auto& nb : adj) {
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }
    return have_header;
}

static void print_coloring(const std::vector<int>& c) {
    for (size_t i = 0; i < c.size(); ++i) std::printf(i ? " %d" : "%d", c[i]);
    std::printf("\n");
}

int main(int argc, char** argv) {
    int n = 0;
    Adj adj;
    bool ok;
    if (argc > 1) {
        std::ifstream f(argv[1]);
        if (!f) {
            std::fprintf(stderr, "无法打开图文件：%s\n", argv[1]);
            return 1;
        }
        ok = load_graph(f, n, adj);
    } else {
        ok = load_graph(std::cin, n, adj);
    }
    if (!ok) {
        std::fprintf(stderr, "图文件格式错误：需要首行 \"n m\"，其后每行一条边 \"u v\"，且顶点在 0..n-1 内\n");
        return 1;
    }

    const Coloring two = two_color(n, adj);
    std::printf("two_color: %s\n", two ? "SAT" : "UNSAT");
    if (two) print_coloring(*two);

    long long nodes = 0;
    const Coloring three = three_color_backtrack(n, adj, nodes);
    std::printf("three_color_backtrack: %s\n", three ? "SAT" : "UNSAT");
    if (three) print_coloring(*three);
    std::printf("NODES %lld\n", nodes);
    return 0;
}
