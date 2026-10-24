"""solver.py —— 任务一~三：从线性时间到指数爆炸，再到启发式剪枝

【你要完成】two_color、three_color_backtrack、three_color_mrv

三个函数的通用约定：
  - n: 顶点数；adj: 邻接表（adj[v] 是 v 的邻居集合）
  - 返回染色方案（长度为 n 的列表，coloring[v] ∈ {0,1,2}）或 None（不可染）
  - stats: 一个 dict，每进入一次递归就 stats["nodes"] += 1。
    它是性能实验的唯一度量，三个函数口径必须一致（按"递归调用次数"计）。

动手前请先读 docs/implement_guide.md 第 4 节的函数契约。
"""

import sys
from collections import deque

NUM_COLORS = 3
UNCOLORED = -1  # 未染色的哨兵值


def two_color(n, adj):
    """任务一：判定图能否二染色（二分图判定），能则返回染色方案。

    提示（建议按顺序思考）：
        1. 二染色 = 交替染色：一个顶点染 0，它的所有邻居就必须染 1，
           邻居的邻居又必须染 0……用 BFS 或 DFS 把这个"必须"传播出去。
        2. 传播过程中如果发现某条边两端颜色相同，立即返回 None。
        3. 图可能不连通！外层要遍历所有尚未染色的顶点，
           每个连通分量任选一个起点染 0 即可。
        4. 想一想：为什么二染色不需要"回溯"？（验收时会问）
           ——因为第一个顶点的颜色一旦定下，整个连通分量就被唯一决定了，
           没有任何"选择"，自然没有"选错了重来"。

    参考规模：约 20 行。
    """
    coloring = [UNCOLORED] * n
    for start in range(n):
        if coloring[start] != UNCOLORED:
            continue
        # 每个连通分量任选起点染 0，之后整个分量的颜色被唯一决定
        coloring[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if coloring[v] == UNCOLORED:
                    coloring[v] = 1 - coloring[u]
                    queue.append(v)
                elif coloring[v] == coloring[u]:
                    # 同色相邻（含自环 u == v）：存在奇环
                    return None
    return coloring


def three_color_backtrack(n, adj, stats):
    """任务二：朴素回溯三染色判定。

    算法骨架（伪代码见 docs/implement_guide.md 第 4 节）：
        - 按固定顶点顺序 0, 1, 2, ... 逐个染色；
        - 每个顶点依次尝试颜色 0, 1, 2；
        - 颜色与某个【已染色】邻居冲突 → 试下一个颜色；
        - 三个颜色都冲突 → 回溯（返回上一层换颜色）。

    提示（建议按顺序思考）：
        1. coloring = [UNCOLORED] * n；递归函数 dfs(v) 表示"准备染顶点 v"。
        2. 进入 dfs 第一件事：stats["nodes"] += 1。
        3. 递归出口：v == n 说明全部染完，成功。
        4. 冲突检查：对 v 的邻居 u，若 coloring[u] != UNCOLORED 且等于 c，冲突。
        5. 递归失败返回后，记得把 coloring[v] 恢复为 UNCOLORED——
           "回溯"的"回"字就体现在这里。忘了它，后面所有分支都会被污染。

    参考规模：约 20 行。
    """
    # 自环无法合法染色；冲突检查只看已染色邻居，必须在搜索前单独处理
    if any(v in adj[v] for v in range(n)):
        return None

    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 100))
    coloring = [UNCOLORED] * n

    def dfs(v):
        stats["nodes"] += 1
        if v == n:
            return True
        for c in range(NUM_COLORS):
            if any(coloring[u] == c for u in adj[v]):
                continue
            coloring[v] = c
            if dfs(v + 1):
                return True
            coloring[v] = UNCOLORED  # 回溯：撤销当前尝试
        return False

    return coloring if dfs(0) else None


def three_color_mrv(n, adj, stats, forward_check=False):
    """Advance C1：MRV 启发式 + 可选的前向检查。

    与任务二的唯一区别是【选顶点的顺序】和【死路发现的时机】：

    MRV（Most Constrained Variable，最受约束者优先）：
        每一步不再按固定顺序，而是选择"当前可用颜色最少"的未染色顶点先染
        （平局时取度数较大的）。直觉：最容易卡死的先安排，早死早超生。
        想一想：为什么改变顺序不影响"穷尽所有可能"，因此不会漏解？

    前向检查（forward_check=True 时开启）：
        染完顶点 v 的颜色 c 后，立刻把 c 从 v 的所有未染色邻居的
        "可用颜色集合"中删除；若某邻居的集合变空，立即回溯——
        不必等到亲自染到它才发现死路。

    实现建议：
        - 维护 domains = [set({0,1,2}) for _ in range(n)] 表示每个顶点的可用色；
        - 每层递归前可以复制 domains（本项目的规模下完全够快，别过度设计）；
        - forward_check=False 时应退化为"纯 MRV"，供对比实验分离两个优化的贡献；
        - stats["nodes"] 的口径与任务二保持一致（每进入一次递归 +1）。

    参考规模：约 35 行。
    """
    # TODO: Advance C1
    raise NotImplementedError("Advance C1还没做：请先完成 solver.py 的 three_color_mrv，然后运行 python src/check.py")
