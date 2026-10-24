"""OJ 单文件 three_color.py 的本地验收：

    python tests/test_oj_three_color.py

按 include/README.md 的 19 个测试点类型构造输入，
用独立的显式栈实现（不使用递归）交叉验证 SAT/UNSAT、颜色方案与 NODES，
再通过子进程检查 stdin/stdout 的完整输出格式（全文精确比较）。
"""
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OJ = ROOT / "include" / "src" / "three_color.py"


def reference(n, edges):
    """显式栈版朴素回溯，返回 (coloring | None, nodes)。规则与题面完全一致。"""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    color = [-1] * n
    nodes = 1  # dfs(0)
    v = 0
    nxt = [0] * (n + 1)  # nxt[v]：顶点 v 下一个要尝试的颜色
    while True:
        if v == n:
            return color[:], nodes
        placed = False
        while nxt[v] < 3:
            c = nxt[v]
            nxt[v] += 1
            if all(color[u] != c for u in adj[v]):
                color[v] = c
                v += 1
                nxt[v] = 0
                nodes += 1  # 进入 dfs(v + 1)
                placed = True
                break
        if placed:
            continue
        if v == 0:
            return None, nodes
        v -= 1
        color[v] = -1


def expected_output(n, edges):
    if any(u == v for u, v in edges):
        return "UNSAT\nNODES 0\n"
    col, nodes = reference(n, edges)
    if col is None:
        return f"UNSAT\nNODES {nodes}\n"
    return "SAT\n" + " ".join(map(str, col)) + f"\nNODES {nodes}\n"


def make_input(n, edges):
    return f"{n} {len(edges)}\n" + "".join(f"{u} {v}\n" for u, v in edges)


def complete(k, offset=0):
    return [(offset + i, offset + j) for i in range(k) for j in range(i + 1, k)]


def cycle(n, offset=0):
    return [(offset + i, offset + (i + 1) % n) for i in range(n)]


def path(n):
    return [(i, i + 1) for i in range(n - 1)]


def grid(r, c):
    e = []
    for i in range(r):
        for j in range(c):
            v = i * c + j
            if j + 1 < c:
                e.append((v, v + 1))
            if i + 1 < r:
                e.append((v, v + c))
    return e


def petersen():
    return [(i, (i + 1) % 5) for i in range(5)] \
        + [(5 + i, 5 + (i + 2) % 5) for i in range(5)] \
        + [(i, i + 5) for i in range(5)]


def grotzsch():
    # 5 元环 0-4；影子点 5-9（影子 i 连 v_i 的两个邻居）；中心 10 连所有影子点
    e = [(i, (i + 1) % 5) for i in range(5)]
    for i in range(5):
        e += [(5 + i, (i + 1) % 5), (5 + i, (i - 1) % 5), (5 + i, 10)]
    return e


def layered_tripartite(n, m, seed=7):
    """第 v 个顶点属于第 v%3 层，只在不同层之间连边。

    先给每个顶点向"层号更小的每一层"各连一个更早的顶点，
    保证贪心给 v 的颜色恰好是它的层号，因此朴素回溯零回溯、NODES = n + 1。
    """
    rng = random.Random(seed)
    edges = set()
    for v in range(n):
        for layer in range(v % 3):
            u = max(x for x in range(v) if x % 3 == layer)
            edges.add((u, v))
    pool = [(u, v) for u in range(n) for v in range(u + 1, n)
            if u % 3 != v % 3 and (u, v) not in edges]
    edges.update(rng.sample(pool, m - len(edges)))
    return sorted(edges)


def random_graph(n, m, seed):
    rng = random.Random(seed)
    pool = [(u, v) for u in range(n) for v in range(u + 1, n)]
    return sorted(rng.sample(pool, m))


# (名称, n, edges, 手算/已知的 NODES 或 None)
CASES = [
    ("1 三角形（题面样例）", 3, cycle(3), 4),
    ("2 K4", 4, complete(4), 16),
    ("3 单点无边", 1, [], 2),
    ("4 空图 n=5", 5, [], 6),
    ("5 五元环", 5, cycle(5), 6),
    ("6 链 n=10", 10, path(10), 11),
    ("7 三角形+K4 不连通", 7, cycle(3) + complete(4, 3), None),
    ("8 三角形+五元环 不连通", 8, cycle(3) + cycle(5, 3), None),
    ("9 Petersen", 10, petersen(), 11),
    ("10 4x4 网格", 16, grid(4, 4), 17),
    ("11 Grötzsch", 11, grotzsch(), 472),
    ("12 K5", 5, complete(5), None),
    ("13 长链 n=200", 200, path(200), 201),
    ("14 偶环 n=500", 500, cycle(500), None),
    ("15 K(40,40)", 80, [(i, 40 + j) for i in range(40) for j in range(40)], 81),
    ("16 分层三部图 n=500 m=1500", 500, layered_tripartite(500, 1500), 501),
    ("17 含自环", 4, [(0, 1), (2, 2), (1, 3)], 0),
    ("18 n=1 且自环", 1, [(0, 0)], 0),
    ("19 随机图 n=28 m=62", 28, random_graph(28, 62, 1), 224443),
]


def run_oj(text):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, str(OJ)], input=text, capture_output=True, text=True, timeout=30)
    return p.stdout, p.returncode, time.perf_counter() - t0


def validate(n, edges, out):
    """独立检查输出：SAT 时颜色方案必须合法。"""
    lines = out.split("\n")
    if lines[0] == "SAT":
        colors = list(map(int, lines[1].split()))
        assert len(colors) == n and all(c in (0, 1, 2) for c in colors), "颜色数量或取值非法"
        assert all(colors[u] != colors[v] for u, v in edges), "SAT 方案含同色边"


def main():
    failed = 0
    for name, n, edges, known in CASES:
        exp = expected_output(n, edges)
        out, rc, dt = run_oj(make_input(n, edges))
        ok = rc == 0 and out == exp and dt < 2.0
        if ok and known is not None:
            ok = out.strip().split("\n")[-1] == f"NODES {known}"
        try:
            validate(n, edges, out)
        except AssertionError as e:
            ok = False
            print("   ", e)
        failed += not ok
        lines = out.strip().splitlines() or ["(无输出)"]
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<30} {lines[0]:<6} {lines[-1]:<14} {dt * 1000:7.1f} ms")
        if not ok:
            print("   期望：", repr(exp[:80]))
            print("   实际：", repr(out[:80]))

    # 题面样例：全文精确比较
    out, _, _ = run_oj("3 3\n0 1\n1 2\n2 0\n")
    sample_ok = out == "SAT\n0 1 2\nNODES 4\n"
    print(f"[{'PASS' if sample_ok else 'FAIL'}] 题面样例全文精确比较")
    failed += not sample_ok

    total = len(CASES) + 1
    print(f"\n{total - failed}/{total} 通过")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
