"""生成报告用的实验数据（不属于提交包，只有生成的 results/*.csv 进入提交包）：

    python experiments/gen_basic_data.py        # 在项目根目录运行

输出：
  results/hand_cases.csv        手工分析的 5 个小图：二染色、三染色与 NODES
  results/phase_transition.csv  n=30，不同平均度数下朴素回溯的搜索规模（每个度数 30 个随机图）
  results/solve_data.txt        data/ 下 6 个实例的 solve 输出
  results/test_record.txt       check.py、OJ 单文件测试、C++ 副本对照测试的运行记录
指数爆炸实验用 main.py 自带的 bench 命令生成 results/bench.csv，见 docs/basic_report.md。
"""
import csv
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph import is_valid_coloring, random_graph  # noqa: E402
from solver import three_color_backtrack, two_color  # noqa: E402

OUT = ROOT / "results"


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def fmt(coloring):
    return "无解" if coloring is None else "".join(map(str, coloring))


def hand_cases():
    cases = [
        ("三角形 K3", 3, [(0, 1), (1, 2), (2, 0)]),
        ("四边形 C4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("完全图 K4", 4, [(a, b) for a in range(4) for b in range(a + 1, 4)]),
        ("三角形 + 孤立点", 4, [(0, 1), (1, 2), (2, 0)]),
        ("C4 + 孤立点", 5, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ]
    rows = []
    for name, n, edges in cases:
        adj = adjacency(n, edges)
        two = two_color(n, adj)
        stats = {"nodes": 0}
        three = three_color_backtrack(n, adj, stats)
        assert two is None or is_valid_coloring(n, adj, two, 2)
        assert three is None or is_valid_coloring(n, adj, three, 3)
        rows.append([name, n, len(edges), fmt(two), fmt(three), stats["nodes"]])
    return ["图", "n", "m", "二染色", "三染色（朴素回溯）", "NODES"], rows


def phase_transition(n=30, reps=30, seed=2026, degrees=(2, 3, 3.5, 4, 4.5, 4.7, 5, 5.5, 6, 7, 8)):
    rows = []
    for degree in degrees:
        m = round(n * degree / 2)
        nodes, colorable = [], 0
        for rep in range(reps):
            nn, adj = random_graph(n, m, seed=seed + rep)
            stats = {"nodes": 0}
            colorable += three_color_backtrack(nn, adj, stats) is not None
            nodes.append(stats["nodes"])
        rows.append([degree, m, f"{colorable}/{reps}", f"{statistics.mean(nodes):.1f}",
                     f"{statistics.median(nodes):.1f}", max(nodes)])
    return ["平均度数", "m", "可三染色", "平均节点数", "中位节点数", "最大节点数"], rows


def write(name, header, rows):
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"已写入 {OUT / name}")


def run_logged(name, commands):
    """依次运行命令并把「$ 命令」与输出（含退出码）记入 results/<name>。"""
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8") as f:
        for cmd in commands:
            proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8")
            f.write(f"$ python {' '.join(cmd)}\n{proc.stdout}{proc.stderr}[退出码 {proc.returncode}]\n\n")
    print(f"已写入 {OUT / name}")


def main():
    write("hand_cases.csv", *hand_cases())
    write("phase_transition.csv", *phase_transition())
    names = ["triangle", "k4", "cycle5", "grid_4x4", "petersen", "mycielski4"]
    cmds = [["src/main.py", "solve", "--data", f"data/{x}.txt"] for x in names]
    cmds.insert(4, ["src/main.py", "solve", "--two", "--data", "data/grid_4x4.txt"])
    run_logged("solve_data.txt", cmds)
    run_logged("test_record.txt", [["src/check.py"], ["tests/test_oj_three_color.py"], ["tests/test_cpp.py"]])


if __name__ == "__main__":
    main()
