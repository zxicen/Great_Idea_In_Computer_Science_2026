"""main.py —— 命令行入口（完整提供）

用法（在项目根目录 23-GraphColoring/ 下运行）：

    python src/main.py solve --data data/triangle.txt              # 朴素回溯
    python src/main.py solve --data data/triangle.txt --mrv --fc   # MRV + 前向检查
    python src/main.py solve --data data/grid_4x4.txt --two        # 二染色判定
    python src/main.py gen --n 20 --m 32 --seed 1 --out data/rand.txt
    python src/main.py bench --sizes 10 12 14 16 18 20             # 性能对比
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# 让 Windows 老式终端也能正常显示中文与符号
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from graph import load_graph, save_graph, random_graph, is_valid_coloring, describe
from solver import two_color, three_color_backtrack, three_color_mrv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 性能实验的三个组别：(名称, 对应求解函数)
def _solvers():
    return [
        ("朴素回溯", lambda n, adj, st: three_color_backtrack(n, adj, st)),
        ("MRV", lambda n, adj, st: three_color_mrv(n, adj, st, forward_check=False)),
        ("MRV+前向检查", lambda n, adj, st: three_color_mrv(n, adj, st, forward_check=True)),
    ]


def cmd_solve(args):
    n, adj = load_graph(args.data)
    describe(n, adj)

    if args.two:
        result = two_color(n, adj)
        if result is None:
            print("该图【不能】二染色（不是二分图）。")
        else:
            print(f"二染色成功：{result}")
        return

    stats = {"nodes": 0}
    t0 = time.perf_counter()
    if args.mrv or args.fc:
        result = three_color_mrv(n, adj, stats, forward_check=args.fc)
        algo = f"MRV{'+前向检查' if args.fc else ''}"
    else:
        result = three_color_backtrack(n, adj, stats)
        algo = "朴素回溯"
    elapsed = time.perf_counter() - t0

    print(f"算法：{algo}，搜索节点数：{stats['nodes']}，耗时：{elapsed * 1000:.1f} ms")
    if result is None:
        print("结论：该图【不能】三染色。")
    elif is_valid_coloring(n, adj, result):
        print(f"结论：【可以】三染色，方案：{result}")
    else:
        print("警告：求解器返回了非法染色方案，请检查实现！")
        sys.exit(1)


def cmd_gen(args):
    n, adj = random_graph(args.n, args.m, seed=args.seed)
    out = args.out or str(PROJECT_ROOT / "data" / f"rand_n{args.n}_m{args.m}_s{args.seed}.txt")
    save_graph(out, n, adj, comment=f"random graph n={args.n} m={args.m} seed={args.seed}")
    print(f"已生成：{out}")
    describe(n, adj)


def cmd_bench(args):
    """在相同规模的随机图上对比三组算法（每个规模测 args.reps 个实例取平均）。"""
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "bench.csv"

    solvers = _solvers()
    if args.skip_naive:
        solvers = solvers[1:]
    elif args.naive_only:  # Basic 阶段：MRV 尚未实现时只测朴素回溯
        solvers = solvers[:1]

    header = ["n", "m", "可染色比例"] + [f"{name}节点数(均值)" for name, _ in solvers] \
        + [f"{name}耗时ms(均值)" for name, _ in solvers]
    rows = []
    print(" | ".join(header))
    print("-" * 100)

    for n in args.sizes:
        m = round(n * args.degree / 2)
        nodes_sum = {name: 0 for name, _ in solvers}
        time_sum = {name: 0.0 for name, _ in solvers}
        colorable = 0
        for rep in range(args.reps):
            nn, adj = random_graph(n, m, seed=args.seed + rep)
            results = []
            for name, fn in solvers:
                stats = {"nodes": 0}
                t0 = time.perf_counter()
                result = fn(nn, adj, stats)
                elapsed = time.perf_counter() - t0
                results.append(result)
                nodes_sum[name] += stats["nodes"]
                time_sum[name] += elapsed
            # 各组算法的结论必须一致，否则说明实现有 bug
            if len({r is None for r in results}) > 1:
                print(f"警告：n={n} seed={args.seed + rep} 上各算法结论不一致，请检查实现！")
                sys.exit(1)
            colorable += results[0] is not None
        row = [n, m, f"{colorable}/{args.reps}"]
        row += [f"{nodes_sum[name] / args.reps:.1f}" for name, _ in solvers]
        row += [f"{time_sum[name] / args.reps * 1000:.2f}" for name, _ in solvers]
        rows.append(row)
        print(" | ".join(map(str, row)))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\n数据已保存到 {csv_path}（可用 Excel 画对比曲线）")


def main():
    parser = argparse.ArgumentParser(description="图的三染色判定：回溯搜索与启发式剪枝")
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="求解一个图实例")
    p_solve.add_argument("--data", required=True, help="图文件路径")
    p_solve.add_argument("--two", action="store_true", help="二染色判定（任务一）")
    p_solve.add_argument("--mrv", action="store_true", help="使用 MRV 启发式（Advance C1）")
    p_solve.add_argument("--fc", action="store_true", help="开启前向检查（Advance C1）")

    p_gen = sub.add_parser("gen", help="生成随机图实例")
    p_gen.add_argument("--n", type=int, required=True, help="顶点数")
    p_gen.add_argument("--m", type=int, required=True, help="边数")
    p_gen.add_argument("--seed", type=int, default=1)
    p_gen.add_argument("--out", default=None, help="输出路径（默认 data/rand_n.._m.._s..txt）")

    p_bench = sub.add_parser("bench", help="三组算法性能对比")
    p_bench.add_argument("--sizes", type=int, nargs="+", default=[10, 12, 14, 16, 18, 20])
    p_bench.add_argument("--degree", type=float, default=3.0, help="平均度数（m = n*degree/2）")
    p_bench.add_argument("--reps", type=int, default=3, help="每个规模的重复次数（取平均）")
    p_bench.add_argument("--seed", type=int, default=2026)
    p_bench.add_argument("--skip-naive", dest="skip_naive", action="store_true",
                         help="跳过朴素回溯（规模较大时它可能跑不动）")
    p_bench.add_argument("--naive-only", dest="naive_only", action="store_true",
                         help="只测朴素回溯（Basic 指数爆炸实验，不需要 MRV）")

    args = parser.parse_args()
    handler = {"solve": cmd_solve, "gen": cmd_gen, "bench": cmd_bench}[args.command]
    try:
        handler(args)
    except NotImplementedError as e:
        # 学生还没完成对应任务时，给出友好提示而不是堆栈
        print(f"\n{e}")
        print("提示：先运行 python src/check.py 查看当前进度。")
        sys.exit(1)


if __name__ == "__main__":
    main()
