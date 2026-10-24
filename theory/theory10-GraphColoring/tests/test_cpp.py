"""C++ 副本的本地验收：

    python tests/test_cpp.py          （需要 g++ 或 clang++）

1. include/src/three_color.cpp：对 19 个 OJ 风格测试点做全文精确比较，并与 Python 版 three_color.py 逐字节对照；
2. src/solver.cpp：在 data/ 的全部图和一批随机图上，与 Python 版 two_color / three_color_backtrack
   对照染色方案与搜索节点数 NODES，并检查染色合法。
"""
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import is_valid_coloring, load_graph, random_graph, save_graph  # noqa: E402
from solver import three_color_backtrack, two_color  # noqa: E402
from test_oj_three_color import CASES, expected_output, make_input, run_oj  # noqa: E402

CXX = shutil.which("g++") or shutil.which("clang++")


def compile_cpp(src, out):
    subprocess.run([CXX, "-std=c++17", "-O2", "-Wall", "-Wextra", "-o", str(out), str(ROOT / src)],
                   check=True, capture_output=True, text=True)


def parse_solver_output(text):
    """解析 solver 的输出，返回 (two, three, nodes)，染色方案不可染色时为 None。"""
    lines = text.strip().split("\n")
    i = 0
    assert lines[i].startswith("two_color: "), text
    two = None
    if lines[i].endswith("SAT") and not lines[i].endswith("UNSAT"):
        i += 1
        two = list(map(int, lines[i].split()))
    i += 1
    assert lines[i].startswith("three_color_backtrack: "), text
    three = None
    if not lines[i].endswith("UNSAT"):
        i += 1
        three = list(map(int, lines[i].split()))
    i += 1
    assert lines[i].startswith("NODES "), text
    return two, three, int(lines[i].split()[1])


def main():
    if CXX is None:
        print("未找到 C++ 编译器，跳过")
        return 0
    failed = total = 0
    tmp = Path(tempfile.mkdtemp())
    try:
        oj_bin, solver_bin = tmp / "three_color", tmp / "solver"
        compile_cpp("include/src/three_color.cpp", oj_bin)
        compile_cpp("src/solver.cpp", solver_bin)

        # 1. OJ 单文件
        for name, n, edges, known in CASES:
            text = make_input(n, edges)
            t0 = time.perf_counter()
            cpp = subprocess.run([str(oj_bin)], input=text, capture_output=True, text=True, timeout=30).stdout
            dt = time.perf_counter() - t0
            py, _, _ = run_oj(text)
            ok = cpp == expected_output(n, edges) == py and dt < 2.0
            if ok and known is not None:
                ok = cpp.strip().split("\n")[-1] == f"NODES {known}"
            total += 1
            failed += not ok
            lines = cpp.strip().splitlines() or ["(无输出)"]
            print(f"[{'PASS' if ok else 'FAIL'}] OJ C++ {name:<28} {lines[0]:<6} {lines[-1]:<14} {dt * 1000:7.1f} ms")
        sample = subprocess.run([str(oj_bin)], input="3 3\n0 1\n1 2\n2 0\n", capture_output=True, text=True).stdout
        sample_ok = sample == "SAT\n0 1 2\nNODES 4\n"
        total += 1
        failed += not sample_ok
        print(f"[{'PASS' if sample_ok else 'FAIL'}] OJ C++ 题面样例全文精确比较")

        # 2. solver.cpp 与 solver.py 对照
        graphs = [(p.name, p) for p in sorted((ROOT / "data").glob("*.txt"))]
        rng = random.Random(20261024)
        for i in range(40):
            n = rng.randint(3, 14)
            _, adj = random_graph(n, rng.randint(0, min(n * (n - 1) // 2, 3 * n)), seed=rng.random())
            path = tmp / f"rand{i}.txt"
            save_graph(path, n, adj)
            graphs.append((f"随机图 #{i} n={n}", path))
        loop = tmp / "self_loop.txt"
        loop.write_text("4 3\n0 1\n2 2\n1 3\n", encoding="utf-8")
        graphs.append(("含自环", loop))
        one = tmp / "single.txt"
        one.write_text("# 单点\n1 0\n", encoding="utf-8")
        graphs.append(("单点无边", one))

        for name, path in graphs:
            n, adj = load_graph(path)
            stats = {"nodes": 0}
            exp_two, exp_three = two_color(n, adj), three_color_backtrack(n, adj, stats)
            out = subprocess.run([str(solver_bin), str(path)], capture_output=True, text=True, check=True).stdout
            two, three, nodes = parse_solver_output(out)
            ok = (two == exp_two and three == exp_three and nodes == stats["nodes"]
                  and (two is None or is_valid_coloring(n, adj, two, 2))
                  and (three is None or is_valid_coloring(n, adj, three, 3)))
            total += 1
            failed += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] solver C++ {name:<24} 二染色={'是' if two else '否'} "
                  f"三染色={'是' if three else '否'} NODES={nodes}")
            if not ok:
                print("   期望：", exp_two, exp_three, stats["nodes"])
                print("   实际：", two, three, nodes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{total - failed}/{total} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
