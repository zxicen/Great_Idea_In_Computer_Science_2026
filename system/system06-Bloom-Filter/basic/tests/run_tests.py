"""Basic Bloom Filter 测试：

    python tests/run_tests.py            # 在 basic/ 目录下运行

1. cases/*.in 与 *.out 的精确比较（手算用例：样例、空集合、重复插入、碰撞、m=1、最大长度）；
2. 固定种子的随机测试：与独立的 Python 参考模型逐行比较；
3. 每个用例都交给 oracle_check（C++，std::set 精确基线）确认已插入元素全部返回 MAYBE。
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

BASIC = Path(__file__).resolve().parent.parent
TESTS = BASIC / "tests"
CXX = "g++"


def positions(s, m):
    c = s.encode("ascii")
    h1 = sum(c) % m
    h2 = sum((i + 1) * b for i, b in enumerate(c)) % m
    return h1, h2


def model(m, ops):
    """参考模型：不直接复用 main.cpp 的任何逻辑。"""
    bits = [0] * m
    out = []
    for op, s in ops:
        h1, h2 = positions(s, m)
        if op == "add":
            bits[h1] = bits[h2] = 1
        else:
            out.append("MAYBE" if bits[h1] and bits[h2] else "NO")
    return out


def format_input(m, ops):
    return f"{m} {len(ops)}\n" + "".join(f"{op} {s}\n" for op, s in ops)


def random_ops(rng, m, n, alphabet, max_len):
    ops, pool = [], []
    for _ in range(n):
        if pool and rng.random() < 0.3:
            s = rng.choice(pool)
        else:
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_len)))
        if rng.random() < 0.5:
            ops.append(("add", s))
            pool.append(s)
        else:
            ops.append(("query", s))
    return ops


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for src, exe in (("main.cpp", "bloom"), ("tests/oracle_check.cpp", "oracle")):
            r = subprocess.run([CXX, "-std=c++17", "-O2", "-Wall", "-Wextra", str(BASIC / src), "-o", str(tmp / exe)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"编译 {src} 失败：\n{r.stderr}")
                return 1

        def run_case(name, input_text, expected):
            nonlocal failures
            inp = tmp / "case.in"
            outp = tmp / "case.out"
            inp.write_text(input_text)
            with open(inp) as fi, open(outp, "w") as fo:
                rc = subprocess.run([str(tmp / "bloom")], stdin=fi, stdout=fo, timeout=5).returncode
            actual = outp.read_text()
            ok = rc == 0 and (expected is None or actual == expected)
            oracle = subprocess.run([str(tmp / "oracle"), str(inp), str(outp)], capture_output=True, text=True)
            ok = ok and oracle.returncode == 0
            failures += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<34} {oracle.stdout.strip() or oracle.stderr.strip()}")
            if not ok and expected is not None:
                print("   期望：", repr(expected[:100]))
                print("   实际：", repr(actual[:100]))

        for inf in sorted((TESTS / "cases").glob("*.in")):
            run_case(inf.stem, inf.read_text(), inf.with_suffix(".out").read_text())

        rng = random.Random(20261024)
        configs = [
            (1, 200, "ab", 3), (2, 200, "ab", 3), (3, 300, "abc", 4), (8, 500, "abcdefghij", 3),
            (17, 800, "abcdefgh", 5), (64, 1000, "abcdefghijklmnop", 4),
            (1000, 2000, "abcdefghijklmnopqrstuvwxyz", 6), (1000000, 10000, "".join(chr(c) for c in range(33, 127)), 100),
        ]
        for m, n, alphabet, max_len in configs:
            ops = random_ops(rng, m, n, alphabet, max_len)
            exp = "".join(x + "\n" for x in model(m, ops))
            run_case(f"随机 m={m} n={n}", format_input(m, ops), exp)

    print(f"\n{'全部通过' if not failures else str(failures) + ' 项失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
