"""假阳性实验：固定插入 1000 个字符串，改变位数组长度 m，用 main.cpp 本身统计假阳性与假阴性。

    python tests/fp_experiment.py          # 在 basic/ 目录下运行，需要 g++ 与 Python 3

输出：results/fp_vs_m.csv，并打印同一张表。
理想值 (1 - exp(-k n / m))^k（k=2）只是参照：两个教学哈希相关性很强、取值范围很窄，
实测假阳性远高于该公式，因此不能用它宣称真实误判率。
"""
import csv
import math
import random
import subprocess
import tempfile
from pathlib import Path

BASIC = Path(__file__).resolve().parent.parent
N_INSERT = 1000
N_QUERY = 20000
SEED = 20261024
M_VALUES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1000000]
DATASETS = {
    "小写字母，长度 5–10": ("abcdefghijklmnopqrstuvwxyz", 5, 10),
    "可打印 ASCII，长度 10–30": ("".join(chr(c) for c in range(33, 127)), 10, 30),
}


def random_strings(rng, count, alphabet, lo, hi, exclude=frozenset()):
    out = set()
    while len(out) < count:
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))
        if s not in exclude:
            out.add(s)
    return sorted(out)


def run(exe, m, members, outsiders):
    ops = [f"add {s}" for s in members] + [f"query {s}" for s in members] + [f"query {s}" for s in outsiders]
    text = f"{m} {len(ops)}\n" + "\n".join(ops) + "\n"
    out = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True).stdout.split()
    member_out, outsider_out = out[:len(members)], out[len(members):]
    return member_out.count("NO"), outsider_out.count("MAYBE")


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "bloom"
        subprocess.run(["g++", "-std=c++17", "-O2", str(BASIC / "main.cpp"), "-o", str(exe)], check=True)
        for name, (alphabet, lo, hi) in DATASETS.items():
            rng = random.Random(SEED)
            members = random_strings(rng, N_INSERT, alphabet, lo, hi)
            outsiders = random_strings(rng, N_QUERY, alphabet, lo, hi, exclude=frozenset(members))
            for m in M_VALUES:
                false_neg, false_pos = run(exe, m, members, outsiders)
                ideal = (1 - math.exp(-2 * N_INSERT / m)) ** 2
                rows.append([name, m, N_INSERT, false_neg, N_QUERY, false_pos,
                             f"{false_pos / N_QUERY:.4f}", f"{ideal:.6f}"])
    header = ["字符串集合", "m", "已插入个数", "已插入元素返回NO的次数", "未插入查询数", "假阳性次数",
              "假阳性率", "理想公式值(k=2)"]
    out_dir = BASIC / "results"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "fp_vs_m.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(map(str, r)))
    print(f"\n已写入 {out_dir / 'fp_vs_m.csv'}")


if __name__ == "__main__":
    main()
