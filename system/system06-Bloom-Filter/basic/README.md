# Basic Bloom Filter 插入与查询

先阅读 [项目任务书](../README.md) 的 Basic 部分，并完成 Q1-Q5。本阶段自行编写 C++ 程序，对应 [ACMOJ 3261](https://acm.sjtu.edu.cn/OnlineJudge/problem/3261)。

用 0/1 数组保存状态，按任务书的两个教学哈希实现 `add` 和 `query`。查询输出 `NO` 或 `MAYBE`，不能用精确集合代替 Bloom Filter 的查询。

自行创建 `main.cpp` 后，在本目录编译：

```bash
g++ -std=c++17 -O2 main.cpp -o bloom-basic
./bloom-basic < input.txt
```

测试空集合、重复插入、碰撞、`m=1` 和不存在的元素。提交 OJ AC 记录、源码、测试记录与 Q1-Q5。进阶任务见 [Advance 说明](../advance/README.md)。

## 本目录内容

- [main.cpp](main.cpp)：Basic 实现（两个教学哈希，`vector<uint8_t>` 位数组）。
- [report.md](report.md)：Q1–Q5 回答与测试记录。
- [tests/](tests/)：手算用例 `cases/`、`std::set` 精确基线检查器 `oracle_check.cpp` 与测试脚本；运行 `python tests/run_tests.py`。
