# Transformer · Basic

本目录是学生下发入口。完整课程要求以 [Basic 任务书](materials/任务书-Basic.md) 为准；[OJ 说明](materials/OJ-说明.md) 只解释自动检查部分。

## 目录结构

- [下发压缩包](05-Transformer-Basic.zip)：可直接下载、解压后独立使用。
- [materials/](materials/)：与压缩包相同的展开内容，便于在 GitHub 上阅读。
- [下发说明](materials/下发说明.md)：介绍包内文件与验收边界。
- [attention.py](materials/attention.py)：单头注意力、softmax、因果掩码的参考演示。
- [requirements.txt](materials/requirements.txt)：NumPy 依赖；[README-示例代码.md](materials/README-示例代码.md) 说明运行方法。
- 任务书 Basic 部分：Q1–Q5、矩阵日志、权重热力图与因果性检查。

## 建议阅读与使用顺序

1. 阅读 Basic 任务书，从 Q/K/V 的维度和缩放点积公式开始。
2. 运行 attention.py，对照手算的小矩阵、行和、单位置及因果掩码输出。
3. 自己记录 Q、K、V、S、A、O 的形状和值，绘图并做修改未来输入的因果性实验。
4. OJ 说明要求从标准输入输出矩阵；演示脚本只打印内置样例，不能直接提交。

从解压后的包顶层运行相应命令：

    python -m pip install -r requirements.txt
    python attention.py

掩码在 softmax 前应用；稳定 softmax 先减每行最大值。

完成 Basic 后，再读 [Advance 说明](../advance/README.md)。

## 本目录内容

- [attention_basic.py](attention_basic.py)：单头缩放点积注意力（形状检查、稳定 softmax、因果掩码）。
- [oj_solution.py](oj_solution.py)：OJ 3280 的标准输入输出程序（纯标准库）。
- [run_basic.py](run_basic.py)：生成 `output/matrix_log.txt` 与 `output/attention_heatmap.png`，并做因果性、置换等实验。
- [test_basic.py](test_basic.py)：自动验收（`python -m unittest -v test_basic`）。
- [report.md](report.md)：Q1–Q5 答案、验收对照与 OJ 提交说明。
- C++ 副本：[attention_basic.cpp](attention_basic.cpp)（与 `attention_basic.py` 对应，含内置自检）、[oj_solution.cpp](oj_solution.cpp)（与 `oj_solution.py` 输出一致）；[test_cpp.py](test_cpp.py) 把它们与 Python/NumPy 实现逐元素对照（`python -m unittest -v test_cpp`）。
