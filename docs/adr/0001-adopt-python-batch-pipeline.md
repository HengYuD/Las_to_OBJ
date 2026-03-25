# ADR-0001: 采用 Python 本地离线批处理架构

## Status
Accepted

## Context

当前目标是从零搭建一个可迭代的室内点云到仿真 OBJ 的工程仓库。已知输入数据是单层约 `1.8G` 的 `LAS` 点云，核心问题是尽快跑通可复现、可调参、可扩展的处理链路，而不是在第一版就追求最高性能或最复杂重建能力。

## Decision

首版采用 Python 本地离线批处理架构：

- `laspy` 负责 LAS 读入
- `numpy` 负责数值计算
- `open3d` 负责点云预处理和平面分割
- `argparse + JSON` 负责 CLI 和配置

系统保持单仓库、单进程批处理结构，后续再根据性能瓶颈拆分模块或替换底层实现。

## Consequences

### Positive

- 实现速度快，适合快速验证真实数据
- 算法和参数试验成本低
- 可直接在本地处理链路里增加调试产物
- 后续容易扩展为 GUI 或批处理任务系统

### Negative

- 超大点云上的纯 Python 编排性能有限
- 高级几何修复能力需要更多第三方库或自研
- 对 Windows 本地环境依赖较强

### Neutral

- 后续若引入 PDAL 或 C++，可保留当前仓库作为上层编排层

## Alternatives Considered

### CloudCompare/PDAL 工具链拼接

- 被拒绝：适合实验，不适合形成完整可复现系统

### C++ 优先实现

- 被拒绝：当前阶段投入过大，反馈周期过长

## References

- `docs/plans/2026-03-24-indoor-obj-pipeline-design.md`

