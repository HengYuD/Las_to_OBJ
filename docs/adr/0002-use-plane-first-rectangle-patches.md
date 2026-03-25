# ADR-0002: 首版采用平面矩形 patch 而非复杂多边形重建

## Status
Accepted

## Context

目标输出是适合射线追踪仿真的轻量结构模型。当前最重要的指标是：

- 模型体积可控
- mesh 数量足够少
- 墙、地、顶等主结构面足够平整
- 处理流程可稳定复现

如果首版直接追求带门窗孔洞的复杂多边形边界重建，工程复杂度会显著上升，且容易阻塞整体系统搭建。

## Decision

首版在每个检测到的结构平面上，只生成一个简化矩形 patch：

- 先把平面点云投影到局部二维坐标系
- 计算截尾后的二维包围盒
- 用两个三角形生成一个矩形 mesh
- 以 `wall/floor/ceiling` 标签输出到 `OBJ`

## Consequences

### Positive

- 输出 mesh 极少，天然适合仿真输入
- 平面打薄和平整效果明确
- 实现简单，便于快速调参和排错

### Negative

- 会丢失门洞、窗洞和凹凸边界
- 部分平面可能被过度补全
- 复杂空间的几何真实性有限

### Neutral

- 该策略可作为基线结果，为后续多边形轮廓重建提供对照

## Alternatives Considered

### Alpha Shape / Poisson 重建

- 被拒绝：更适合表面重建，不利于生成“平整且低面数”的结构模型

### 二维轮廓重建 + 孔洞检测

- 暂缓：这是更合理的中期路线，但不适合作为首版入口

## References

- `docs/plans/2026-03-24-indoor-obj-pipeline-design.md`
