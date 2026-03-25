# LAS to OBJ

用于把室内激光点云 LAS 数据处理为适合射线追踪仿真的轻量级 OBJ 结构模型。

首版目标不是做高保真 BIM 重建，而是构建一条可持续迭代的工程流水线：

1. 读取大体量室内 LAS 点云
2. 按 ROI 框选剔除区域外点云
3. 对区域内点云做体素降采样和游离点过滤
4. 迭代提取墙面、天花板、地板等主平面
5. 对平面点云做打薄和平整
6. 输出低 mesh 数量的 OBJ 结构面模型

## 当前仓库能力

- `LAS` 分块读入，避免一次性把整层数据完整塞进内存
- `ROI` 轴对齐包围盒裁剪
- `Open3D` 体素降采样与统计离群点去除
- `RANSAC` 平面提取
- 墙/地/顶基础分类
- 将平面点云压平成矩形 patch，并导出为轻量 `OBJ`
- 输出处理报告和中间调试点云

当前 mesh 生成策略是“结构仿真优先”的极简矩形 patch，不保留门洞、窗洞和复杂边界。这样能先把仓库的核心链路跑通，后续再逐步升级成多边形裁剪、孔洞恢复、墙体融合和拓扑修复。

## 目录结构

```text
src/las_to_obj/
  cli.py           # 命令行入口
  config.py        # JSON 配置解析
  io.py            # LAS 读取 / OBJ 导出 / 报告写出
  preprocess.py    # 降采样、去噪、调试点云导出
  geometry.py      # 平面投影、分类、矩形 mesh 生成
  planes.py        # 结构平面提取
  pipeline.py      # 端到端流水线编排
docs/
  plans/           # 系统设计文档
  adr/             # 架构决策记录
examples/
  pipeline.sample.json
tests/
  test_geometry.py
```

## 安装

建议使用独立虚拟环境。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

如果需要处理 `LAZ`，还需要额外安装 laspy 的后端，例如：

```powershell
pip install "laspy[lazrs]"
```

## 运行

先复制示例配置：

```powershell
Copy-Item examples\pipeline.sample.json pipeline.local.json
```

编辑 `pipeline.local.json` 中的输入输出路径和 ROI，然后执行：

```powershell
python main.py run --config pipeline.local.json
```

或：

```powershell
python -m las_to_obj.cli run --config pipeline.local.json
```

## 处理建议

- 对 `1.8G` 单层点云，优先先用 ROI 把单层中实际有效区域裁出来。
- 首轮参数调优先看 `preprocessed.ply`、`remaining_after_planes.ply` 和报告里的平面数量。
- 若墙面过碎，先调大 `voxel_size`，再调大 `distance_threshold` 和 `min_plane_points`。
- 若平面被错误合并，先减小 `distance_threshold`，再减小 `extent_trim_percent`。

## 下一步建议

- 增加交互式 ROI 可视化框选
- 增加共面墙体融合与洞口识别
- 从矩形 patch 升级为二维轮廓重建 + 孔洞保留
- 增加楼层坐标系、批处理和任务缓存
- 增加仿真前的 mesh 质量检查
