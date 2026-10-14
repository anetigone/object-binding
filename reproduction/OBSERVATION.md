# L23 实例关系观察与训练目标对照

入口 `observe_binding.py`，配置 `config-observation.json`。

这轮从具体关系和空间近邻线索出发。默认独立输出 `runs/observation_l23_v1`，支持重新提取特征并训练全部 probe，不要求旧缓存、预测或 checkpoint。已有本轮完整缓存/任务可继续；旧实验目录不能用作新输出。

## 服务器执行

将配置中的 dataset_root、hf_cache 和可选 dataset_factory 调整为服务器现用路径/读取接口，然后在项目根目录执行：

```bash
python reproduction/observe_binding.py --stage plan
python reproduction/observe_binding.py --stage all
```

如希望直接与上一轮比较，复用确认实验的图像 ID，但重新提取和训练：

```bash
python reproduction/observe_binding.py --splits-file runs/confirmation_l23_v1/splits.json --stage plan
python reproduction/observe_binding.py --splits-file runs/confirmation_l23_v1/splits.json --stage all
```

两种方式都不构成新的独立确认。默认 split_seed 与既有实验相同；不传 splits-file 时按普通 dataset loader 抽样，没有排除旧图像。使用 splits-file 时，ID 会验证为读取接口的合法、不重叠 train/val/test 子集。后续各阶段保持相同参数；配置、源码或 split 文件变化需新 output。

`--dataset-root`、`--output`、`--device` 可覆盖相应字段。阶段可拆成 extract、train、report、visualize；all 顺序为提取→3 图计算位置数值检查→训练→统计→案例图。train 需要本轮完整缓存，缺数值检查时先检查；要补缺失缓存使用 extract 或 all。

plan 不加载 ML 库、模型、数据，不创建输出。默认 2000/300/500 图，两位置 FP32 特征约 5.47 GiB，预算 8 GiB。8 个 probe ×32 epochs，最多约 16128 个 optimizer steps；专项训练遇到无关系的 batch 会跳过，真实更新量记在 history。提取同一次 crop 前向同时保存两个位置；数值检查额外处理 3 张图；案例图只读取原图和已存预测，不运行 backbone。

本地没有正式 ADE20K 图像/权重缓存且缺 transformers，未运行正式 all。小型 CPU 测试使用实际作者 full/factor 类和 ImageProcessor，不是新机制证据。

## 实验矩阵

|因素|取值|
|---|---|
|计算位置|L23 post_attention（z）、residual（h）|
|读出器|full_seed0、factor64|
|训练目标|all_pairs BCE、matched_anchor ranking|

共 8 任务，全部在本轮重新训练。每种读出器四个任务共享 seed、图像顺序、采样 patch、epoch/lr/scheduler/weight_decay。history 保存逐 epoch 采样 SHA256，report 核对四个任务一致。均以 validation 上每图等权的距离匹配 anchor 排序选 epoch，平分选最早。不会继续沿用旧 all-pairs accuracy selector，也不把这一基线称为旧 checkpoint 的精确复现。

all_pairs 保留旧 BCE 目标，包括背景、跨类别、跨 crop 对和对角。matched_anchor 用同 crop、同语义类别、同 anchor 的同实例正样本/不同实例负样本；排除背景、未知类、自配对。正负到 anchor 的 patch 网格欧氏距离落入同一固定 2 patch 宽度箱。

专项目标为 `softplus(score(anchor, negative) - score(anchor, positive))`，同 anchor 的全部合格三元组平均，再对支持 anchor 平均，再对 batch 中支持图像平均。无合格关系的图像不参与该次 loss。两目标的监督范围、loss 尺度和有效更新量不同，因此结果不能唯一归因为某个优化机制；不要用两个目标的 train_loss 数值直接比较优劣。

## 指标与判断

主要指标：每图内支持 anchor 等权的距离匹配排序，再对支持图像等权平均，ties 计半分。次要指标采用相同语义/crop/anchor 限制，但不匹配距离。报告负欧氏距离基线、支持图像/anchor/类别/三元组数量，以及同 instance ID 但语义不一致的前景 pair 数（它们从专项关系排除）。这个计数不能证明实例标注整体正确。

14 个配对对比：每种 probe、每种训练目标的 h−z（匹配和未匹配）、每端专项训练−all-pairs，以及两个训练目标局部变化之差。图像配对 bootstrap 10000 次，条件于有支持图像和已拟合 probe，不包含训练 seed 方差、无多重比较校正。无关系的图像不是零分；支持不足应收缩解释，不根据 test 改距离箱。

距离箱只降低空间近邻差异，箱内距离仍可不同；负距离基线能显示残余优势。它不能排除绝对位置、外观和其他空间线索，也不能独立识别真实 binding 或信息删除。

先看匹配后的局部差距是否仍在、覆盖率如何；再看专项训练是否提高后端绝对表现，以及是否也同样提高前端。仅缩小前后差距不足以称为恢复。

## 案例图

每个训练目标/读出器，从匹配后三元组至少 100 个的图像中，分别按固定 seed 抽取最多 6 张 win→loss、loss→win、稳定图像；再均匀抽取一条相应三元组。类别可重叠，同一图可能出现在不同组。这是有意选取的探索案例，不能用展示数量推断失败比例；稳定组包含两端都赢和两端都输。

每张图展示实际 crop、两个 GT 实例 mask、MLP 前后在已采样 token 上的 score rank。cyan 星为 anchor、green 加号为正样本、red 叉号为负样本。只画已有 token，不把稀疏采样插值成完整关系图；两端使用相同 rank 色标，图中 rank 是各端独立排序，不是校准概率。

case.json 保存具体位置、距离、两端原始分数、采样规则；可视化从原图重算 patch mask 与缓存标注核对，不一致则停止。支持不足时减少案例，不放宽阈值。cases/index.json 保存候选图像数与实际导出数。

## 产物与验证

主要产物：

- observation_protocol.json / requested_config.json：固定配置与实现哈希。
- observation/runtime_position_audit.json / cache_audit.json：位置数值检查和缓存身份。大特征只记 size/mtime；标签/manifest 做 SHA256。
- observation/{objective}/{variant}/{channel}/：checkpoint、history、selection、test_predictions、metrics、artifacts。
- observation/report.json：覆盖率、近邻基线、两种训练目标和两端差异的配对区间。
- observation/cases/：PNG、case.json 和 index.json。

现有 get_resulta.ps1 仅下载 JSON/CSV；这一轮审阅还需要带回 observation/cases/ 下的 PNG。
要在本地重算指标，还需带回各任务 test_predictions.npz；无需为看案例下载 5.47 GiB 特征。

完成任务重启时验证文件存在与 SHA256；不完整任务会重新训练，metrics 已存在但产物不完整或改变则报错。report 不读取完整特征或原图，但需要本轮预测和任务产物来保证配对；训练配置/实现变化不能混用结果。

```powershell
& H:\anaconda3\envs\py312\python.exe reproduction/test_observation.py
& H:\anaconda3\envs\py312\python.exe reproduction/observe_binding.py --stage plan
```

测试覆盖距离捷径、跨 crop/背景/跨语义排除、anchor 权重、真实 ranking 梯度、配对/缺支持、8 个实际 probe 小型训练、采样共享、完整任务复用、配置/产物变更拒绝、以及作者实际预处理与案例图导出。
