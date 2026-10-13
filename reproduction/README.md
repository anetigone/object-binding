# 确认实例绑定随深度的变化

目的：先检验 residual 中“同实例”与“同类别不同实例”的分离是否随深度非单调变化，再对照 Q/K/V。跨模型、跨数据集重复出现后，才进入理论与因果实验。中层峰值、深层类别分组都不是预设结论。

本次修改没有运行预训练模型或正式训练。现有 ADE20K 数据读取接口保持不变；服务器自行处理的版本没有被本地覆盖。**主入口改为 `phenomenon.py`**，训练/评估复用 `run_server.py`。

## Git 同步

服务器 clone 自己的仓库后，在项目根目录执行：

```bash
python reproduction/bootstrap_upstream.py
python -m pip install -r reproduction/requirements-server.txt
python reproduction/test_phenomenon.py
python reproduction/check_numerics.py
```

使用已有 CUDA PyTorch 环境，torchvision 与其匹配。bootstrap 获取作者固定 commit `014c66b45ea262f9b6eec83ff388a1e1c10dfcaa`，作为忽略的依赖目录，避免嵌套 Git 仓库混入自己的源码；有上游本地改动时拒绝覆盖。

后续同步用 `git pull --ff-only`，不再用旧 zip。数据、权重、缓存及运行输出均忽略；以前已跟踪的 zip/partial 需要从索引移除，ignore 本身不能取消跟踪。运行记录自己的代码 commit、模型 resolved commit 和配置；在干净且已提交的代码上正式运行。

默认使用新目录 `runs/phenomenon_dinov2l_ade`，不删除服务器已有结果。旧缓存不是schema2，需要留在原目录，重新跑新入口。

## 第一轮：residual

```bash
python reproduction/phenomenon.py --stage plan
python reproduction/phenomenon.py --dataset-root /datasets/ade20k --stage all
```

**默认阶段是 plan**：不加载模型、不下载数据、不启动训练。先检查计划，再显式运行。

默认 train/val/test=2000/300/500，层0/6/12/18/23，256 saved patches、每轮采样64、quadratic probe、1 seed、16 epochs、FP32。缓存约 **13.67 GiB**，5个任务，约5040 optimizer steps。数量是上限估计，提取前按实际split重新核算。

可分阶段 `--stage extract/train/report/visualize`，保持配置和输出目录一致。**所有所选层同一次前向提取**，hook只搬运选中的patch。多crop逐个前向，保留作者518短边/14patch预处理。

加seed/probe无需重新提取，已完成任务跳过；加层/通道只补缺失特征，但仍需对图像重新前向。改变模型、图像划分、patch数或预处理时拒绝旧缓存，需要新输出目录。中断的split会重建缺失层，不将半成品标记为完成。

`cache_budget_gib`、`max_tasks` 为显式上限，超限拒绝运行；加seed时也要有意调整任务上限。提取检查已有缓存+增量与实际磁盘余量，不自动清理已有缓存。墙钟时间用服务器实际速度估算，计划不猜测耗时。

## 第二轮：按需对照 Q/K/V

```bash
python reproduction/phenomenon.py --config reproduction/config-qkv.json --stage plan
python reproduction/phenomenon.py --config reproduction/config-qkv.json --dataset-root /datasets/ade20k --stage all
python reproduction/phenomenon.py --config reproduction/config-qkv.json --dataset-root /datasets/ade20k --stage interactions
```

只比较12/18/23层的 residual/Q/K/V，12个任务；首轮已完成的residual跳过。所选特征32.81 GiB；保留首轮其他两层时总计 **38.28 GiB**，预算48 GiB。QKV新增24.61 GiB，同一次前向保存三个通道及所选层。

- residual[l]：block l输出，未过模型最终LayerNorm。
- q/k/v[l]：block l的norm1(input)经过实际query/key/value投影的输出，所有head拼接，未经过attention。
- 同block不同计算位置，不能当作同一张量；编号从0开始。

`interactions` 用缓存测各head对称pre-softmax `QKᵀ/sqrt(head_dim)`，只在val选择head，再报告该head test AUROC；所有head数据也保存用于探索。没有训练额外probe，不把原始0阈值当作校准分类阈值。少量缓存patch不能重建完整attention softmax分母，因此不称它为attention概率。

Q/K/V上的二次probe差异不足以证明信息迁移：它们是归一化token的线性投影，重参数化、probe容量也影响解码。这里只确认信号和交互的深度变化，不作因果结论。

## 第三轮：扩展一个模型，不展开大矩阵

```bash
python reproduction/phenomenon.py --config reproduction/config-base.json --stage plan
python reproduction/phenomenon.py --config reproduction/config-base.json --dataset-root /datasets/ade20k --stage all
```

已实现DINOv2 small/base/large的384/768/1024维、12/12/24层，probe随宽度变化；base示例比较0/3/6/9/11层，缓存10.25 GiB。报告保留block编号和归一化位置。small/base解析并记录HF commit，正式比较应将它固定到配置。

这只是同训练家族规模检查，不是训练目标对照。CLIP/MAE等需增加模型adapter及预处理验证，目前明确拒绝未实现模型。

## 数据接口保持不变，扩展单独接入

默认仍读原有 `index_ade20k.pkl`、JPG和`_seg.png`。已将HF数据处理成该格式时可直接用。不在此修改ADE20K loader。

未来配置 `dataset_factory: "my_module:make_dataset"` 可接另一数据集。factory接收配置，返回：

- `splits`：train/val/test各自的整数图像ID清单，固定、互斥。
- `read(id)`：RGB uint8图像、semantic mask、instance mask、background mask，空间对齐；已知类别正整数、未知类别0，背景单独标记。实例编号无需跨图一致。
- 遵循max_train/max_val/max_test和split_seed，或提供明确固定清单。

多模型要使用相同split和配对；多数据集的类别、实例标注定义需单独记录，不能把各自总体accuracy直接比较。

## 主指标与结果

优先看两个不依赖固定阈值的任务：

1. `instance_vs_same_class_auroc`：已知类别前景中，同实例正例 vs 同类别不同实例负例。
2. `class_grouping_cross_instance_auroc`：仅在不同实例中，区分同类与跨类，避免类别指标被同实例正例撑高。

分别报告样本量；缺少正/负例时AUROC为空。前景balanced accuracy、同实例召回、同类/跨类FPR继续报告，阈值logit>0。不能只凭FPR升高宣布类别分组增强。

输出根目录：`requested_config.json`、`resource_plan.json`、`splits.json`、`environment.json`、`phenomenon_metrics.csv`、`phenomenon_curves.png`、`report_status.json`、`observations.md`，按需加`qk_interactions.json`。

各通道位于 `channels/{residual,q,k,v}`，内含cache和原来的probe六件套、residual固定query图。seed阴影是训练seed标准差，不是图像置信区间。

`test_predictions.npz` 增加image_id及条件编码：0背景、1已知类别同实例、2同类不同实例、3跨类不同实例、4其他前景。`per_image.json`保存每个子组TP/FP/正负计数，可供逐图配对bootstrap。

报告允许部分完成，但标记完成度；QKV报告比较三层，首轮五层数据仍保留。旧`config-full.json`不作为默认筛查项，新入口因预算拒绝它。

作者来源：https://github.com/liyihao0302/vit-object-binding ，论文：https://papers.neurips.cc/paper_files/paper/2025/file/050b8ff31bee2dfea65b731e71baccd5-Paper-Conference.pdf 。沿用公开源码口径；论文文字512裁剪再pad与当前源码518多crop的差异仍需记录。
