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

已有文件可放在项目根目录：

```text
project/
  reproduction/
  ade20k_hf/                 已处理的数据
  hf_cache/                 已有HF模型缓存
```

`dataset_root` 指向 `ade20k_hf` 的绝对路径；`hf_cache` 可在JSON配置中填绝对路径。未指定 `hf_cache` 时，默认是 `reproduction/hf_cache`。如果不改配置，可将已有缓存放在默认位置或用符号链接复用。数据名不会决定格式：默认接口要求root中有 `index_ade20k.pkl`，或root下有 `ADE20K_2021_17_01/index_ade20k.pkl`。只有HF的`dataset_dict.json`和Arrow文件的目录不能直接传给这个默认接口，需使用服务器已有的读取factory。

HF缓存root通常包含 `models--facebook--dinov2-large/{blobs,refs,snapshots}`；应将这个root设为 `hf_cache`，不是它的某个snapshot。模型的缓存revision应与配置一致。数据集相关下载缓存可以共用HF目录，提取后的特征仍在runs下。

## 较长一轮的配置

`config-overnight.json` 比较12/18/23层的residual/Q/K/V，2000/300/500张图、1seed、quadratic、32epochs，StepLR在16epoch衰减。缓存32.81GiB、12任务、约24192 optimizer steps，预算48GiB。它是增加训练预算的配置，**不是保证7–8小时完成的时间预测**，需按实际GPU速度估算。改变epoch/scheduler使用新输出目录，不能与16epoch结果混称同一训练设置。

在项目根目录修改该配置的 `dataset_root` 和 `hf_cache` 后：

```bash
python reproduction/phenomenon.py --config reproduction/config-overnight.json --stage plan
mkdir -p logs
nohup bash -c 'python -u reproduction/phenomenon.py --config reproduction/config-overnight.json --stage all && python -u reproduction/phenomenon.py --config reproduction/config-overnight.json --stage interactions' > logs/overnight.log 2>&1 &
```

确认所有所需依赖与模型revision已在本地后，可以 `export HF_HUB_OFFLINE=1`，缺少权重时直接报错而非下载。观察日志中的提取速度和训练epoch速度，再估算剩余时间；不要为凑满时长直接展开全量矩阵。

较长配置现在默认 `offline: true`，通过 `local_files_only` 禁止权重联网检查。缓存必须包含配置指定的revision。QKV hook同时支持旧的嵌套query/key/value及新版直接q_proj/k_proj/v_proj结构，未支持结构会在提取前报错且不留下部分hook。可视化也直接hook block输出，避免不同版本hidden_states元组约定影响层号。模型实现版本写入environment.json；跨版本的数值一致性仍需另行验证。

## 先验证 L18–L23 的下降，不扩大模型数量

新入口 `validate_decline.py` 只消费现有run；不会提取新的特征，也不修改原有checkpoint。默认只检查residual两层。其他通道先不自动展开。

```bash
python reproduction/validate_decline.py --config reproduction/config-overnight.json --stage plan
python reproduction/test_validation.py
python reproduction/validate_decline.py --config reproduction/config-overnight.json --stage all
```

配置沿用服务器实际完成的那份config，其 `output` 必须指向已有的 overnight run。`config-validation.json` 是独立的小面板设置。结果写到原run下的 `validation/`，不占新的大特征缓存。也可分阶段运行 `audit`、`verify-positions`、`bootstrap`、`readouts`、`geometry`、`report`。

1. **逐图配对bootstrap**：同一重采样图像的全部patch对一起进入两个层，报告 pooled pair AUROC 的L23−L18与95% CI，B=10000。不把patch对视为独立样本。为节省CPU，CI采用固定分位数直方图近似，点估计仍使用精确排序；输出近似误差，超过0.0005会停止并要求增加bins。它不是精确排序bootstrap，也不是逐图AUROC的macro平均；大量配对的图像仍有更大权重。
2. **计算位置检查**：audit核对manifest和预测配对；verify-positions额外对3张测试图像的实际crop做前向，逐向量对照现有缓存，确认post-block与投影位置，并检查当前实现/预处理是否与缓存一致。前向只做少量审计图像，不建立新缓存。失败应先解释版本/预处理差异，不继续读出器实验。
3. **有限读出器面板**：full seed0、full L2=1e−4、author factor width64/128、full seed1，以及同一final LayerNorm作用于两层的full seed0，共12个训练任务。所有新任务固定训练batch/patch抽样seed，初始化seed可变；统一按验证集hard-instance AUROC选epoch，测试集只在选择后评估。这与原run按总体val accuracy选epoch的口径不同，不能直接当作原checkpoint的重复测量。作者factor宽度r对应对称W秩至多2r，不宣称rank=r。L2作用于参数，full与factor参数化的惩罚不等价。
4. **几何差异检查**：只补读测试标注；使用真实crop像素中的实例面积比例与4邻域边界像素比例，避免把抽中patch占比误称真实面积。检查正/负例距离、平均实例面积、最大端点边界比例的分布，并在粗联合分层内比较两层AUROC，用同一组overlap权重汇总。跨crop距离没有统一坐标，明确排除并报告其数量；小样本/无重叠cell也排除并报告覆盖率。分层是探索性诊断，不是充分控制所有混杂或因果证据。

输出 `paired_bootstrap.json`、`position_audit.json`、`runtime_position_audit.json`、`readout_sensitivity.csv`、`geometry_checks.json`、`test_geometry.npz` 和 `status.json`。readouts各自保存checkpoint/selection/history/metrics/per_image/test_predictions；用于bootstrap的原预测不覆盖。LayerNorm对照只额外加载一次冻结权重并保存小量LN参数，不对整个数据集前向。

若修改面板、epoch或其他设置，使用新的 `--validation-output` 目录。默认面板预先固定，不根据测试结果继续挑rank/正则化；若方向对合理配置敏感，先研究测量问题再扩展模型/数据集。

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
