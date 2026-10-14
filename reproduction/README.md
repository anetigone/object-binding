# 确认实例绑定随深度的变化

## 当前下一轮：新图像上的 L23 确认

完整安排和停止规则见 [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md)。
新增 `confirm_l23.py` 与独立 `config-confirmation*.json`：先排除旧 train/val/test
图像再抽样，显式冻结图像清单/配置/实现哈希，只提取 L23 两个位置，新增 4 个
full/factor64 binding probe。默认 plan 只读；先 freeze，再 all。缓存约 5.47 GiB。
不覆盖探索 run；不自动重新开启预条件、补深度或扩大模型。

```powershell
& H:\anaconda3\envs\py312\python.exe reproduction/test_confirmation.py
& H:\anaconda3\envs\py312\python.exe reproduction/confirm_l23.py --stage plan
```

正式数据/权重/依赖准备好后：

```bash
python reproduction/confirm_l23.py --stage freeze
python reproduction/confirm_l23.py --stage all
```

## 第二关：两端可逆线性预条件

L23 factor64 的 MLP 前后 AUROC 为 0.847471→0.837745，delta=−0.009726，
95% CI=[−0.013677,−0.005496]，局部下降仍存在。下一步使用 `precondition_readout.py`，
固定 `config-precondition-readout.json`：最多新增两个 full binding probe，两端预算一致。

```bash
python reproduction/precondition_readout.py --config reproduction/config-overnight.json --stage plan
python reproduction/precondition_readout.py --config reproduction/config-overnight.json --stage all
```

必须使用服务器实际完成原 run 的配置和原有路径。`all` 顺序为 `fit→train→report`；
可分别用 `--stage fit/train/report`。默认只执行 `plan`，不导入 ML 库、不写输出、不训练。
结果写到原 run 的 `investigation/precondition_readout/`。

预条件矩阵只使用同一组训练图/token：最多 512 图×64 token，固定 ridge_relative=0.01，
不中心化、不截断维度，并保持统计样本平均平方范数。两端各自拟合一个可逆线性矩阵，
作为冻结 buffer 随新 full probe checkpoint 保存；不生成完整变换特征缓存、不运行 backbone。
两个 1024×1024 FP32 矩阵共约 8 MiB；实际统计开销由服务器 CPU/BLAS 和缓存读取速度决定。

复用 `investigation/steps/binding/post_attention/layer_23` 与源 steps 设置中配置的
residual full baseline。正式运行先核验 source settings、原 full readout 配置/预测与两端缓存。
缺原 checkpoint、预测 NPZ 或特征时直接报错，不自动重训或提取。
已完成任务跳过；设置/源结果/实现改变时用新的 `--experiment-output`。

输出 `report.json` 与 `measurements.csv`，四端点共享图像聚类 bootstrap。
主指标为“原始 MLP 差距−预条件后 MLP 差距”，正值表示差距缩小，并报告该量的直接 CI、
两端各自改善及预条件后的局部下降。固定实质效应门槛 0.005，不扫描参数或自动进入下一分支。
成功支持读出几何/优化/有限样本偏置参与差距，不证明全部信息保留或模型功能改善。

```powershell
& H:\anaconda3\envs\py312\python.exe reproduction/test_precondition_readout.py
```

## L23 局部定位的同型低秩读出检查（下一步）

`local_readout.py` 只给 L23 `post_attention` 缓存补一个 `factor64` probe，复用
`validation/readouts/residual/factor64/layer_23`。不会重跑 backbone、提取特征、
训练类别 probe 或自动开始预条件实验。不要跨 full/factor 读出器比较 MLP 两端。

传入服务器实际完成原 run 的配置，保持数据路径和 output 一致：

```bash
python reproduction/local_readout.py --config reproduction/config-overnight.json --stage plan
python reproduction/local_readout.py --config reproduction/config-overnight.json --stage all
```

默认 `plan` 只列出缺失文件，不写文件、不导入 ML 依赖。设置在
`config-local-readout.json`，默认只允许一个新增 binding 任务。`--stage train/report`
可分别训练和生成报告。新增结果写到原 run 下 `investigation/local_readout/`，
已有缓存、原 checkpoint 和结果保持不变；修改设置或代码后使用新的
`--experiment-output`（必须是原 run 内独立目录）。

正式运行前核对已完成 residual 的 `run.json`、模型/data/训练参数、rank、seed、
配对抽样和验证选择规则，审计两端缓存与原预测。缺少服务器特征或原 checkpoint、
预测 NPZ 时直接报错，无提取/重训 fallback；只下载 JSON/CSV 的本地 runs 不足以正式运行。

报告 `report.json` 给出 residual−post_attention 的 hard-instance AUROC 配对图像
bootstrap CI，以及同 crop 的固定 anchor 排序。默认实质效应阈值为 0.005，
只在整个 CI 落入 ±阈值时标注小差异；CI 跨零但较宽时标注不确定。
这些是固定 split、已拟合 probe 条件下的探索结果，不证明信息丢失或原模型功能受损。

本地小型 CPU 测试（不运行预训练模型）：

```powershell
& H:\anaconda3\envs\py312\python.exe reproduction/test_local_readout.py
```

目的：先验证 residual 中实例区分的深度变化，再定位受损的 patch 关系与计算步骤。已有 L18→L23 下降的测量验证后，优先用同模型、同数据的条件分析和末段定位推进机制研究；跨模型/数据集放到解释有信号后。中层峰值、深层类别分组都不是预设结论。

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

审计重算与原提取都先调用同一数值初始化，关闭matmul/cuDNN的TF32并固定cuDNN计算设置；旧审计代码曾漏掉初始化。审计失败先看各层max/RMS/relative-L2误差、patch cosine及版本/精度设置，保持原allclose阈值，不据此自动删除缓存或改容差。

若修改面板、epoch或其他设置，使用新的 `--validation-output` 目录。默认面板预先固定，不根据测试结果继续挑rank/正则化；若方向对合理配置敏感，先研究测量问题再扩展模型/数据集。

默认仍读原有 `index_ade20k.pkl`、JPG和`_seg.png`。已将HF数据处理成该格式时可直接用。不在此修改ADE20K loader。

未来配置 `dataset_factory: "my_module:make_dataset"` 可接另一数据集。factory接收配置，返回：

- `splits`：train/val/test各自的整数图像ID清单，固定、互斥。
- `read(id)`：RGB uint8图像、semantic mask、instance mask、background mask，空间对齐；已知类别正整数、未知类别0，背景单独标记。实例编号无需跨图一致。
- 遵循max_train/max_val/max_test和split_seed，或提供明确固定清单。

多模型要使用相同split和配对；多数据集的类别、实例标注定义需单独记录，不能把各自总体accuracy直接比较。

## 验证完成后的下一轮：条件分析与计算步骤定位

新入口 `investigate_decline.py`，设置在 `config-investigation.json`。必须传**服务器实际完成 validation 的原配置**；不改服务器 ADE20K loader。默认仍是 `plan`，不加载模型、不下载、不训练。所有新报告和读出器在原 run 的 `investigation/` 下，原 `validation/`、checkpoint、requested_config 和 environment 不覆盖。增量特征仍放原 `channels/*/cache/`，便于复用；提取 provenance 单独保存到对应 investigation scope。

先做无新前向、无训练的条件分析：

```bash
python reproduction/test_investigation.py
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --stage conditions
```

默认读取 `validation/readouts/residual/full_seed0/layer_{18,23}` 的验证集选择后预测、`validation/test_geometry.npz`、residual labels 和 splits；不会重新计算几何或读取原图。重新核验缓存签名、预测标签、patch 逐位配对和 crop 标识。源读出器的训练/评估参数必须与传入配置一致。缺文件时明确报错，不退回原主运行的不同 selector。若验证输出位置不同，修改 `baseline_source`（相对 run 的路径）。

- `conditions.json` / `.csv`：hard-instance、前景 binding、同实例 vs 跨类负例、同/跨 crop、近/远、内部/边界、观测到2个/至少3个同类实例的条件内 AUROC，以及 pooled 和逐图 macro 降幅。每组保留正/负样本数、覆盖图像数；缺支持时为空，不把空组计作无下降。
- 实例数量只统计**同 crop 中 saved patches 观测到的同类实例**，是实际数量的下界；不是完整场景竞争量。距离/边界阈值预设在设置文件中。这是探索性分组，没有标注外观相似性或遮挡连续性，不能用这些名称解释输出。
- `anchored_rankings.json`：同 crop 内固定的 anchor/同实例正例/同类不同实例负例三元组，比较两个层的排序正确率，计数 win→loss 与 loss→win，平分 ties，并保存逐图结果。该量对严格单调分数校准不变。三元组共享图像和 anchor，不作独立三元组显著性检验。
- 原始正负 logit 中位数只作辅助；跨层校准不同，不能将它直接解释为“聚拢/排斥”。条件分析并不证明某个因素导致了下降。

需要条件内图像聚类 CI 时再显式执行（CPU 开销明显增加）：

```bash
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --stage condition-ci
```

输出 `condition_ci.json` / `.csv`，沿用配对图像 bootstrap 与分箱误差检查；不具备双标签的重采样组会标记 CI 不可用。多组 CI 是探索性结果，不包含多重比较校正。

随后只补 residual 的 L20/L22，原 L18/L23 缓存复用。新训练沿用验证阶段 full_seed0 的训练 batch/patch 抽样、seed、L2 与 validation hard-instance AUROC selector，首 epoch 赢得 ties。已验证的 L18/L23 通过引用复用，默认仅新增2个 binding 训练任务：

```bash
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --scope depth --stage plan
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --scope depth --stage all
```

`all` 仅执行当前 scope 的 `extract → train binding → report`，不启动条件分析、另一个 scope 或独立类别训练。2000/300/500图、256patch、1024维时，两个新 residual 层增加约 **5.47 GiB**；原32.81 GiB缓存变为约38.28 GiB。实际 plan 根据已有manifest列出缺失项，extract再次核算保留缓存、增量和磁盘余量。

深度定位有信号后，再选择计算步骤对照。默认只对 L23 做三个位置：

- `block_input`：block 输入，norm1 前。
- `post_attention`：norm2 输入，**实际 attention 残差相加之后**，含模型原有 layer scale/drop path，不是单独的 attention 分支输出。
- `residual`：block 输出，MLP 残差相加之后，final LN 前。

```bash
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --scope steps --stage plan
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --scope steps --stage all
```

新增两个位置约 **5.47 GiB**，含depth增量时总计约43.75 GiB；L23 residual的缓存和已验证probe复用，仅新增2个binding训练。`step_layers` 可改为定位出的单个block；不要默认同时展开多个block。当前预算48 GiB，depth后同时缓存L22/L23的两个新位置会超过预算，提取会拒绝。配置更改需指定新的 `--investigation-output`，例如 `runs/phenomenon_overnight/investigation_block22`，原缓存依然复用。

报告 `measurements.csv`、`measurements.png`、`report.json` 分开呈现 hard-instance AUROC、binding分数的category AUROC和独立类别probe。相邻深度/同block计算步骤的 binding 变化附配对图像CI。完成数量和来源明确记录，允许先生成部分报告。不同位置各自训练读出器，所以这些测量定位的是**可解码性变化，不是因果干预**；本轮未实现分支缩放或恢复能力的实验。

独立类别probe只用已缓存特征：已知前景patch上的150类线性分类器，按val宏平均召回选择epoch，报告test accuracy、macro recall、每类支持与逐图计数；不使用binding分数充当语义能力。默认16epoch、1seed，无新的backbone前向。它既不证明类别聚类增强，也不直接衡量实例身份。

```bash
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --scope depth --stage semantic
python reproduction/investigate_decline.py --config reproduction/config-overnight.json --scope depth --stage report
```

steps scope也可单独执行semantic，默认3个任务；depth默认4个。`semantic_num_classes` 是明确的标注taxonomy，接自定义数据时需修改，不能从test标签推断类别数。`max_binding_tasks`、`max_semantic_tasks` 控制任务上限。未执行semantic时图中标记Not run，不将其算作binding流程未完成。

本地只验证CPU合成数据、真实probe类、缓存/配对检查和hook；服务器正式模型的提取与数值一致性仍由服务器运行确认。

## 下一轮：类别变化的 CI 与 MLP 更新中间点

新增 `mlp_update.py`，默认配置 `config-mlp-update.json`。只读取现有缓存和已完成结果，**没有提取阶段，不加载backbone，也不写入大特征缓存**。代码改动、配置与测试都在本地完成；远端同步同一版本，避免直接编辑服务器代码产生分叉。已有 `config-investigation.json` 不改动，因此不会使之前 `depth/steps/requested_settings.json` 失效。

默认 L23、alpha=[0,0.5,1]。两个端点分别引用steps的post_attention与residual读出器（residual binding优先复用validation/full_seed0），不重新训练。中间点逐batch从FP32 mmap计算 `z + alpha*(h-z)`：z为attention残差更新后、h为block输出。只新增**1个binding probe、1个独立类别probe**。初始化、patch/batch采样、优化器、epochs与validation选择规则从已完成 `steps/requested_settings.json` 的设置继承，不用test结果选择alpha。

同步后使用实际完成旧实验的config；默认plan不需要ML依赖，不创建输出：

```bash
python reproduction/test_mlp_update.py
python reproduction/mlp_update.py --config reproduction/config-overnight.json --stage plan
```

先补已有类别结果的CI，不训练任何模型：

```bash
python reproduction/mlp_update.py --config reproduction/config-overnight.json --semantic-scope both --stage semantic-ci
```

`--semantic-scope steps` 只检查L23的attention/MLP两个步骤，`depth` 检查相邻深度及首末层，`both` 检查两者。默认输出 `investigation/mlp_update/semantic_ci_both.json` / `.csv`（文件后缀随scope变化）。源 `depth/steps/report.json`、checkpoint和缓存不改写。

- 成对重采样图像，保留每张图所有已知前景patch；每次先汇总每类support/correct，再计算pooled accuracy与macro recall，**不平均逐图macro recall**。
- 输出两指标的点估计、配对变化CI与bootstrap SE。每类报告support、涉及图像数、两个位置召回差和对总体macro变化的贡献，按绝对贡献排序，仅作探索性诊断，不据此继续挑alpha。
- macro每次对该重采样中有support的类别求平均，稀有类别可能消失；显式报告类别覆盖数区间。CI只反映当前拟合probe和test split下的图像采样不确定性，不包括训练seed波动或多重比较校正。
- 核验source训练/数据provenance、image ID、逐图class support、当前缓存上的固定eval patch选择，以及汇总metrics与逐图计数一致性。新增semantic评估会保存patch/label fingerprint；旧端点无需重跑，使用已记录设置和重建support验证，报告指纹检查的覆盖数量。

再运行中间点（沿用GPU配置，仅训练probe）：

```bash
python reproduction/mlp_update.py --config reproduction/config-overnight.json --stage binding
python reproduction/mlp_update.py --config reproduction/config-overnight.json --stage semantic
python reproduction/mlp_update.py --config reproduction/config-overnight.json --stage report
```

或一次执行当前小面板，默认先做steps类别CI：

```bash
python reproduction/mlp_update.py --config reproduction/config-overnight.json --stage all
```

`all` 为 `semantic-ci → 中间点binding → 中间点semantic → report`，不启动其他模型、补层或backbone前向。已完成中间点跳过。任务上限写在新配置中；增添alpha需显式提高上限并使用新 `--output`。source路径不同可传 `--source-investigation`，所有原始特征仍通过原config的output定位。

新目录保存 `requested_settings.json`、`code_provenance.json`、中间点checkpoint/history/selection/predictions、`report.json`、`measurements.csv` / `.png`。报告同时比较0→0.5、0.5→1和0→1的binding配对CI及semantic配对CI，标记端点复用和完成数量。源metadata保存SHA256，特征文件记录大小/mtime并验证签名和逐位配对，不额外读取几十GiB作完整hash。训练实现或源设置改变时拒绝混用完成的probe；报告单独记录当前代码provenance。

该实验测量缓存表示中MLP更新强度与**重新拟合读出器的可解码性**关系。并未修改完整模型的前向或验证下游行为，不能直接宣称恢复了模型的实例推理能力，也不预设语义覆盖解释成立。

## 范数匹配对照：保留输出方向，匹配 attention 后的尺度

新增独立入口 `norm_control.py`、配置 `config-norm-control.json`，不修改旧alpha配置或已有实验目录。默认只对L23新增一个条件：

```text
matched = h * ||z||_2 / max(||h||_2, epsilon)
z = post_attention, h = post_block, epsilon = 1e-12
```

每个patch独立计算最后一个维度的L2范数，不在图像或batch间汇总，不减均值，不做LayerNorm。范数与缩放使用float64，返回FP32；非零token保留h的方向，范数匹配z。h近零但z明显非零时无法满足方向保持，直接报错，不引入替代方向。近零token计数和数值误差写入审计。

原始z/h的binding与类别probe均引用steps/validation的已完成结果。只新增**1个binding、1个类别probe**，训练设置、抽样seed和val选择规则继承已完成steps；不重训旧alpha中间点。逐batch从现有mmap生成matched特征，没有backbone前向和新的大特征缓存。

代码在本地修改并验证，远端只同步；同步后传服务器实际完成旧实验的config：

```bash
python reproduction/test_norm_control.py
python reproduction/norm_control.py --config reproduction/config-overnight.json --stage plan
python reproduction/norm_control.py --config reproduction/config-overnight.json --stage all
```

默认plan只读、无需ML依赖；all为 `audit → binding → semantic → report`。也可以单独执行同名阶段，完成的probe自动跳过。默认新目录 `investigation/norm_control/`，拒绝把输出设为原run、steps、depth或mlp_update目录；原checkpoint、报告和缓存不改写。`--source-investigation` 可指向自定义steps所在的父目录，`--output` 可改新结果目录。原model/data配置仍从 `--config` 定位。

- `norm_audit.json`：使用**test固定eval patch样本**，报告所有patch和已知前景patch的z/h/matched范数分位数、h/z范数比、范数相对误差、h与matched余弦相似度及近零计数；要求匹配范数满足rtol=2e−6、atol=2epsilon，不扫描整个train缓存。
- 新条件checkpoint/history/selection、测试预测与逐图计数，沿用原训练规则。配置与源文件provenance改变时拒绝混用已完成probe。
- `report.json`、`measurements.csv` / `.png`：呈现z、h、matched三个条件，并对z→h、h→matched、z→matched报告binding配对图像CI、类别accuracy/macro recall配对CI及逐类贡献。允许部分报告，明确完成数量；不按测试结果选择其他epsilon。

若h→matched的读出性能恢复，说明范数成分参与可读出性变化。该对照使用了z的标量范数，可能也传递z中携带的标量信息，**不能单凭恢复就证明纯优化伪影、唯一尺度原因或模型内部信息没有丢失**。若差异仍在，再研究表示方向/几何。所有结果仍是重新拟合probe的缓存测量，不是完整模型的行为干预。

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
