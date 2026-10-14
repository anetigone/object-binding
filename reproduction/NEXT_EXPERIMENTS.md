# 下一轮实验：先确认 L23，再区分读出几何与关系线索变化

安排日期：2026-10-14。所有 L 编号从 0 开始；L23 是 DINOv2-L 最后一个 block。
问题限定为同类不同实例关系的可解码性。新 split 的结果尚未产生。

## 已有实验不重复

本地已找到 depth、steps、mlp_update、norm_control 的报告及测量表。
L20/L22 补点、alpha=0.5 和独立类别读出已经有结果摘要，不能再列为尚未执行。
alpha=0/0.5/1 的 full binding AUROC 记录为 0.828678/0.813466/0.809123；
独立类别 accuracy 为 0.753380/0.748658/0.741863。这些仍是旧探索 split 的记录，
不作为新确认集的先验保证，也不支持“binding 下降意味着语义增强”。
本地没有对应特征、预测 NPZ 或 checkpoint，未独立复算这些数值。

## A：新图像确认，首先执行

入口 `confirm_l23.py`；模型/训练配置 `config-confirmation.json`；
判据/排除清单 `config-confirmation-checks.json`。

|项目|固定安排|
|---|---|
|模型|DINOv2-L，revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c，冻结 backbone|
|计算位置|L23 post_attention（norm2 输入）与 residual（block 输出、最终 LN 之前）|
|图像|2000 train / 300 val / 500 test；三个 split 都排除所有旧 split 的图像|
|读出器|full seed0 为主要检验；factor64 seed0 为预设次要稳健性检验|
|预算|每个读出器两端同预算，32 epochs、Adam lr=0.001、weight_decay=0、StepLR 16/0.2|
|采样|每图存 256 patch、训练/评估取 64；训练 sampling_seed=0；两端共享抽样|
|选择|只按 val hard-instance AUROC 选 epoch，平分选最早 epoch|
|主要指标|hard-instance pooled pair AUROC，delta=MLP 后−前|
|主要判据|full 的图像聚类配对 95% CI 上界 < −0.005|
|区间|10000 次图像聚类 bootstrap；保持整张图的全部 pair；固定直方图近似|
|次要结果|factor64 的同型前后差、固定同 crop anchor 排序、binding probe 的类别关系分数|

bootstrap 点估计用精确排序；CI 用 2048 个固定分位数 bins。
若点估计近似误差 >0.0005，停止报告；不能偷偷扫 bins。
若确需更改设置，建新实验并明确其属于方法修订，不继续宣称原方案完全未变。
当前区间仅包含图像采样不确定性，不包含 probe 初始化/训练或 split 波动。
500 test 图沿用旧预算，并非事先完成了功效计算；宽 CI 应报告为不确定。
不把 pooled pair AUROC 称为逐图 macro AUROC。

主检验只有 full；factor64、anchor 和类别关系结果不替代失败的主检验。
anchor 当前报告点估计、转换计数和逐图计数，不给三元组独立性检验。
binding probe 的类别关系 AUROC 不等于独立类别 probe 的 accuracy/macro recall。
不直接比较两种任务的百分点变化来断言 binding 特异性。

### 冻结与数据边界

先将配置中的 dataset_root、hf_cache 指向服务器的真实数据和固定权重缓存。
如果服务器使用自定义 dataset_factory，必须在新配置中保留同一 factory；它需要在
max_train/max_val/max_test=0 时返回完整的合格候选池，不允许仅返回旧抽样结果。
自定义 factory 另设 confirmation_index_file，指向决定图像 ID 含义的稳定索引文件。

checks 的 exclusion_split_files 默认包含 `runs/phenomenon_overnight/splits.json`。
若任何其他 run 的图像参与过训练、验证、测试或研究决策，都追加其 splits.json。
仅更换随机 seed 不足以保证新图像；代码先排除旧 ID，再抽样，数量不足直接停止。
ID 去重假设所有清单使用同一数据索引。旧清单的索引来源需要人工核对；
当前旧产物未保存索引哈希，代码无法追溯证明旧 ID 对应相同内容，亦未做跨 ID 重复图像检测。

在完成本地检查、同步并提交代码后，先运行 plan，再显式 freeze。
freeze 只读取索引/候选池，不提取图像特征、不读取新 test 指标；保存固定图像清单、
配置、排除清单哈希、实现/作者源码/索引哈希和主要依赖版本。
后续拒绝变更。all 不自动 freeze，也不调用主入口的默认 train（其 selector 不同）。
运行输出必须是独立新目录；旧结果和缓存保持可追溯。

```bash
python reproduction/test_confirmation.py
python reproduction/test_validation.py
python reproduction/test_local_readout.py
python reproduction/confirm_l23.py --stage plan
python reproduction/confirm_l23.py --stage freeze
python reproduction/confirm_l23.py --stage all
```

all 顺序：extract → verify-positions → train → report。
也可以分别执行这些阶段；train 要求三张新 test 图的缓存位置数值审计通过。
verify-positions 是既定的机械数值审计，不用于选择超参数。
完成任务跳过；有 metrics 但缺 checkpoint/预测/history 等必需文件时直接停止。
最终报告在 `runs/confirmation_l23_v1/confirmation/report.json`。
保存全量特征、checkpoint、预测 NPZ、selection/history 和逐图结果，不能只取 JSON/CSV。

### 成本与停止规则

新图像需要一次新的 backbone 提取；同一次前向同时捕获两个位置。
每个 crop 单独前向。FP32 特征约 5.47 GiB，预算 8 GiB，不是显存要求。
训练新增 4 个 binding probe、约 8064 optimizer steps；3 张图额外前向做位置审计。
不猜运行小时数；按实际提取/训练速度估计。

- full CI 上界 <−0.005：确认在固定读出器/预算下存在实质下降，进入 B。
- CI 上界 <0 但未越过 −0.005：仅确认负向变化，不把实质下降门槛算通过。
- CI 完全位于 ±0.005：支持小差异；停止把旧下降作为已确认的主要现象。
- CI 宽且跨零：不确定，不解释为等价，不自动扩样本或调参。
- 方向相反，或 factor64 与 full 不一致：如实报告，收缩主张至读出器依赖。

## B：确认后，先复用缓存做低成本分解

以下是后续设计，尚未由新入口自动执行；不得将方案当作机制结果。

1. **条件分析**：对旧探索集的 L23 两端，以及确认集预先规定的分组，使用真实
   crop 位置/边界/面积；分开同 crop 与跨 crop、近远距离、内部边界、观测同类实例数。
   报告支持数、覆盖图像数、pooled 与逐图宏平均。已有 conditions 入口比较 L18/L23，
   不能不改源端点就称为 L23 的 MLP 条件定位。旧图像上的分组用于探索；
   看完确认结果再新增分组也要标为探索，不借本次确认集宣称新的机制确认。
2. **固定读出器交叉应用**：缓存保持不变，对每种已拟合的 probe 做 z→z、z→h、
   h→z、h→h 四格评估。零新增训练/前向，固定 patch 和条件。它能描述坐标/分数
   的迁移敏感性；失败仍不能证明信息丢失，成功也不证明原模型使用该信息。
3. **独立类别指标**：如果主要门槛通过，在同一缓存两端各训一个线性类别 probe，
   固定旧设置：150 类、16 epochs、lr=0.001、seed/sampling_seed=0，按 val macro recall
   选 epoch。共 2 任务、零新前向，单独报告 accuracy/macro recall 的配对图像 CI。
   此项要使用独立子实验 provenance，不能修改已冻结确认目录的协议或输出。

不重复 L20/L22、alpha=0.5 或标量范数匹配；先审阅并复核已有完整产物。
不自动扫描 ridge/rank/alpha、延长预算、扩大模型或数据集。

## C：有明确线索后再做恢复/行为实验

只有 B 指出可检验的原因，才冻结下一项设计：训练集上拟合 h→z 的线性重建对照与
一个预设的小非线性重建器，冻结 z 上的 probe；用独立图像同时报告重建误差、
hard-instance AUROC 和 anchor 排序。后端关系表现改善是恢复证据的必要部分，
不能只看“前后差距缩小”。未恢复也不能证明信息不存在。

完整模型行为干预需先指定确实依赖实例关系的下游任务及独立评估；当前只有外部
probe 流程，不能把缓存插值、交叉读出或重建叫作模型实际使用实例线索的证据。

## 本地验证边界

使用 `H:\anaconda3\envs\py312\python.exe`，Python 3.12.11、torch 2.9.1+cu126、
numpy 2.0.2；环境检查显示 CUDA 可用。该环境缺 transformers，工作区也没有 ADE20K
原图/权重缓存或原特征/预测/checkpoint，故本地不运行正式提取与确认实验。
合成 CPU 测试执行实际作者 full/factor probe、实际训练/评估函数，验证共享采样、
读出器输出、配对报告、冻结/排除/不完整产物检查；通过不代表新现象得到确认。
测试使用 AST 加载真实函数/类，沿用现有测试方式，不需 transformers 或模型权重。
正式环境依赖见 requirements-server.txt，须记录 torch/transformers 等版本。

本次本地验证结果：test_confirmation.py 5 项、test_validation.py 9 项、
test_local_readout.py 3 项全部通过；新入口 plan 通过，预算内；Python 编译检查与
git diff --check 通过。确认测试包含两种真实 probe 的小型训练、共享采样、配对报告、
排除/冻结与不完整产物拒绝；没有运行 DINOv2 正式前向。
