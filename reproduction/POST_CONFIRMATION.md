# L23 确认后的缓存分析

入口：`post_confirmation.py`。设置：`config-post-confirmation.json`。
默认 source 为 `runs/confirmation_l23_v1`，从其 frozen_protocol.json 读取实际模型、
训练和采样配置，无需再手动复制服务器路径到一份新模型配置。
默认输出为源 run 内独立的 `post_confirmation_v1/`，禁止覆盖 channels 或 confirmation。

这一轮是对已看过结果的确认 split 的探索性分析，不能称为第二次独立确认。
没有 backbone 前向、完整模型权重加载或特征提取；模型加载仅限小型 probe checkpoint。

## 运行

本地验证使用指定的 Python 环境：

```powershell
Set-Location G:\Documents\papers\cvpr28
& H:\anaconda3\envs\py312\python.exe reproduction/test_post_confirmation.py
& H:\anaconda3\envs\py312\python.exe reproduction/post_confirmation.py --stage plan
```

在保存着确认实验完整缓存和 checkpoint 的服务器项目根目录执行：

```bash
python reproduction/post_confirmation.py --stage plan
python reproduction/post_confirmation.py --stage all
```

默认 plan 只读、不导入 ML 库、不创建输出，会列出缺失源文件。
all 顺序为：源审计 → cross-readouts → semantic → report。
可以分别执行，所有阶段都先验证源数据/配置/产物和独立输出 provenance：

```bash
python reproduction/post_confirmation.py --stage audit
python reproduction/post_confirmation.py --stage cross-readouts
python reproduction/post_confirmation.py --stage semantic
python reproduction/post_confirmation.py --stage report
```

自定义源目录与新输出：

```bash
python reproduction/post_confirmation.py --source-run runs/confirmation_l23_v1 --output runs/confirmation_l23_v1/post_confirmation_v2 --stage all
```

`--device cpu` 可覆盖 probe 设备，设备写入新输出的 provenance。不要在同一输出目录
中切换设备或修改设置/代码/源文件；这些变化会被拒绝，需要新的 --output。
完成任务自动跳过，metrics 已存在但其他必需产物缺失时直接停止。

## 固定 binding 读出器四格评估

每种已拟合的 full/factor64 probe 固定所有参数，不再拟合：

|训练位置|评估在 post_attention=z|评估在 residual=h|
|---|---|---|
|z|z→z|z→h|
|h|h→z|h→h|

两种读出器共 8 个评估格；零新增 binding 训练。
先加载 checkpoint，并重新计算对角格；要求 score 与确认时保存的 NPZ 逐位匹配，
固定 atol=1e-5、rtol=1e-4。失败停止，不放宽容差，不重新训练源 probe。
其后评估另一位置；共享原 eval patch、image ID 与 pair 条件编码。

报告 5 种预设对比，每种读出器各一组：

- 固定 z probe，表示从 z 换到 h。
- 固定 h probe，表示从 z 换到 h。
- 固定 z 表示，probe 从 z 上训练的换成 h 上训练的。
- 固定 h 表示，probe 从 z 上训练的换成 h 上训练的。
- 两个分别拟合的对角格，即原局部读出差距。

每项分别给出 hard-instance 与 binding-probe 类别关系 AUROC 的图像聚类配对 CI；
10000 repeats、2048 bins、seed=31415、点估计近似误差上限=0.0005，沿用既有方法。
CI 条件于已拟合 probe，无训练 seed 区间、无多重比较校正；anchor 仅给点估计与逐图计数。
这五项是探索性描述，不把某个显著比较独立称为恢复/机制确认。
坐标迁移失败不能证明关系信息丢失，成功也不证明原模型使用该关系。

## 独立类别 probe

两端各一个线性分类器，不复用 binding 分数当作类别指标。
固定 150 类、16 epochs、Adam lr=0.001、seed=0、sampling_seed=0；
训练 batch/patch 预算从源确认配置继承；无额外 scheduler/weight decay。
按 val macro recall 选择 epoch，平分选最早 epoch；测试集不参与选择。
只使用已知类别前景 patch，背景和未知类别排除。

分别报告 pooled accuracy 与 macro class recall，以及后−前的配对图像 CI。
每次重采样汇总 class support/correct 再计算指标；不平均逐图 macro recall。
报告稀有类别导致的 bootstrap 类别覆盖变化、逐类贡献和固定 patch/label 指纹。
指纹与 support 从原缓存重建验证，不能仅凭两端支持数相同就声称配对正确。

新增训练任务严格限定为 2 个；没有 alpha/rank/ridge 搜索或训练预算扩展。
不能直接用不同任务的百分点差值断言 binding 特异性。

## 源审计和产物

正式阶段要求源确认主要门槛通过，并核对冻结协议与 report 的哈希、split、
原 run/config/variant、验证 epoch 选择、预测重算 AUROC、缓存签名/header、
图像/位置/标注逐位对齐、所有冻结记录中的 reproduction/作者源码哈希。
旧原图、完整模型与数据 index 不读取；源提取正确性依赖原数值审计和缓存 provenance。
缓存大特征只记录 size/mtime，不额外全文件哈希；小源文件/预测/checkpoint 做 SHA256。

输出：

- `requested_settings.json`：源文件哈希、实现哈希、设置、依赖版本、特征 size/mtime。
- `cross_readouts/{variant}/{fitted}_to_{evaluated}/`：metrics、per_image、预测 NPZ。
- `semantic/{channel}/layer_23/`：run、checkpoint、history、selection、metrics、per_image。
- `report.json`：完成度、四格结果、配对变化 CI、anchor 计数、独立类别 CI/逐类贡献。
- `cross_matrix.csv`、`semantic_summary.csv`：便于快速读表。

report 可以展示部分完成状态，明确预期 8 格和 2 个类别任务，不把缺结果填成零。
只有下载 JSON/CSV 的本地目录不足以执行正式阶段；缺 NPZ/checkpoint/特征直接报错，
没有自动重新提取、重训源任务或下载权重的 fallback。

## 本地验证

小型 CPU 测试使用实际 full/factor64 类、实际评估和类别训练流程；
验证相同端点零差异、不同端点的 checkpoint/feature 两个轴、对角复现失败、
类别指纹、来源缺失、完成状态恢复、设置变化拒绝、主要门槛与输出目录边界。
测试会对所有源文件比较前后 SHA256，确保后续分析没有改写原结果。

本次使用 H:\anaconda3\envs\py312 验证：新测试 6 项、确认实验回归 5 项、
已有语义区间/MLP 回归 7 项全部通过，共 18 项。plan、Python 编译检查与
git diff --check 通过。本地 plan 列出 20 个缺失的特征/labels/checkpoint/预测文件；
因此尚未执行正式四格评估或类别训练，应在持有完整源产物的服务器运行。
