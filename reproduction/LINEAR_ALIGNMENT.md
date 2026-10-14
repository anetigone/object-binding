# L23 原始产物复核与线性对齐

入口：`linear_alignment.py`，设置：`config-linear-alignment.json`。
研究过程记录在项目根目录的 `research_notes/2026-10-14-l23-progress.md`。
旧确认与后续分析文件不改写。新输出在 `runs/confirmation_l23_v1/linear_alignment_v1/`。

## 执行顺序

本地仅凭已下载 JSON/CSV 即可复核记录与类别计数：

```powershell
Set-Location G:\Documents\papers\cvpr28
& H:\anaconda3\envs\py312\python.exe reproduction/test_linear_alignment.py
& H:\anaconda3\envs\py312\python.exe reproduction/linear_alignment.py --stage review
& H:\anaconda3\envs\py312\python.exe reproduction/linear_alignment.py --stage plan
```

review 保存 `research_notes/2026-10-14-l23-review.json`，列出可核验的源文件和缺件，
从逐图 class counts 重算独立类别 CI。该阶段不宣称 binding 原始预测已复核。
plan 只读、不导入 ML 库、不创建实验输出。

同步代码后，在有完整特征、标签、checkpoint 和预测数组的服务器项目根目录执行：

```bash
python reproduction/linear_alignment.py --stage plan
python reproduction/linear_alignment.py --stage all
```

all 顺序：audit → fit → evaluate → report。可以单独运行：

```bash
python reproduction/linear_alignment.py --stage audit
python reproduction/linear_alignment.py --stage fit
python reproduction/linear_alignment.py --stage evaluate
python reproduction/linear_alignment.py --stage report
```

默认读取确认 run 的冻结配置与 `post_confirmation_v1` 的实际设置，无需复制旧路径。
可用 --source-run、--source-followup 指定已有目录；--output 必须是源 run 内独立目录。
--device 可改变评估设备，但设备/代码/设置变化需要新的输出目录。

## 原始产物门槛

源确认主要门槛和后续完成状态必须通过；缺任一必需原始文件直接报错，
没有重提取/重训/下载 fallback。拟合前保存成功且与当前 provenance 对应的 raw_audit.json：

- 核验原配置、split、模型/采样/probe identity、源文件哈希和缓存逐位配对。
- 原 full/factor64 四个 checkpoint 在自己的位置复现逐位 binding score。
- 从原预测 NPZ 重算两个局部差值的 AUROC/10000 次配对 CI，核对原确认报告。
- 从八个交叉预测 NPZ 重算 AUROC，并核对固定图像/patch/条件编码。
- 两个独立类别 checkpoint 复现逐图 class_correct、support、fingerprint；重算类别 CI。

这里没有重算交叉读出的全部对比 CI，raw_audit 明确标记 cross_ci_recomputed=false。
对原几何提取仍依赖已有数值审计；不读取原图或加载完整 backbone。
大特征文件用缓存签名/header/size/mtime 核验，小文件/预测/checkpoint 用 SHA256。

## 固定假设与拟合

检验：训练集上的线性坐标对齐，能否改善冻结 z-probe 在 h 上的实例关系读出？
z=MLP 前 post_attention，h=MLP 后 residual。

只拟合一个无截距、不中心化的行向量映射 hB：

```text
G = mean_train(hᵀh)
C = mean_train(hᵀz)
lambda = 0.001 * trace(G) / d
B = solve(G + lambda I, C)
```

固定 statistics_seed=20261014，最多 512 张 train 图，每图 64 个 paired saved token。
使用所有抽中 token，不使用实例/类别标签；不看 val/test 来选择 ridge、token 或图像。
两端抽取相同图像/patch/位置；记录训练 ID、token 数及采样指纹。
FP64 统计/求解，实际评估使用 FP32 B，报告求解残差与矩阵哈希。
不要求 B 可逆，不做非线性、均值平移或额外范数调整。

统计内存主要是两个 1024×1024 FP64 矩阵约 16 MiB；FP32 B 约 4 MiB。
映射逐图/批次按需计算，不建立新大特征缓存；零 backbone 前向、零 probe 训练。
CPU/BLAS 统计与求解时间按实际速度确定，不承诺固定耗时。

## 评估与判据

full 为主要冻结读出器，factor64 为次要。各报告四个条件：

1. 原 z，冻结 z-probe，作为参考。
2. 原 h，冻结 z-probe，作为 identity 迁移基线。
3. hB，仍使用同一个冻结 z-probe。
4. 原 h，上次重新拟合的 h-probe，作为有限读出预算的上下文对照。

主要指标为条件3−条件2的 hard-instance AUROC；配对图像 95% CI 下界 >0.005
才标记 meaningful_migration_improvement。factor64 不能替代未通过的 full。
同时给条件3−条件1（剩余参考差距）、条件3−条件4，及同 crop anchor 排序。
不能只凭显著改善宣称回到 z 水平；CI 跨零也不等于两端相等。

独立类别对照也冻结 z 上已拟合的线性分类器，在原 h 与 hB 上评估，报告
accuracy/macro recall 的配对图像变化，以及相对 z 的剩余差距。
不使用 binding probe 的类别分数代替独立类别指标。
重建指标使用固定 eval tokens，在 val/test 分别报告 identity MSE、mapped MSE、
相对 z 的平方误差和“预测训练 z 均值”的常数基线；这些数据不用于选择映射。

所有正式评估都保留预测/逐图计数。输出包括 raw_audit.json、requested_settings.json、
alignment.npz、fit_audit.json、binding/、semantic/、reconstruction.json、report.json、measurements.csv。
代码/参数/源结果改变拒绝混用旧输出；完成映射与评估跳过，不完整产物拒绝静默复用。

## 解释边界与停止规则

该 split 已经用于决策，本轮是固定设计的探索实验，并非新独立确认。
CI 条件于 fitted probe 与 fitted map，仅包含图像采样，不包括映射拟合/训练 seed 波动，
没有多重比较校正。直方图 bootstrap 参数固定；误差超限时停止，不自动调整。

成功说明线性对齐改善外部读出迁移；不证明原模型行为改善、所有信息保留或纯几何唯一解释。
对 full bilinear probe，s(hB,hB) 仍是 h 上的 full bilinear 函数，映射不扩大其函数族。
这轮把表示重建作为额外训练先验，不是公平比较 probe 优化上限的证明。
低 MSE 但关系不改善，不算关系恢复；映射失败不证明信息删除。
不自动扫描 ridge、扩大训练样本、加入截距或进入非线性重建。

## 本地验证

使用 H:\anaconda3\envs\py312；合成 CPU 测试执行真实源 checkpoint 审计、训练 token
拟合、冻结 binding/类别评估与配对报告。检查已知线性关系、来源缺失、审计前禁止拟合、
拟合不读取 val/test、源文件不变、参数改变拒绝复用及映射哈希保护。
小型测试不产生正式机制结果。

本轮使用指定 py312 环境：线性对齐测试 4 项、四格/类别回归测试 6 项全部通过，
共 10 项；本地 review 重算类别 CI 成功，plan/编译/diff 检查通过。
