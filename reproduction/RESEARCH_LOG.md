# 研究记录

## 2026-10-14：L23 的同型 factor64 局部检查

范围：DINOv2-L、ADE20K、当前固定 split 与配对。来源：
`runs/phenomenon_overnight/investigation/local_readout/report.json` 和 `requested_settings.json`。

|读出器|MLP 前 AUROC|MLP 后 AUROC|后−前|95% CI|
|---|---:|---:|---:|---|
|full|0.828678|0.809123|−0.019554|[−0.028699,−0.010809]|
|factor64|0.847471|0.837745|−0.009726|[−0.013677,−0.005496]|

factor64 的 500 图、2000 次图像聚类配对 bootstrap 全部有效。CI 整体低于预设
−0.005 门槛，但上界距离门槛较近。局部下降在这两种读出器下成立；幅度减半不意味着
“一半信息损失来自 probe”，也未对两种读出差值之差作显著性检验。

固定 anchor 排序：0.821713→0.813620，delta=−0.008093。552,968 个三元组，
415 张图有支持；212 张下降、157 张改善、46 张相同。win→loss=32,036，
loss→win=27,561。由逐图计数补算 pooled 排序差值的图像聚类 bootstrap
（2000 次、seed=31415）：95% CI=[−0.013203,−0.003599]。
三元组不独立；补算是探索分析，不包含训练不确定性。

本地核验：新增 provenance 的配置、variant、抽样 seed 与 residual 原记录一致，
source run.json SHA256 匹配，逐图 anchor 计数汇总匹配。新增本地文件只有
report.json/requested_settings.json，没有新训练历史、checkpoint 或预测数组；
未独立核验训练曲线或重算 AUROC。

保留结论：L23 MLP 后实例关系可解码性下降在 full 和 factor64 下均成立，并伴随
固定 anchor 排序下降；下降幅度依赖读出参数化。尚不能判断信息丢失、binding 特异性，
或原模型是否依赖这条信号。

## 下一关：预先固定的两端可逆线性预条件

入口 `precondition_readout.py`；配置 `config-precondition-readout.json`。
只新增两个 full binding probe，两端采用原训练预算、seed、抽样和验证 selector。

最多 512 张训练图×每图 64 token，以无标签固定抽样估计不中心化二阶矩 S。
两端使用相同样本，固定 lambda=0.01*trace(S)/d，各自拟合
T=g(S+lambda I)^−1/2；g 保持统计样本平均平方范数。不中心化、不截断、不扫描参数。
T 的实际 float32 矩阵经过可逆性/精度审计，并作为冻结 buffer 写入新 checkpoint。

主指标 N=(AUC(raw z)−AUC(raw h))−(AUC(Tz z)−AUC(Th h))，正值表示局部差距缩小。
四端点共享图像重采样，直接报告 N 的 95% CI，以及各端改善与预条件后的局部差距。
实质效应门槛固定 0.005；不自动增加训练或进入其他分支。

若差距明确缩小，支持优化/有限样本/参数化几何参与差距，下一步需独立图像确认。
否则结束本轮，不继续自动搜索 rank/ridge/归一化。失败不能证明信息删除。
可逆线性变换不扩大 full 对称双线性函数族；成功也不能唯一归因为纯优化或解释为
完整信息保留、原模型功能改善。当前 test split 已用于探索。

预算：零 backbone 前向、零完整变换特征缓存，两个 1024×1024 FP32 矩阵共约 8 MiB。

## 第二关结果：未得到恢复，差距缩小主要来自 MLP 前读出恶化

来源：`runs/phenomenon_overnight/investigation/precondition_readout/` 的 report、
requested_settings、两端 audit/run/history/selection/metrics JSON。

|条件|MLP 前 AUROC|MLP 后 AUROC|后−前|95% CI|
|---|---:|---:|---:|---|
|原始 full|0.828678|0.809123|−0.019554|[−0.028699,−0.010809]|
|预条件 full|0.811064|0.802832|−0.008233|[−0.013818,−0.002479]|

四端点共享 500 张图、2000 次图像聚类 bootstrap，全部有效。
差距缩小 N=+0.011322，95% CI=[+0.002538,+0.019960]。
CI 排除零，但不能确认 N 大于预设的 0.005 实质效应门槛；自动 assessment 为
`inconclusive_at_prespecified_effect`。更关键的是，差距缩小不是恢复：

- MLP 前：预条件−原始=−0.017613，95% CI=[−0.023954,−0.011150]。
- MLP 后：预条件−原始=−0.006291，95% CI=[−0.013937,+0.001523]。
  点估计下降，未检测到改善；CI 跨零不能解释为保持不变。

预条件后的局部下降 CI 仍排除零，但不能确认下降大于 0.005。
不能用差距缩小本身声称成功恢复，或声称原始下降已经被解释为纯优化问题。

审计记录：两端均用同样的 512 张训练图、32,768 token，图像 ID 全部属于 train，
样本 fingerprint 相同；不中心化、不截断。实际 float32 T 的条件数约 173.92/117.60，
训练统计 trace 比均约 1，矩阵相对误差约 2.7e−8/2.4e−8。
二阶矩极值比由约 8.8万/3.1万降至约 2.91/2.27；统计样本的各向异性明显降低，
仍未得到后端实例关系读出的恢复。

两端新 run 的 config/settings 与 provenance 一致，transform hash 与审计一致；
metrics 与四端点报告匹配；现存原始 metrics/selection/history 哈希匹配。
两端选中 epoch=32，末段验证 AUROC 仍有上升，不能声称读出训练完全收敛。
末尾训练 BCE 与原始相近或更低，也不能用 BCE 代替 hard-instance 泛化表现。
本地没有 transform.npz、checkpoint 或预测 NPZ，以上为记录一致性核验，
未独立重算预条件矩阵与 AUROC/CI。

决策：结束固定预条件的恢复分支，不自动延长训练、扫描 ridge、增加模型或进入非线性恢复。
两关后仍可保留稳健的读出下降与局部定位，尚未形成机制解释和恢复的证据闭环。
恢复分支以 MLP 后关系表现改善为必要证据，差距缩小只作配套指标；
一次预条件失败不能推广为所有线性坐标调整均无效，也不能证明信息丢失。
