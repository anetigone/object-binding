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
