# 方法与实现对应

依据收到的 Overleaf `main.tex` 中顺序单杆主方法及正文方法清单整理。原稿与原实验源码均未改写。本文仅解释代码，不代替论文正文。

## 主方法

`src/search_proposers.py::bom_proposal` 以全部 morphology/reward 历史拟合一个 8D Matérn GP，在 incumbent 上固定另外 7 根杆，每次开放第 `step % 8` 根杆并以 EI 提案。`run.py` 构建候选几何、PPO 训练与评价、记录所有结果，并根据最高已测 reward 选择 incumbent。无 sensitivity、Joint selector、TR、CBTR 或多路径。

GP 常量核幅值、长度尺度、噪声水平在代码中固定；每步用 1024 个 Sobol 候选近似 acquisition 最大化。几何不可行候选记录拒绝原因，不偷偷缩放以挤入墙缝。

## 消融

- `random_rod_bo`：相同 GP/EI 与预算，均匀、有放回随机选杆；不是整条形态向量随机采样。
- `global_bo`：所有 8 维开放，其他规则保持一致。
- `fixed`：共享 nominal 控制器，不搜索形态。
- `fixed_pair_2d`：仅为附录旧版保留，默认不进入正文实验组；与 cyclic Joint-2D 不是同一算法。

## 外部基线任务适配

- RoboMorph：LLM 从 8 维合法范围与已测结果生成杆长。未实现官方完整 grammar / topology mutation。请求失败不冒充另一个算法。
- Text2Robot：固定八杆拓扑上的 EA（早期随机初始化、4 个精英、交叉或高斯长度变异）。未实现 text-to-3D mesh pipeline，不声称官方完整复现。
- Transform2Act：ring message-passing 网络、属性与控制两阶段、动作/概率/熵掩码、可学习 attribute/control actors。未实现变拓扑 skeleton transformation；设计阶段不推进物理时间，PPO transitions 与真实物理控制步数分别记录。

## 环境与指标

56D = 8 角度 + 16 杆中心位置 + 16 杆中心速度 + 8 接触标记 + 8 法向力加权接触高度。4 个受控 actuator；XML 中其余 actuator 保留但 control 置零。

Reward 为 rootz 在一个控制间隔的竖直速度；完成度为最终 COM 相对初始 COM 的增高除以 1 m 后截断到 [0,1]。`sampled_height_hold_success` 是末尾 14 次控制采样内均达到 1 m 且接触墙面，不是原 F 盘 strict-success 判据。控制间隔 0.04 s，一条 rollout 225 个控制步。

实际原始杆长并非稿件早期的等杆 45 度构型。Stepped / Curved 等为当前已有场景编译器的明确分段几何，不声称恢复了作者所有原始场景 XML。收到的辅助 XML 与现用物理模型分别保留，不混用来冒充同一实验。

## 验证边界

移交代码包使用独立新入口，从本机旧路径/开发 supervisor 中解耦；训练、评价、候选、场景和 T2A 主要实现来自已存在源码。所有原源码保持不变。源文件摘要见 `source_extraction.json`。

能运行的方法定义不等于已有该方法的完整结果。实际软件测试、跳过项、是否产生 API 调用，均记录在根目录 `VERIFICATION.json`。
