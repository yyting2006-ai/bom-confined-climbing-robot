# Bayesian Morphology Optimization：论文算法与实验代码

本仓库提供论文方法对应的算法、模型、控制训练、评价及实验入口。仓库只含源码、配置和小型预检文件，不含训练检查点和完整实验数据。默认主方法为 **顺序单杆 BOM**；正文消融为 **Fixed、Random BO、Global BO**。Joint、CBTR、复杂 selector 不在默认实验中。

论文《Bayesian Optimization of Morphology for a Confined Climbing Robot》的正式实验配置是 `configs/confirmatory_root_contact4ms_onset180.json`。其代码流程为：8 维高斯过程拟合杆长与表现，逐轮只在一根杆长上用期望改进选择候选，为候选训练 PPO 控制器，再用独立重置评估选中方案。`SOURCE_PROVENANCE.json` 记录了论文 PDF 和冻结实验源码的哈希。此公开仓库提供复现实验所需的代码，不包含论文中的大规模训练产物。

检查论文配置和全部软件单元测试：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python run_matrix.py --config configs/confirmatory_root_contact4ms_onset180.json --output runs_confirmatory --workers 3
```

第二条命令只打印运行计划。完整训练需要在命令末尾加 `--execute`，预算为 300 次控制器训练、共 9 亿次控制转移。

## 1. 内容

| 论文方法 | 命令名称 | 实现 |
|---|---|---|
| BOM (Ours) | `bom` | 全 8D GP；每步只在一个杆长的 1D slice 搜索 EI；按 0→7 循环 |
| Random BO | `random_rod_bo` | 每步均匀随机选一根杆，再在该杆的 1D slice 做 BO；有放回，不是 Random Search |
| Global BO | `global_bo` | 同一个全维 GP，在全部 8 根杆长上搜索 EI |
| Fixed | `fixed` | 不做形态搜索，只训练/复用共享初始控制器并评价 |
| RoboMorph | `robomorph_llm` | 真实 LLM 根据已测 reward 提议 8 根杆长；任务适配版 |
| Text2Robot | `text2robot_ea` | 精英选择、交叉、变异的固定拓扑杆长 EA；任务适配版 |
| Transform2Act | `transform2act` | 分阶段 attribute/control PPO、ring graph 网络、阶段动作掩码；任务适配版 |
| 附录旧版 Two-Rod | `fixed_pair_2d` | 只优化固定的一对杆，保留为可选项，不列入正文默认消融 |

三个外部方法是本任务的固定拓扑适配实现，不是官方完整代码复现；具体范围见 `docs/METHOD_ALIGNMENT.md`。

## 2. 安装与检查

建议 Python 3.11，Windows 为当前验证平台。克隆仓库后在仓库根目录运行：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python tests/test_methods.py
python preflight.py
```

Linux/macOS 虚拟环境激活改为 `source .venv/bin/activate`；其他系统未在本次验证。必须用 MuJoCo **2.3.3**，原模型使用 global-coordinate MJCF，不可直接换成新版 MuJoCo。

`preflight.py` 只做四场景短仿真、观测/动作和奖励检查，不训练 PPO、不调用 API。

## 3. BOM 单次运行

先打印配置，不启动训练：

```bash
python run.py --method bom --scene parallel --repeat 1
```

正式启动：

```bash
python run.py --method bom --scene parallel --repeat 1 --execute
```

默认 3M PPO steps / morphology，8 个新候选；6 次搜索评价、10 次独立确认；4 场景 × 3 repeats。初始形态只评价一次形成初始数据；全部候选结果进入历史；只用表现最好的形态作为后续搜索中心。EI 使用 1024 个 Sobol 候选近似最大化，不声称连续全局精确求解。

换方法只修改 `--method`。同一 output、scene、repeat 的 BOM/Random BO/Global BO/EA/LLM/Fixed 共享初始控制器和评价 seed。

短流程检查使用单独输出目录，结果不是论文数据：

```bash
python run.py --config configs/software_smoke.json --method bom --output smoke_runs --execute
```

这会执行少量真实 PPO 训练，不会生成论文中的完整实验结果。

## 4. 批量实验

```bash
python run_suite.py --group ablation
python run_suite.py --group ablation --execute
python run_suite.py --group non-api --execute
```

- `ablation`：BOM、Fixed、Random BO、Global BO，共 48 个单元。
- `non-api`：排除 LLM 的 6 方法，共 72 个单元。
- `baseline`：四个基线/主方法，共 48 个单元。
- `all`：4 场景 × 3 repeats × 7 方法，共 84 个单元。
- 每次最多 1 个训练进程；启动检查物理可用内存至少 2 GiB、提交余量至少 2.5 GiB；训练中保留低内存退出保护。
- 可在 output 中建立 `STOP_REQUESTED` 文件，批处理会在当前单元结束后停止；移除后再执行同一命令可继续。不要同时运行多个入口。
- 输出目录绑定代码哈希、配置和 device，已有协议不一致时拒绝混用。若修改方法/配置，请新建 output。
- 普通中断后运行目录如残留 `.runner.lock`，先确认原进程已退出，再手动删除该锁。不要移动已有运行目录，检查点引用使用该运行目录的绝对路径。
- 恢复加载已保存的模型与 normalizer，但不声称恢复了模拟器和所有 RNG 状态的逐位连续训练。

## 5. RoboMorph / DeepSeek

使用可选的 LLM 基线时，在使用者自己的终端设置密钥：

```powershell
$env:DEEPSEEK_API_KEY = "在本机填入自己的密钥"
$env:DEEPSEEK_MODEL = "deepseek-chat"
python run.py --method robomorph_llm --execute --allow-api --max-api-calls 16
```

请求上限累计记录在本输出目录的 `api_log/requests.jsonl`，失败请求也计数。未显式启用时不会请求 API；失败不回退成随机搜索或其他方法。LLM 只输出候选，不提供实验 reward。模型名称由使用者按其可用接口配置。

## 6. Transform2Act

```bash
python tests/test_transform2act.py
python tests/test_transform2act.py --ppo-smoke
python run.py --method transform2act --scene parallel --repeat 1 --execute
```

默认总预算 27M PPO transitions = 3M × (1 + 8)，同时记录设计决策与真实控制步数。此方法是联合设计/控制训练，不按“8 次独立形态重训”解释。Fixed 只有共享初始控制器的 3M 预算，不是算力完全匹配的 27M control-only 基线。

## 7. 冻结与重训

默认 `paper_retrain.json` 是当前论文程序的 **56D 观测、4D 动作、±10% 杆长、每形态 PPO 训练** 版本。

如需在相同 56D 环境中另做冻结控制器实验：

```bash
python run.py --method bom --controller-mode frozen --output runs_frozen56 --execute
```

先训练共享初始控制器，搜索时固定它。该开关**不是**此前 F 盘 37D、±25%、17 初始观测的旧实验复现。不要用这个新运行输出覆盖旧图表结果。Transform2Act 的联合训练不能使用 frozen 模式。

CPU 为默认。装有兼容 CUDA 的 PyTorch 时可显式传 `--device cuda`；MuJoCo 物理仿真仍在 CPU，未在本包中承诺 GPU 加速倍数。

## 8. 模型与结果

- `models/`：实际运行的机器人 XML。
- `src/parallel_eight_rod_morphology.py`：严格保持目标杆长的闭环几何生成器。
- `src/scene_builder.py` 与 `paper_env.py`：Parallel / Tilted / Stepped / Curved 场景构建与统一控制环境。
- `preflight_output/`：本次短仿真构建的四个 nominal XML 及检查记录；不是性能实验结果。
- `runs/`：运行后产生候选、XML、模型/normalizer、search evaluation、confirmation、状态轨迹等。

完成实际运行后汇总：

```bash
python collect_results.py runs
```

只汇总已有结果，缺项不填零。本仓库不含训练检查点、密钥、后台调度器或自动重启任务。论文中的定量结果需要另行运行完整实验矩阵才能复现。
