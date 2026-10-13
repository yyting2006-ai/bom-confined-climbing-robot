# 原实物机构控制口径

## 结论

- 机构在 MuJoCo 中有 **8 个转动关节、8 路 actuator 槽位**。
- 实物只有 **4 个主动电机**。
- 原始控制代码始终只给 actuator 索引 `0, 2, 4, 6` 非零命令；
  `1, 3, 5, 7` 保持为零。
- 四个实物动作依次映射到：
  `1_joint, 111_joint, 11111_joint, 1111111_joint`。

因此面向**现有四电机实物**的学习策略动作维度是 4，而不是 7 或 8：

```text
[a0, a1, a2, a3]
  -> [a0, 0, a1, 0, a2, 0, a3, 0]
```

## 本轮执行器选择实验

用户允许同时比较 4 电机和 8 电机方案，并以严格力学成功率选择主结果：

- `four_motor`: 保持现有实物接口，4D 动作按上述方式映射；
- `eight_motor`: 同一八关节机构的八路 actuator 全部受控，8D 动作直接映射。

二者必须使用同一个模型、成功谓词、训练种子池和开发种子池。若八电机方案胜出，
论文必须把它标为需要增加四个主动执行器的设计方案，不能宣称可直接部署到当前四电机实物。

原始证据：

- `../爬墙机器人/2.xiafeng_test/climb_control_project_debug_success_ENV_1/control.py`
  中的八维控制数组全部遵循上述交替零模式。
- 同目录 `ENV_1_M_0.8_chuizhi.xml` 声明八个 actuator。

冻结原始证据 SHA-256：

- `control.py`: `88177aa5baaac3165bd2bb2a78e79832ef0eeeffaf4c6a880ced4b167ef03af9`
- `ENV_1_M_0.8_chuizhi.xml`: `2034627b4272a35b3d20688fce8b5357ff194225adca7ae35ab9ea6a0fc6df60`

论文和训练统一表述为：**8-joint simulation model with a 4-dimensional
hardware action space**。不能把八个仿真关节误写成八个实物电机，也不能把闭环中
去掉根关节后的七维研究支线误写成实物控制接口。

`parallel_7joint_ppo_*` 是在收到实物四电机信息前建立的模型审计支线，
不得作为论文主控制器或硬件部署接口。
