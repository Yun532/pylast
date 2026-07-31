# LACT_sim 数据字段、pyLAST 层级与画图

本文说明 LACT_sim 的相机输出怎样进入 pyLAST，以及画图时实际使用的是哪一层数据。

## 1. 探测器处理顺序

当前无波形积分模式的主链为：

```text
Cherenkov p.e. + NSB p.e.
          |
          v
饱和前总输入（primary p.e.）
          |
          v
SiPM 微单元饱和
          |
          v
最终积分图像（fired p.e.）
          |
          v
ROOT observations.image_pe
          |
          v
pyLAST event.dl0.tels[tel_id].image
```

Cherenkov 真值同时走一条独立的诊断路径：

```text
饱和前 Cherenkov p.e.（不含 NSB）
          |
          v
ROOT observations.image_cherenkov_pe
          |
          v
pyLAST event.simulation.tels[tel_id].true_image
```

因此，`image_pe - image_cherenkov_pe` 不能简单解释为 NSB。`image_pe`
还经过了非线性的 SiPM 饱和；只有在关闭饱和时，这个差值才可直接对应加入的
NSB。

## 2. 字段与数据层级对应

| 文件字段 | 含义 | pyLAST 对应 | 推荐用途 |
| --- | --- | --- | --- |
| ROOT `observations.image_pe` | Cherenkov 与 NSB 合并后，再经过 SiPM 饱和的最终积分图像 | 无波形时为 `event.dl0.tels[tel_id].image` | 默认相机图、清洗和重建输入 |
| ROOT `observations.image_cherenkov_pe` | 饱和前的 Cherenkov 真值，不含 NSB | `event.simulation.tels[tel_id].true_image` | 真值诊断；读取时会四舍五入为整数 p.e. |
| ROOT `observations.image_nsb_pe` | 可选的饱和前 NSB 分量 | 当前不映射到标准 pyLAST 层级 | NSB 分量诊断 |
| HDF5 `/images/dense/pe` | 经过 SiPM 饱和的最终积分图像 | `LactEventSource` 不直接读取 HDF5 | HDF5 直接诊断 |
| HDF5 `/images/dense/primary_pe` | 饱和前的 Cherenkov+NSB 总输入；只在启用饱和时写出 | `LactEventSource` 不直接读取 HDF5 | 饱和损失诊断 |
| HDF5 `/images/dense/cherenkov_pe`、`nsb_pe` | 可选的饱和前分量 | `LactEventSource` 不直接读取 HDF5 | 分量诊断 |

这里的“真值”必须区分：

- ROOT `image_cherenkov_pe` 是 **Cherenkov-only truth**；
- HDF5 `primary_pe` 是 **Cherenkov+NSB 的饱和前总输入**。

两者在启用 NSB 时不是同一个量。

## 3. 无波形和有波形的区别

当 LACT ROOT 没有 `waveforms` 树时，`LactEventSource` 直接把
`observations.image_pe` 写入 DL0。这是当前 LACT_sim 用户配置的默认方式：

```python
from pylast.io import LactEventSource

source = LactEventSource("lact_events.root", max_events=-1)
event = source[0]

image = event.dl0.tels[tel_id].image
truth = event.simulation.tels[tel_id].true_image
```

当文件包含 `waveforms` 树时，读取器先建立 R1 波形，DL0 由后续标定得到。
此时不能再把 ROOT 的积分 `image_pe` 与标定后的 DL0 假定为逐像素完全相同。

## 4. 画图层级

LACT ROOT 快速绘图接口默认使用 DL0，也就是无波形模式下的
`observations.image_pe`：

```python
from pylast.visualize import plot_event_quicklook

result = plot_event_quicklook(
    "lact_events.root",
    output_dir="plots",
    event_index=0,
    image_level="dl0",
)
```

直接使用 `EventVisualizer` 时也建议显式写出层级：

```python
from pylast.visualize import EventVisualizer

visualizer = EventVisualizer(source)

# 默认物理分析图：饱和后的 image_pe
visualizer.plot_event(event, image_level="dl0")

# 真值诊断图：饱和前、无 NSB 的 image_cherenkov_pe
visualizer.plot_event(event, image_level="simulation")
```

不要依赖通用绘图函数的默认值。pyLAST 同时服务于 LACT ROOT、simtelarray
等输入，不同接口的历史默认层级可能不同；显式指定 `image_level` 最安全。

## 5. NSB 应该在哪里加入

### 推荐：在 LACT_sim 中加入

生产模拟应在 LACT_sim 中启用 NSB。这样处理顺序是：

```text
Cherenkov + NSB -> SiPM 饱和 -> 触发 -> image_pe -> pyLAST DL0
```

pyLAST 读取 DL0 后不应再次加入 NSB，否则会重复计数。

### pyLAST 现有的近似

pyLAST 的 `ImageProcessor` 有 `image_processor.poisson_noise`。它对
`simulation.true_image` 逐像素加入固定均值的 Poisson 噪声，并用于假触发和
算法压力测试。随后会扣除该固定均值，结果保存在 simulation 层的
`fake_image` 中。

这个功能不是 LACT 探测器响应模拟，原因包括：

- 输入是 Cherenkov-only `true_image`，不是 LACT_sim 的饱和前总输入；
- 没有光子到达时间和积分窗结构；
- 不执行 LACT_sim 的 SiPM 微单元饱和模型；
- 不复现 LACT_sim 的真实相机与阵列触发顺序；
- 不会把结果替换成 LACT ROOT 已提供的 DL0 `image_pe`。

因此它可以用于重建算法的统计测试，但不能替代 LACT_sim 中的 NSB。

若以后要在 pyLAST 内实现等价的探测器级 NSB，至少需要饱和前
`primary_pe`（或逐 p.e. 到达时间/波形）、NSB 率和积分窗，并重新执行饱和与
触发。对当前无波形 ROOT，只有 `image_pe` 和 Cherenkov-only 真值，信息不足以
严格反演并重做这条链。
