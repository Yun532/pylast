# 实测 LACT 数据入口

`pylast.io.LactRealEventSource` 把实测 ROOT 转成现有的 `ArrayEvent`；
`pylast.calib.LactRealCalibrator` 产生 DL0 图像，随后继续调用原有
`ImageProcessor`、重建器及 `EventVisualizer`。参考配置为
[`config/real_data.json`](../config/real_data.json)。默认单台望远镜的数据键为 0，
绘图显示编号为 1。旧脚本中的 `LACT1EventSource`、`LACT1Calibrator` 名字保留为别名。

## 输入与预处理

支持两种输入，均需与实测相机一致的 mapping CSV：

| 输入 | 内容 | 行为 |
| --- | --- | --- |
| `eventShow` | 每事件的 `map_number/HG_LG/base/amp/area` 等向量 | 直接读取已有摘要；`PeakPos` 存在时保留峰值采样点 |
| `waveforms` | 每电子学通道/增益一行，包含 `event/FEE_ID/channel/HG_LG/waveform[256]` | 按事件组装双增益相机，保留原始波形 |

原始 `waveforms` 有两种明确的预处理方式：

- 不传 `analysis_file`：使用服务器传统独立 L1 提取。8 段各 32 点中选择
  `mean + 3*RMS` 最小段作为基线；使用连续 5 点高于 `10*RMS` 的脉冲门限、
  LocalGrad 排尖峰及 10% 振幅边界环绕积分。
- 传入匹配的官方 `analysis_file`：按 event ID 使用其 `eventShow` 的
  `base/amp/area/PeakPos`，同时保留输入原始波形，并核对时间及通道一致性。
  这用于精确接回已完成的官方预处理。

两种结果可能不同。当前参考官方摘要使用外部 ML `effectivePulse` 标签，
其中有效脉冲还会经过三点 `[1,2,1]/4` 平滑。传统 L1 不生成这些 ML 标签，
也不冒充相同处理。对已处理的 `eventShow` 直接作为主输入即可，不再传 `analysis_file`。
原始输入允许按 FEE 等顺序交错存储事件通道，会先按事件号建立行索引；
重复或未映射的通道、同一事件中不一致的 Rabbit 时间戳会报错。

## 标定、时间与指向

有 LED 标定文件时，读取 `led_cal` 的 `event_time`、`GainFactor_AreaH/L`，
选择事件时间最近的标定条目，用面积除以相应通道系数。事件超出标定时间覆盖范围时报错。
低增益峰值大于配置中的 `gain_threshold_mv`（默认 36 mV）时优先选低增益；
缺失或系数无效的通道不会作为有效电荷使用。

参考文件的 LED `event_time` 使用原始 Rabbit 时标，并非已经转换好的 UTC：
其中两条时间分别与实测事件 352009、352462 的
`RabbitTime + Rabbittime*8e-9` 精确相同。因此默认配置同时设置
`source.tai_minus_utc_s=37` 和 `calibrator.led_time_offset_s=-37`，
把两者都转成 UTC 后再匹配。`led_time_offset_s` 的含义是加到 LED 秒时间上的偏移；
若另一个 LED 文件本来已是 UTC，应设为 0。自定义源数据的 TAI−UTC 偏移时，
若 LED 仍使用相同原始 Rabbit 时标，也须对应调整 LED 偏移。
旧参考 pylast 仅转换事件时钟，会错配约 37 秒；此处修正了该问题，
不声称与旧错误配对下的电荷逐值相同。实测证据见
[`led_time_audit.json`](../../real_data_plot/validation/led_time_audit.json)。

`eventShow.area` 已由扣除基线后的波形积分得到，不能再减一次 `base`。
当前两份样本还混有全相机照明的 LED 事件，LED 事件得到接近全相机亮图是预期现象。
源将官方 `LED_event` 保留为 `event.is_led`（`True` / `False`）；缺少标识的 raw
输入为 `None`，不能当成非 LED。标识随 ROOT 输出保留，LED 图会明确注明。
可用 `event.is_led is False` 挑选已确认的非 LED 事件，不自动丢弃记录。

不提供 LED 文件时，只使用参考 nominal peak 换算：高增益峰值 `/2.5`、
低增益峰值 `*10.2`；这适合检查图像，不代表完成了 LED 面积标定。
此模式只产生 DL0，不产生 R1，原始 mV 波形仍保留在 `event.c1`。
有 LED 且输入包含原始波形时，才建立所选增益的 R1 逐采样点电荷。
官方摘要可能使用平滑及脉冲边界积分，因此整段 R1 求和不能替代官方 DL0 面积摘要。

`event.c1.tels[0].peak_sample` 保存采样点索引。当前采样间隔尚未核实，
默认 `sampling_interval_ns=0`，物理 `peak_time` 保持缺失；波形横轴使用 sample，
不会标成 ns。只有确认采集设置后，才设置正的 `sampling_interval_ns`。
Rabbit 的细时间戳间隔为 8 ns，它不是波形采样间隔。

联合方向投影需要实测指向。可以传 `pointing_file`，其中需要 `MJD`/`mjd`，
以及方位、俯仰列；当前日志的 `方位当前角度`、`俯仰当前角度` 可直接读取，
也支持 `az_deg/alt_deg` 等列名。事件须在日志覆盖范围内，方位跨 360° 时会先展开再插值。
也可以同时提供 `pointing_alt`、`pointing_az`，单位均为度。

## 最小 API

```python
import json
from pathlib import Path
from pylast.io import LactRealEventSource
from pylast.calib import LactRealCalibrator
from pylast.image import ImageProcessor
from pylast.visualize import EventVisualizer, plot_image_detail

config = json.loads(Path("config/real_data.json").read_text())
source = LactRealEventSource(
    "build.root", "mapping-v6.csv",
    analysis_file="analysis_build.root",       # 仅 raw + 官方摘要模式需要
    pointing_file="encoder_data.txt",
    config_str=config["source"],
)
event = source[0]                              # 此处是事件位置，不是 event ID
LactRealCalibrator("led_cal.root", config_str=config["calibrator"])(event)
ImageProcessor(source.subarray, config_str=json.dumps(config["image_processor"]))(event)
plot_image_detail(event, tel_id=1, visualizer=EventVisualizer(source),
                  output_path="event_detail.png", show=False)
```

绘图只消费已经标定、处理的事件，不再重算图像或 Hillas。相机细节、联合投影、
KM2A/WCDA 单独绘图及 sidecar 匹配规则详见
[`real_data_visualization.md`](real_data_visualization.md)。

## 按 event ID 运行示例

在已构建并可导入本仓 pylast 的环境中，从仓库根目录执行。下面使用本地
2026-02-14 的第二份样本中有双方匹配的非 LED 事件，省略了依赖机器的绝对路径：

```bash
python scripts/real_data_quicklook.py \
  --input ../real_data_plot/build_20260215064950.root \
  --mapping-file ../real_data_plot/mapping-v6.csv \
  --analysis-file ../real_data_plot/reference_data/20260214/analysis/analysis_build_20260215064950.root \
  --calibration-file ../real_data_plot/reference_data/20260214/led/LC01_analysis_build_20260214212003_led_cal.root \
  --pointing-file ../real_data_plot/reference_data/20260214/pointing/2026021415_encoder_data.txt \
  --km2a-file ../real_data_plot/reference_data/20260214/km2a/analysis_build_20260215064950.rec.root \
  --wcda-file ../real_data_plot/reference_data/20260214/wcda/analysis_build_20260215064950.merge.root \
  --event-id 352455 --output-dir real_quicklook
```

`352455` 的官方 `LED_event=false`，且 KM2A、WCDA 都有接受的匹配。
其中 KM2A 的方向、核心重建值无效，仅绘可用命中并注明缺失；WCDA 方向与
Hillas 长轴约相差 3.239°，旧脚本中也有同样偏差。时间匹配不等于方向一致，
程序保留原记录，不通过旋转图像强行对齐。
相反，352012、352052、352441 均为 LED 事件，不适合作为物理簇射图示。
若要挑选联合事件，使用 `list_matched_event_ids(km2a_root_file=..., wcda_root_file=...)`
查看 `KM2A`、`WCDA`、`both` 列表，再把目标 ID 传给脚本。
两种 sidecar 必须属于同一 LACT build 文件，不能仅按事件数字跨文件配对。

脚本输出 raw/clean/detail 图；有匹配记录时附加联合 core 图，以及相应探测器的
单独图。无匹配的探测器会明确跳过。KM2A 单独图显示 ED/MD hit 电荷；
当前 WCDA sidecar 没有 cell hit 数据，其单独图显示匹配的方向和核心位置。
用 `--pointing-alt 70 --pointing-az 12` 可以替代 `--pointing-file`；
去掉 `--analysis-file` 运行传统 L1；去掉 `--calibration-file` 仅做 nominal peak 检查。
默认不需要、也不应猜测 `--sampling-interval-ns`。

这份 LED 文件转换到 UTC 后，仅覆盖第二份样本至事件 352462；后续
352463–352478 共 16 事件超出其标定覆盖范围，需要后续标定文件，不能静默外推。
上面的 352455 示例位于覆盖范围内。

## 后续重建与写出

清洗及 Hillas 参数继续使用现有 `ImageProcessor`。实测事件没有 MC truth，
重建配置应使用 `use_fake_hillas=false`。当前单台样本不能独立给出有效的立体
方向与核心位置；立体重建无效是预期结果，不能用 MC 值补齐。
能量/PID 模型需核对其输入字段、单位、训练范围及所需重建是否有效，
不会因为完成了读取和标定就自动适用于这台实测相机。

已有 `hillas_reco` 的 `--input-format real` 路径接入同样的 source/calibrator，
可继续用 `--mapping-file/--calibration-file/--analysis-file` 及实测指向参数。
无指向时该处理路径只保存标定图像，跳过 shower reconstruction。

直接使用 native `DataWriter` 时，需要解包 Python source：

```python
from pylast.io import DataWriter

writer = DataWriter(source._source, "real_dl.root",
                    config_str=json.dumps(config["data_writer"]))
try:
    writer(event)                              # 写已处理的同一个事件
finally:
    writer.close()
```

实测配置关闭 simulation shower、simulation config、simulated camera 写出，
不调用 `write_all_simulation_shower`。读写入口的区别到此结束，后续处理仍使用
原有事件数据层和绘图接口。

## 本地验证记录（2026-10-20）

在本地 WSL 编译后，对 `build_20260215064901.root` 和
`build_20260215064950.root` 的 238 个事件逐项核对了原始波形与官方摘要配对。
其中 119 + 103 = 222 个事件完成独立 LED 电荷计算对照、现有清洗/Hillas、
ROOT 写出再读回及时间戳、LED 标记检查；另外 16 个事件明确报出标定覆盖范围外。
两份数据共含 159 个 LED 事件和 79 个非 LED 事件。单台立体重建无效符合预期。
同时通过 61 项 Python 测试、4 项 CLI 子测试、4 项 C++ 回归及 7 项独立波形检查。
详细逐文件结果位于
[`validation.json`](../../real_data_plot/validation/validation.json)，示例图位于
`../../real_data_plot/validation/quicklook/`。

服务器代码参考快照及文件校验记录位于
[`REFERENCE_NOTES_20261020.txt`](../../real_data_plot/reference_code/REFERENCE_NOTES_20261020.txt)。
该快照来自实际服务器 `PrototypeCalibration` 工作树；GitHub `calibration_dev`
的分支 HEAD 已核实，但未将它与服务器快照认作相同版本。
