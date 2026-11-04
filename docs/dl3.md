# 实测重建到 DL3 事件组件

导出接在现有重建之后，读取已经保存的事件及 prediction，不重新标定、清洗、移动像素或运行模型。
FITS 和 ROOT 保存等价的标量表；原来的多级 ROOT 继续用于回看 DL0/DL1 像素图和波形。

当前状态始终是 **EVENTS_ONLY**，不是完整科学 DL3：真实 GTI、活时间、观测模式及匹配 IRF
尚待补齐。新示例目录 dl3_release_v1 表示本次导出结构的版本，不表示已经具备曝光、流量或能谱分析条件。
规范参考：[GADF EVENTS](https://gamma-astro-data-formats.readthedocs.io/en/v0.3/events/events.html)、
[GADF 时间](https://gamma-astro-data-formats.readthedocs.io/en/v0.3/general/time.html)、
[LSTchain DL2→DL3](https://cta-observatory.github.io/cta-lstchain/introduction.html#dl2-to-dl3)。

## 文件里的表

| 表 | 一行的含义与关系 |
|---|---|
| EVENTS | 一个事件；主方向、能量和 ACCEPTED 始终来自本次明确指定的 LACT 重建 |
| IMAGE_PARAMETERS | 一个事件的一台望远镜；以 OBS_ID、EVENT_ID、TEL_ID 关联，不包含像素数组 |
| POINTING | 全部输入事件各自时刻的指向采样；不随 ACCEPTED 筛选缩短 |
| GTI | 仅在提供真实有效观测区间时存在，不从事件首末时刻或时间间隔推断 |
| SCHEMA | 每张表的每列 dtype、单位和说明，包含外部 RAW 字段的限制 |
| PROVENANCE | JSON 来源记录，与 ROOT 的 metadata JSON 内容相同 |

没有有效 DL1 图像的事件可以没有 IMAGE_PARAMETERS 行，不能用零值补出虚构图像。
图像参数事件键必须属于导出的 EVENTS，且同一 OBS_ID、EVENT_ID、TEL_ID 不可重复。
POINTING 从全部输入事件生成，保留原事件键及 TT 时间、Alt/Az、可用时的 ICRS 指向；
这是事件时刻的采样，不是独立连续监测日志，也不代表这些时刻之间均为 good time。

ROOT 各同名 TTree 保存等价字段，metadata 对象保留 header、units、provenance，
并以 schema_version 和 tables 描述各表字段。CSV 供快速检查，单位与时间参考仍以 SCHEMA、
对应观测的 FITS/ROOT header 和 manifest 为准。

## 在已有事件后追加导出

以下循环中的 event 和 prediction 来自已有重建结果；导出本身不调用模型。

    from pylast.io import (
        DL3ExternalSource, event_to_dl3_row,
        event_to_dl3_image_rows, write_dl3_events,
    )

    # 两者均为 None 时保持稳定字段，并标记 NOT_REQUESTED。
    external = DL3ExternalSource(km2a_file=None, wcda_file=None)
    rows, image_rows = [], []
    for event, prediction in stored_pairs:
        options = dict(obs_id=obs_id, geometry_name=geometry_name, energy_name=energy_name)
        row = event_to_dl3_row(event, prediction, **options)
        row.update(external.rows_for_event(event))
        rows.append(row)
        image_rows.extend(event_to_dl3_image_rows(event, prediction, **options))

    options = dict(
        obs_id=obs_id, image_rows=image_rows, preview=True, accepted_only=False,
        provenance={"external_sources": external.provenance},
    )
    write_dl3_events(rows, "events.fits", **options)
    write_dl3_events(rows, "events.root", **options)

event_to_dl3_row 的 prediction 必须包含匹配的 event_id 和明确布尔 accepted。
event_to_dl3_image_rows 的 prediction 可省略；没有对应图像参数时返回空列表。
TEL_ID 使用保存的原 telescope ID，本机单台样例为 0，不采用绘图界面的一基编号。

Native GEOMETRY_VALID、ENERGY_VALID 与最终 ACCEPTED 分开保存；
重建检查、图像 cut、quality 与 PID 状态也各自保留。未知状态或缺失值以 SCHEMA 定义为准。
三态标志保存为整数，-1 明确表示未知；FITS 不把它转换为布尔真值。
QUALITY_PROXY 是原模型的质量指标，不冒充已经校准的 DIR_ERR。
ACCEPTED 只是模型选择通过，不能据此逐事件确认为 gamma。

API 默认 accepted_only=True；设为 False 保留重建检查表。
筛选时 IMAGE_PARAMETERS 与 EVENTS 一起关联筛选，POINTING 仍保留全部输入事件。
OBS_ID 必须由观测记录或显式映射指定；本机 native run_id=0 不能区分文件。
每个文件只写一个 OBS_ID，同一观测内 EVENT_ID 必须唯一。

## KM2A / WCDA 是可选标注

DL3ExternalSource 只读取源文件已有的匹配/重建标量，不加载完整探测器 hit 图。
两类字段始终使用 KM2A_、WCDA_ 前缀：

- MATCH_STATE 区分未启用、没有可用匹配及实际匹配状态。NOT_REQUESTED 表示开关关闭，
  不能把它解释为“没有匹配”。IDENTITY_VERIFIED、DIRECTION_VALID、CORE_VALID 是不同信息。
- 选中的匹配记录校验 LACT Rabbit 身份。外部原始事件号、候选数与源条目号用于追溯，
  不取代主 OBS_ID / EVENT_ID。
- 外部 ALT / AZ 使用明确转换后的度数；CORE_EAST / CORE_NORTH 保留既有列名，实际为
  `LHAASO_ARRAY_MAP` 的两个米坐标轴，不表示已测绘确认的 geographic ENU。
  源 theta / phi 和原 core 坐标另以 RAW 字段保留，转换及 ground_frame 记录在外部 provenance 中。
- DELTA_T_RAW 保留匹配生产链原值；当前没有把其单位或数值中的约 1e7 常数偏置解释为秒。
  外部 MJD_RAW 类列即使单位为天，其时标也须查来源说明，不能直接当 UTC。
- 已直接核对事件 225185 的源 ROOT：部分 KM2A 原始事件号为负值，Event/mjd 也异常。
  这些值原样保留 RAW 并注明，不用作事件关联键或天球换算时间；原因仍需源生产链确认。
- 原始 EREC、ENERGY、LOGE 类字段保留 RAW 名称及来源说明；未证实的能量定义不能写成标准 TeV。
  标准 ENERGY 列仍只表示主 LACT 能量重建。

启用标注不会改变 LACT 的 ACCEPTED、主方向、主能量或模型筛选。
这些扩展列与 IMAGE_PARAMETERS 是本项目约定，通用工具不会自动将外部重建参与科学拟合。
如果以后用 KM2A/WCDA 做联合选择、PID 或能量/方向融合，必须定义新的分析类别，
保存对应筛选和共同有效观测时间，并制备包含联合选择效率的 IRF；不能直接沿用 LACT-only 响应。

## 时间、台址和完整观测元数据

TIME 是相对 MJDREFI / MJDREFF 的 float64 TT 秒，TIMEREF=TOPOCENTER。
从 Rabbit 整数秒和 8 ns tick 构造 Unix TAI 后转换 TT，原两个整数也保留。
TIME_MJD 是辅助 UTC MJD；不会把 UTC 数值直接改标签为 TT，也没有做质心修正。
标准 ENERGY 使用 TeV；事件与指向的 Alt/Az、RA/Dec 使用度。方位角北起向东。

传入真实 Astropy EarthLocation 后，按每条事件自己的时刻把地平坐标转换为 ICRS，压力设为 0。
KM2A_RA / KM2A_DEC 与 WCDA_RA / WCDA_DEC 继承各自上游方向约定转换后的 ALT / AZ，
也使用对应 LACT 事件的 Rabbit 时刻和同一台址；EXTTIME=LACT_EVENT 明确记录这一选择。
没有使用时标未确认的外部原始 MJD 重新计算天球坐标。
header 指向为全部输入事件的代表值；逐事件指向仍在事件列和 POINTING 中。
未传台址只允许 preview，RA/DEC 保持 NaN。不能把局部阵列米坐标当作大地坐标。

KM2A 方向复现实际上游 `phi_array = rec_phi - 0.61 + 270`，因此北起向东的
`AZ = (180 - rec_phi + 0.61) % 360`；这包含传播方向到天空来向的转换。
地面 core、ED/MD 背景和 T1 使用共同阵列图，KM2A 原始 core 映射为 `(-rec_y, rec_x)`，
不再单独旋转 0.61°。WCDA 全阵列 `xc/yc` 沿用原脚本的 30° 旋转与 `(-7, 7.97) m` 平移，
其方向保留 29.442° 参数；不同的旧 pond1 `nkg_x/nkg_y` schema 不混用。
CORESEP_KM2A_WCDA 是共同阵列图中的距离，不是独立测绘验证结果。
公式、原始来源与限制见[实测坐标说明](real_data.md#实测方向与阵列图坐标)。
本次不改变模拟 reader 或重建器；现有转换一致性也不证明绝对测绘基准、encoder 指向
和外部方向偏置已经校准。

提供真实观测记录后，可通过以下参数补齐观测组件：

    from astropy.time import Time
    gti = Time(utc_intervals, format="isot", scale="utc")
    write_dl3_events(
        rows, "events.fits", obs_id=obs_id, image_rows=image_rows,
        location=location, gti=gti, livetime=livetime_seconds,
        observation_mode="POINTING", accepted_only=False,
    )

POINTING 模式表示跟踪天球目标，DRIFT 表示固定地平指向；不能靠默认值猜模式。
GTI 不能是空区间、重叠或漏掉输出事件。LIVETIME 是扣死时间后的秒数，必须在 0 与 ONTIME 之间。
缺真实信息时，ONTIME、LIVETIME、DEADC 等不填零，不生成虚构 GTI。
DL3STAT=EVENTS_ONLY、IRFAVAIL=False 和 MISSING 明确保留这些限制。

真实科学分析还需要对应本仪器、重建、筛选和观测条件的响应。常规 full-enclosure
空间分析使用有效面积、EDISP 和 PSF；point-like 谱分析需保存真实方向 cut（常数 RAD_MAX
或随能量/offset 变化的表），不能把所有 PSF / BKG HDU 一概列为必需。背景模板取决于实际
背景估计方法；外部 off/reflected-region 方法与模板 BKG 的要求不同。

## 本地发布样例与浏览器

以下是原工作区 `real_data_plot` 的验证记录。输入数据和这些本地脚本单独保存。

已有两份已重建 ROOT 和 predictions.csv 可直接转换，不必重新跑模型。
外部默认关闭，输出默认写入 E:/LACT_pylast2/real_data_plot/dl3_release_v1：

    wsl -- bash /mnt/e/LACT_pylast2/real_data_plot/run_dl3_export.sh --preview

显式启用两种标注：

    wsl -- bash /mnt/e/LACT_pylast2/real_data_plot/run_dl3_export.sh --preview --with-km2a --with-wcda

--sidecar-dir 默认是 real_data_plot/reference_data/20260214，按下面路径查找同 stamp 文件；
启用的文件缺失会报错，不静默退回关闭：

    km2a/analysis_build_{stamp}.rec.root
    wcda/analysis_build_{stamp}.merge.root

已有输出时显式加 --overwrite。--accepted-only 只保留原模型接受的事件与关联图像参数；
建议另设 --output-dir 保存。默认全表导出历史试跑的 140 个非 LED 记录，原模型接受 9 个；
这是输入版本的历史计数，实际发布以 manifest 为准。

每观测保存一份 FITS 和一份 ROOT，另有 events.csv、image_parameters.csv、manifest.json。
转换器逐列核验全部 FITS/ROOT 表，包括 SCHEMA 和 PROVENANCE：字符串按值，数值允许相同 NaN；
同时核对图像参数的事件关联、accepted-only 联动，以及 POINTING 不随接受筛选缩短。

manifest 保存实际行数、外部匹配状态计数、缺项、外部源 provenance、输入文件 SHA256、
模型试跑 summary / release hash、导出代码，以及可访问时的 git commit 和工作区状态；
执行环境无法读取 Git 信息时明确记为未知，不从参考版本推断当前 commit。
当前参照配置与标定/映射/指向文件也记录来源；当前配置不冒充已留档的历史配置。
具体模型、清理条件和训练域限制以原试跑 summary 为准。

配置文件 real_data_plot/dl3_observation.json 当前使用获准的公共 LHAASO 近似台址：
东经 100.1387944444°、北纬 29.3576555556°、高度 4410 m。
SITEAPP=True 和来源记录保留近似性质，未声称这是 LACT1 精确椭球高。
确认真实台址、GTI、活时间和模式后修改配置；本导出器仍不会凭空产生 IRF。

生成并执行字段浏览 notebook / HTML：

    wsl -- bash /mnt/e/LACT_pylast2/real_data_plot/validation/run_validation.sh /mnt/e/LACT_pylast2/real_data_plot/notebook_support/build_dl3_viewer.py

浏览器默认读取 dl3_release_v1，可用 --data-dir 指定其他导出目录。
它从 SCHEMA 展示全部列及单位/说明，按 OBS_ID / EVENT_ID 查看 LACT、KM2A、WCDA、
逐台图像参数和指向，兼容两个外部开关关闭。FITS 单表也可用 Astropy Table.read，
或 Gammapy EventList.read(path, hdu="EVENTS") 读取；可读不等于已具备曝光或能谱分析条件。
