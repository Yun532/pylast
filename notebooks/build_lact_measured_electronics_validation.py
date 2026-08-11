#!/usr/bin/env python3
"""Build the reproducible measured-electronics validation notebook."""

from pathlib import Path

import nbformat as nbf


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "pyLAST measured electronics",
    "language": "python",
    "name": "pylast-e2e",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}

nb["cells"] = [
    code(r'''
# Cell 1：所有外部路径和可调参数都集中在这里。
from pathlib import Path

PYLAST_REPO = Path.cwd().resolve()
if PYLAST_REPO.name == "notebooks":
    PYLAST_REPO = PYLAST_REPO.parent
WORKSPACE = PYLAST_REPO.parent
LACT_REPO = WORKSPACE / "LACT_sim_measured_electronics_github_20260811"
DATA_DIR = LACT_REPO / "validation" / "measured_electronics"
OUTPUT_DIR = PYLAST_REPO / "validation" / "notebook_outputs"

EVENT_ID = 1909
TELESCOPE_ID = 19
SAMPLE_WIDTH_NS = 4.0
LOCAL_WINDOW_WIDTH = 7
LOCAL_WINDOW_SHIFT = 3
LOCAL_APPLY_CORRECTION = True
WAVEFORM_PIXEL_COUNT = 4
FIGURE_DPI = 150

MODE_DIRS = {
    "饱和关、波形关": DATA_DIR / "saturation_off_waveform_off",
    "饱和开、波形关": DATA_DIR / "saturation_on_waveform_off",
    "实测波形、无NSB": DATA_DIR / "measured_waveform_no_nsb",
    "仅NSB": DATA_DIR / "nsb_only",
    "切伦科夫+NSB": DATA_DIR / "cherenkov_plus_nsb",
}
ROOT_NO_NSB = MODE_DIRS["实测波形、无NSB"] / "lact_events.root"
H5_NO_NSB = MODE_DIRS["实测波形、无NSB"] / "lact_events.h5"
FORMAT_REPORT = DATA_DIR / "validation_report.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for required in [ROOT_NO_NSB, H5_NO_NSB, FORMAT_REPORT]:
    if not required.exists():
        raise FileNotFoundError(required)
print(f"数据目录：{DATA_DIR}")
print(f"图片目录：{OUTPUT_DIR}")
'''),
    markdown(r'''
## Cell 2：建立统一的读取、标定和相机绘图工具

本 cell 只定义公共函数。相机轮廓、坐标方向、零值白色和 plasma 配色沿用
pyLAST `EventVisualizer`；所有图都显示完整 1664 像素，不使用透明色。
'''),
    code(r'''
import json
import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from pylast.helper import Calibrator, LactEventSource
from pylast.visualize import EventVisualizer

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": FIGURE_DPI})

def h5_image(path, field):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[f"images/dense/{field}"][0], dtype=float)

def h5_waveform(path):
    with h5py.File(path, "r") as handle:
        pixel_axis = np.asarray(handle["waveforms/pixel_id_axis"][:], dtype=int)
        centers = np.asarray(handle["waveforms/time_centers_ns"][:], dtype=float)
        samples = handle["waveforms/samples"][:]
        waveform = np.zeros((len(pixel_axis), len(centers)), dtype=float)
        pixel_to_column = {int(pixel_id): i for i, pixel_id in enumerate(pixel_axis)}
        for row in samples[samples["image_index"] == 0]:
            waveform[pixel_to_column[int(row["pixel_id"])], int(row["time_bin"])] = row["sample_value"]
        attrs = {key: handle["waveforms"].attrs[key] for key in handle["waveforms"].attrs}
    return pixel_axis, centers, waveform, attrs

def common_camera_norm(images):
    vmax = max(float(np.max(image)) for image in images)
    vmax = max(vmax, 1.0)
    colors = [(1, 1, 1, 1)] + plt.cm.plasma(np.linspace(0, 1, 256)).tolist()
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1] + np.linspace(1, vmax, 256).tolist()
    return mcolors.BoundaryNorm(bounds, cmap.N), cmap, vmax

def plot_camera_layers(visualizer, tel_id, layers, output_name):
    labels = list(layers)
    images = [np.asarray(layers[label], dtype=float) for label in labels]
    norm, cmap, vmax = common_camera_norm(images)
    fig, axes = plt.subplots(1, len(images), figsize=(5.2 * len(images) + 1.0, 5.0))
    axes = np.atleast_1d(axes)
    geometry = visualizer.tel_geoms[tel_id]
    collection = None
    for ax, label, image in zip(axes, labels, images):
        visualizer._draw_camera_image(ax, geometry, image, norm, cmap, add_colorbar=False)
        ax.set_title(f"{label}\nΣ={image.sum():.3f} PE")
        collection = ax.collections[-1]
    fig.subplots_adjust(wspace=0.22, right=0.90)
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    fig.colorbar(collection, cax=cax,
                 label=f"PE (shared scale, zero = white, max = {vmax:.3g})")
    path = OUTPUT_DIR / output_name
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    return path

def run_calibrator(source, extractor_config):
    event = source[0]
    calibrator = Calibrator(source.subarray, json.dumps(extractor_config))
    calibrator(event)
    return event, np.asarray(event.dl0.tels[TELESCOPE_ID].image, dtype=float)
'''),
    markdown(r'''
## Cell 3：1909/19 的三层相机图

三层依次为：PDE 后且饱和前的纯切伦科夫 PE、微单元饱和后的整数 fired PE、
实测单 PE 电荷涨落后由 pyLAST 全波形积分重建的 PE。三幅图共用同一色标，
所以白色像素明确表示 0。
'''),
    code(r'''
source = LactEventSource(str(ROOT_NO_NSB))
raw_event = source[0]
assert raw_event.event_id == EVENT_ID
assert TELESCOPE_ID in source.subarray.tels

full_event, full_image = run_calibrator(
    source, {"image_extractor_type": "FullWaveFormExtractor"}
)
primary_image = h5_image(H5_NO_NSB, "primary_cherenkov_pe")
fired_image = h5_image(H5_NO_NSB, "fired_cherenkov_pe")

visualizer = EventVisualizer(source, enable_secondary_axes=False)
THREE_LAYER_PATH = plot_camera_layers(
    visualizer,
    TELESCOPE_ID,
    {
        "Primary Cherenkov": primary_image,
        "Fired after saturation": fired_image,
        "pyLAST FullWaveForm": full_image,
    },
    "event1909_tel19_three_camera_layers.png",
)
print(THREE_LAYER_PATH)
'''),
    markdown(r'''
## Cell 4：FullWaveFormExtractor 与 LocalPeakExtractor 闭合

全波形结果应与 `Σ(mV sample) × 4 ns / 84.0349557 (mV·ns/PE)` 完全一致。
LocalPeak 使用 7 个采样点、向峰前移动 3 点，并用 ROOT 内保存的实测参考脉冲做尾部修正。
'''),
    code(r'''
readout = source.subarray.tels[TELESCOPE_ID].camera.readout
raw_waveform = np.asarray(raw_event.r1.tels[TELESCOPE_ID].waveform, dtype=float)
direct_pe = raw_waveform.sum() / readout.sampling_rate / readout.single_pe_area_mv_ns

local_event, local_image = run_calibrator(
    source,
    {
        "image_extractor_type": "LocalPeakExtractor",
        "LocalPeakExtractor": {
            "window_width": LOCAL_WINDOW_WIDTH,
            "window_shift": LOCAL_WINDOW_SHIFT,
            "apply_correction": LOCAL_APPLY_CORRECTION,
        },
    },
)

closure = pd.DataFrame([
    {"方法": "Direct mV integral", "总PE": direct_pe, "相对Full": 0.0},
    {"方法": "FullWaveFormExtractor", "总PE": full_image.sum(),
     "相对Full": (full_image.sum() - direct_pe) / direct_pe},
    {"方法": "LocalPeakExtractor corrected", "总PE": local_image.sum(),
     "相对Full": (local_image.sum() - full_image.sum()) / full_image.sum()},
])
display(closure)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(closure["方法"], closure["总PE"], color=["#4c78a8", "#f58518", "#54a24b"])
ax.set_ylabel("Integrated charge [PE]")
ax.tick_params(axis="x", rotation=12)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "extractor_charge_closure.png", bbox_inches="tight")
plt.show()
'''),
    markdown(r'''
## Cell 5：五种配置模式的积分结果

这里直接读取 LACT_sim 的格式一致性 JSON。饱和开/关和波形开/关不会混淆；
NSB-only 与切伦科夫+NSB 也分列显示。
'''),
    code(r'''
report = json.loads(FORMAT_REPORT.read_text(encoding="utf-8"))
rows = []
for label, directory in MODE_DIRS.items():
    mode = directory.name
    item = report[mode]
    rows.append({
        "模式": label,
        "切伦科夫 primary": item["primary_cherenkov_pe"],
        "NSB primary": item["primary_nsb_pe"],
        "切伦科夫 fired": item["fired_cherenkov_pe"],
        "NSB fired": item["fired_nsb_pe"],
        "饱和损失": item["saturation_lost_pe"],
        "有波形": item["waveform_enabled"],
        "触发": item["triggered"],
    })
mode_table = pd.DataFrame(rows)
display(mode_table)
'''),
    markdown(r'''
## Cell 6：无 NSB 的实测 4 ns 像素波形

选择峰值最大的若干像素，只画纯切伦科夫事例的 mV 波形。横轴使用相对于相机首个
切伦科夫 PE 的时间，像素之间允许有不同的到达时刻。
'''),
    code(r'''
pixel_axis, centers, waveform, attrs = h5_waveform(H5_NO_NSB)
selected = np.argsort(waveform.max(axis=1))[-WAVEFORM_PIXEL_COUNT:][::-1]
fig, ax = plt.subplots(figsize=(9, 4.5))
for column in selected:
    ax.plot(centers, waveform[column], marker="o", ms=2.5,
            label=f"pixel {pixel_axis[column]}")
ax.set_xlabel("Time relative to first camera Cherenkov PE [ns]")
ax.set_ylabel("Average voltage [mV / 4 ns sample]")
ax.grid(alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cherenkov_only_pixel_waveforms.png", bbox_inches="tight")
plt.show()
'''),
    markdown(r'''
## Cell 7：NSB-only 波形（单独显示）

这个 cell 不叠加切伦科夫信号。上图为所有像素的平均 NSB 波形，下图为峰值最大的
若干像素，用来检查有限读出窗和随机 NSB 脉冲。
'''),
    code(r'''
nsb_h5 = MODE_DIRS["仅NSB"] / "lact_events.h5"
nsb_axis, nsb_time, nsb_waveform, _ = h5_waveform(nsb_h5)
nsb_selected = np.argsort(nsb_waveform.max(axis=1))[-WAVEFORM_PIXEL_COUNT:][::-1]
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(nsb_time, nsb_waveform.mean(axis=0), color="#4c78a8")
axes[0].set_ylabel("Camera mean [mV]")
for column in nsb_selected:
    axes[1].plot(nsb_time, nsb_waveform[column], label=f"pixel {nsb_axis[column]}")
axes[1].set_xlabel("Time relative to readout reference [ns]")
axes[1].set_ylabel("Average voltage [mV]")
axes[1].legend(ncol=2)
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "nsb_only_waveforms.png", bbox_inches="tight")
plt.show()
'''),
    markdown(r'''
## Cell 8：切伦科夫 + NSB 波形（单独显示）

本 cell 使用 LACT_sim 内部已经合成并触发后的总波形，与上一 cell 的 NSB-only 保持
分开，避免视觉上误以为二者是同一个输入。
'''),
    code(r'''
combined_h5 = MODE_DIRS["切伦科夫+NSB"] / "lact_events.h5"
combined_axis, combined_time, combined_waveform, _ = h5_waveform(combined_h5)
combined_selected = np.argsort(combined_waveform.max(axis=1))[-WAVEFORM_PIXEL_COUNT:][::-1]
fig, ax = plt.subplots(figsize=(9, 4.5))
for column in combined_selected:
    ax.plot(combined_time, combined_waveform[column], marker="o", ms=2.0,
            label=f"pixel {combined_axis[column]}")
ax.set_xlabel("Time relative to first camera Cherenkov PE [ns]")
ax.set_ylabel("Average voltage [mV / 4 ns sample]")
ax.grid(alpha=0.25)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cherenkov_plus_nsb_waveforms.png", bbox_inches="tight")
plt.show()
'''),
    markdown(r'''
## Cell 9：ROOT/HDF5/CSV 一致性结论

报告由独立脚本逐像素、逐 fired-hit、逐波形采样比较生成。本 cell 只把最关键的误差
和单 PE 定标常数集中展示，便于以后更换电子学模型时复用同一验收标准。
'''),
    code(r'''
format_rows = []
for mode, item in report.items():
    format_rows.append({
        "模式": mode,
        "图像最大绝对误差": item["root_hdf5_csv_max_abs_error"],
        "波形 ROOT/HDF5 最大误差 [mV]": item.get("waveform_root_hdf5_max_abs_error_mv", np.nan),
        "波形 ROOT/CSV 最大误差 [mV]": item.get("waveform_root_csv_max_abs_error_mv", np.nan),
        "单PE面积 [mV ns]": item.get("single_pe_area_mv_ns", np.nan),
        "直接全积分相对误差": item.get("full_waveform_direct_relative_error", np.nan),
    })
format_table = pd.DataFrame(format_rows)
display(format_table)
print("三层相机图：", THREE_LAYER_PATH)
print("所有图：", OUTPUT_DIR)
'''),
]

output = Path(__file__).with_name("lact_measured_electronics_validation.ipynb")
nbf.write(nb, output)
print(output)
