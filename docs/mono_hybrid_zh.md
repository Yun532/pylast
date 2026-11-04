# Mono / hybrid 的安装与使用

## 可替换的 v3 单台与 E20/PID23 立体模型

服务器版[实测 Notebook](../notebooks/lact_observation_reconstruction.ipynb)通过观测配置 JSON 选择模型，
见 [Linux 服务器指南](observation_server_zh.md)。
更换同接口的版本时，替换整个 v3 bundle（含 recipe、模型和阈值），或修改配置里的
`mono.bundle_dir`；立体的 E18、E20、PID23 三份 LightGBM 文件分别由 `stereo.models`
指定。路径相对于该 JSON。模型、配套阈值和特征约定须来自同一次发布，不混搭。
不需要将私有权重复制到源码仓库。

```python
from pylast.reco.PortableMonoReconstructor import PortableMonoReconstructor
from pylast.reco.PortableStereoReconstructor import PortableStereoReconstructor

# mono_config / stereo_config 是已解析路径的配置字典；event 已做 DL1 参数化。
mono = PortableMonoReconstructor(source.subarray, mono_config)
one = mono(event, telescope_id=actual_tel_id)  # 实际事件 ID；单张可用图像可省略
stereo = PortableStereoReconstructor(source.subarray, stereo_config)
multi = stereo(event)
```

两者也可以用其类名在 `ShowerProcessor.GeometryReconstructionTypes` 显式注册，
同名配置段填模型配置；现有默认及旧 mono/hybrid 路由不变。
v3 单台使用同版图像筛选和质量/PID 节点，在单台实测数据上不计算需要模拟真值的方向误差。
`fixed` 是阈值来源，不能视为任意物理望远镜上的性能已校准；也不会把单台记录认成阵列 `exact_one`。
新版立体使用发布的几何/特征实现：E20 输出能量，E18 供 PID23 使用。模型输出不自动套用
观测曝光相关的 sensitivity h/θ 筛选。实测数据若只有一张图像，就没有该事件的立体解。
在同一进程内调用原生扩展和 Python 立体模型时，两者应使用同一份 LightGBM 动态库；
Notebook 的内核已配置为使用与本机原生扩展相同的 4.7 库。

以下保留旧模型与原有 hybrid 的用法。

此扩展增加 mono 和 hybrid 重建、必要的 Python 绑定，并将原有重建器注册表
统一到共享库中的一个实例，避免 Python 扩展无法找到原立体能量/PID重建器。
原有 C++ 立体重建、
图像清洗、标定和立体模型不变；原配置不启用新类型时仍走原立体流程。

## 安装

在目标 Linux 服务器配置好原 pyLAST 所需的 ROOT、C++20 编译器和 LightGBM。
从代码仓库编译，不要直接复制另一台服务器的 `site-packages` 或假定 wheel 通用：

```bash
git clone --recurse-submodules https://github.com/Yun532/pylast.git
cd pylast
git checkout lact_sim
git submodule update --init --recursive
# 保持你原来的 ROOT / LightGBM 环境设置；LIGHTGBM_LIBRARY 指向已有 .so。
python -m pip install ".[mono]" --config-settings=cmake.args="-DWITH_EXT_REC=ON"
```

如果使用交付的源码压缩包，解压后进入其根目录直接执行上面的 `pip install`，
无需执行 Git 命令；源码包已包含必需的 hessio 子模块。
原 mono 扩展分支保留为 `mono-hybrid` 备份，后续开发使用合并后的 `lact_sim`。

Mono 使用 Python 3.11 或更新版本；`mono` extra 固定了现有模型的
NumPy / pandas / scikit-learn / joblib 版本。
普通立体用户仍可按原方式安装，无需启用 mono extra。
为避免 mono extra 的版本要求升级你的旧环境，先在单独环境中安装/测试；
不要直接覆盖生产环境。不开启新类型时不会导入或加载 mono 模型。

旧 Python 模块若依赖安装目录 `pylast/model/*.pkl`，这些模型也单独私有交付。
用兼容包内的 `restore_legacy_models.py --pylast-dir /path/to/site-packages/pylast`
恢复原路径即可；脚本拒绝覆盖任何不同 SHA 的已有模型。你已有的外部立体
模型路径和配置不需要改。此步骤只恢复资产，不认证原仓库中已缺失绑定的
历史 Python ML 类。

## 模型单独复制

私有 `model_bundle` 放在代码目录**之外**，例如 `/data/lact_models/mono_hybrid`：

```text
mono_hybrid/
  mono_z20.json
  mono_z60.json
  z20/{reconstruction,proxy,pid}.joblib
  z20/{quality_cut,pid_cut}.csv
  z60/{reconstruction,proxy,pid}.joblib
  z60/{quality_cut,pid_cut}.csv
  SHA256SUMS
  manifest.json
```

JSON 中五个资产路径相对于 bundle 根目录，`expected_sha256` 在加载前检查文件。
选 20° 或 60° 对应配置；当前冻结模型不支持天顶角插值或任意新布局。
joblib 是可执行的 Python 序列化格式，只加载你信任的私有模型包。
此包没有独立的 hybrid 模型，也不包含你已有的立体模型；立体模型沿用原配置。
不要把模型、cut 表或模型 ZIP 放进 Git 仓库。

## 接在现有流程之后（推荐）

保留原 EventSource、Calibrator、ImageProcessor 和立体配置。只在同一个事件的
图像处理完成后追加以下对象；`subarray`、`stereo_config` 和 `event` 使用你原来的对象：

```python
import json
from pathlib import Path
from pylast.reco import ShowerProcessor
from pylast.reco.MonoReconstructor import MonoReconstructor
from pylast.reco.HybridReconstructor import HybridReconstructor

model_dir = Path("/data/lact_models/mono_hybrid")
mono_config = json.loads((model_dir / "mono_z20.json").read_text())
stereo = ShowerProcessor(subarray, json.dumps(stereo_config))
mono = MonoReconstructor(subarray, mono_config, model_dir=model_dir)
hybrid = HybridReconstructor(subarray, stereo, mono)

# 在原事件循环里：calibrator(event); image_processor(event) 已执行一次。
stereo(event)
mono_table = mono(event)                 # 固定 tel7 对照
route = hybrid(event, run_stereo=False)  # 不重复执行立体或图像处理
# 然后继续你原来的 writer(event)。
```

`config_str` 接受 JSON 内容字符串或字典，不接受 JSON 文件名。
默认 bundle 使用已有的 simulated-image 参数（`use_fake_hillas=True`），
这不是一套已认证的实测波形模型。要用实测 DL1，需另外验证对应输入/模型域，
不能只改这个开关就声称性能已验证。
合并后保留 `lact_sim` 的 ROOT 真值图读取规则：优先读取
`image_primary_cherenkov_pe`，缺失时读取 `image_cherenkov_pe`，不会把含噪声的
detector-level `image_pe` 当作真值图。原 mono 分支的
`allow_legacy_cherenkov_alias` 参数仍可传入以兼容旧调用；本分支默认已经支持
两个字段，该参数不改变现有优先级。
真正没有图像的事件正常拒绝，不伪填事件或重建结果。

本次合并保留 10 月 7 日原模型接口。10 月 19 日的 Original S 候选包使用不同
模型字典、原 S 图像筛选、tel0 及分组阈值，不能只替换这里的模型路径；应另做
格式和筛选适配及数值验证。实测 CLI 允许显式配置 mono/hybrid，并将 mono 的
`use_fake_hillas` 设为 `False`，默认 `config/real_data.json` 仍使用原 Hillas 配置。

新增 ROOT / DL2 结果名为：

| 结果 | 方向 | 能量 |
|---|---|---|
| 固定 tel7 mono | `MonoDISPReconstructor` | `MonoEnergyRegressor` |
| hybrid | `HybridDirection` | `HybridEnergy` |

原来的 `HillasReconstructor`、`EnergyRegressor`、`ParticleClassifier` 保留。
原生能量单位为 TeV；mono 不估计 core / hmax，这些字段为 NaN，不伪填零。

Mono 的质量/PID状态在 `mono_table` 中，最终选择看 `accepted`，不能只看
DL2 `is_valid`。Hybrid 的 `branch` 表示路由：多台 guard 图像且立体方向有效时
复制原立体结果，恰好一台时走 mono；失败多镜不回退到 mono。
`route["accepted"]` 在立体分支只表示路由成功，**不替代你原来的立体质量/PID cut**。
Writer 不会自动把这些 sidecar 选择变成分析事件筛选。

## 也可在 ShowerProcessor 中显式注册

将你原来的 ShowerProcessor 配置字典增加两项（已有立体参数原样保留）：

```python
from copy import deepcopy

config = deepcopy(stereo_config)
section = config.get("ShowerProcessor", config)
section["GeometryReconstructionTypes"] = section.get(
    "GeometryReconstructionTypes", ["HillasReconstructor"]
) + ["MonoReconstructor", "HybridReconstructor"]
section["MonoReconstructor"] = {**mono_config, "model_dir": str(model_dir)}
section["HybridReconstructor"] = {}  # 仅可选结果名称，不放模型
processor = ShowerProcessor(subarray, config)
processor(event)
mono_table = processor.last_mono_result
route = processor.last_hybrid_result
```

Hybrid 注册要求链中已有 `HillasReconstructor`；立体能量/PID仍按原配置启用。
不要同时使用注册方式和上面的手动追加方式，否则会重复计算。

## 检查

```bash
cd /data/lact_models/mono_hybrid
sha256sum -c SHA256SUMS
python /path/to/pylast/tests/test_mono_contract.py
```

合同测试使用 native/坐标 doubles 和临时小模型，不等同于真实 ROOT 验收。
迁移后先用少量已有事件验证加载和输出，再开始自己的全量重建。
