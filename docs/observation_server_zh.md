# 在 Linux 服务器运行观测重建 Notebook

建议使用独立目录 `~/pylast-lact_sim`，不改动服务器原有工作树：

```bash
repo="$HOME/pylast-lact_sim"
if [ -d "$repo/.git" ]; then
  git -C "$repo" switch lact_sim && git -C "$repo" pull --ff-only origin lact_sim
else
  git clone --recurse-submodules -b lact_sim https://github.com/Yun532/pylast.git "$repo"
fi
git -C "$repo" submodule update --init --recursive
```

模型和数据放在私有目录，例如 `~/pylast-assets`。从私有交付复制完整 `model_bundle_v3/`；
立体链另需匹配天顶角和布局的 E18、E20、PID23 模型及配套 `LACT_36tel_E20_PID23_50h_code.zip`。
模板中的名称分别为 `energy18.txt`、`energy20.txt`、`pid23.txt`。换模型时也可在 JSON 直接填真实文件名。
立体样本和外部匹配文件可选；单台实测的立体结果应为空，可用匹配布局的模拟 DL1 ROOT 做立体验证。

```bash
assets="$HOME/pylast-assets"
cd "$HOME/pylast-lact_sim"
mkdir -p "$assets"
test -e "$assets/observation.json" || cp config/observation_reconstruction.example.json "$assets/observation.json"
```

编辑 `observation.json`：填实测 ROOT、映射 CSV、encoder 指向日志、LED 校准 ROOT，选择非 LED 的
`event_index`（从 0 开始），必要时配置 `analysis_file` 或 `stereo_demo`。
路径可为绝对路径，也可相对这个 JSON 所在目录。模型应由你信任的私有来源提供。

在独立的 Python 3.12 环境和 C++20 编译器下安装。下面使用验证环境对应的 Linux ROOT wheel，
以及同一份 Python/原生 LightGBM 4.7.0 库；不要把这些依赖装到旧分析环境里。
如果登录节点默认 Python 版本较旧，可以先用 `conda create -n pylast-observation python=3.12 pip` 创建环境，
再运行 `conda activate pylast-observation`。

```bash
cd "$HOME/pylast-lact_sim"
python -m pip install -r "$HOME/pylast-assets/model_bundle_v3/requirements.txt" \
  'root==0.1a12' 'lightgbm==4.7.0' scikit-build-core nanobind typing_extensions ninja jupyterlab ipykernel
ROOT_PKG="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("ROOT").submodule_search_locations)))')"
export LIGHTGBM_LIBRARY="$(python -c 'from pathlib import Path; import lightgbm; print(Path(lightgbm.__file__).resolve().parent / "lib/lib_lightgbm.so")')"
export LD_LIBRARY_PATH="$ROOT_PKG/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install --no-build-isolation '.[real-data]' \
  -Ccmake.define.WITH_EXT_REC=ON -Ccmake.define.ROOT_DIR="$ROOT_PKG/cmake"
python -m ipykernel install --sys-prefix --name pylast-observation --display-name 'pyLAST observation'
```

在仓库根目录执行一次 Notebook 并把结果留在私有目录：

```bash
cd "$HOME/pylast-lact_sim"
export PYLAST_OBSERVATION_CONFIG="$HOME/pylast-assets/observation.json"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=pylast-observation \
  --output-dir "$HOME/pylast-assets" --output observation.executed.ipynb \
  notebooks/lact_observation_reconstruction.ipynb
```

如需交互运行，可改用 `jupyter lab --no-browser --ip=127.0.0.1`，打开同一 Notebook 并选择 `pyLAST observation` 内核。
Notebook 的默认代码启用两条链；没有多相机观测或模拟替代样本时，会说明多台分支未产生结果。
