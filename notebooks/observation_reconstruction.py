"""Notebook reconstruction from the selected event and an independent stereo example."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _models(config_path):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    def resolve(value):
        return str((path.parent / value).resolve())
    config["mono"]["bundle_dir"] = resolve(config["mono"]["bundle_dir"])
    stereo = config["stereo"]
    stereo["pipeline_archive"] = resolve(stereo["pipeline_archive"])
    stereo["models"] = {name: resolve(value) for name, value in stereo["models"].items()}
    if config.get("stereo_demo"):
        config["stereo_demo"] = resolve(config["stereo_demo"])
    return config


def run_mono_demo(event, source, config_path):
    from pylast.reco.MonoReconstructor import disp_to_sky
    from pylast.reco.PortableMonoReconstructor import PortableMonoReconstructor

    ids = sorted(getattr(event.dl1, "tels", {}))
    if event.is_led is not False or not ids:
        print("实测单台：本条是 LED / 状态未知，或无清洗图像，跳过。")
        return None
    mono = PortableMonoReconstructor(source.subarray, _models(config_path)["mono"])
    zenith = 90.0 - np.degrees(event.pointing.array_altitude)
    if not np.isclose(zenith, mono.zenith_deg, atol=.1, rtol=0):
        print(f"实测单台：天顶角 {zenith:.3f}° 不在所选 {mono.zenith_deg}° ± 0.1° 模型域。")
        return None
    output = []
    for tel_id in ids:
        row = mono(event, telescope_id=tel_id, store=len(ids) == 1).iloc[0]
        alt = az = np.nan
        if row.reconstruction_valid:
            alt, az = disp_to_sky(row.nominal_x, row.nominal_y, row.nominal_psi,
                row.disp_reco_deg, event.pointing.array_azimuth, event.pointing.array_altitude)
        output.append(dict(event_id=int(event.event_id), telescope_id=int(tel_id),
            altitude_deg=np.degrees(alt), azimuth_deg=np.degrees(az) % 360,
            energy_TeV=row.energy_reco_GeV / 1000, image_cut_pass=row.image_cut_pass,
            quality_pass=row.quality_pass, gamma_probability=row.pid_gamma_score,
            pid_pass=row.pass_pid, accepted=row.accepted))
    print(f"实测单台 | {row.model_candidate_id} | {mono.group} | {mono.branch} 参考阈值")
    return pd.DataFrame(output)


def run_stereo_demo(event, source, config_path):
    from pylast.io import RootEventSource
    from pylast.reco.PortableStereoReconstructor import PortableStereoReconstructor

    config = _models(config_path)
    n_images = len(getattr(event.dl1, "tels", {}))
    if event.is_led is False and n_images >= 2:
        stereo = PortableStereoReconstructor(source.subarray, config["stereo"])
        result = stereo(event)
        dataset = "实测"
    else:
        print(f"实测多台 | 事件 {event.event_id} 只有 {n_images} 张 DL1 图像，不能立体重建。")
        if not config.get("stereo_demo"):
            return None
        simulation = RootEventSource(config["stereo_demo"])
        stereo = PortableStereoReconstructor(simulation.subarray, config["stereo"])
        result = None
        for candidate in simulation:
            if len(getattr(candidate.dl1, "tels", {})) >= 2:
                result = stereo(candidate)
                if result["reconstruction_valid"]:
                    break
        if result is None or not result["reconstruction_valid"]:
            raise RuntimeError("多台模拟样本没有通过当前立体图像筛选的事件")
        dataset = "模拟 DL1"
    print(f"{dataset}多台 | {config['stereo']['group']} | E20 输出能量、E18 供 PID23")
    columns = ("event_id", "n_selected_images", "selected_telescope_ids", "reco_altitude_deg",
        "reco_azimuth_deg", "reco_core_x_m", "reco_core_y_m", "rec_energy_TeV",
        "PID23_rec_energy_E18_TeV", "pid23_hadronness", "reconstruction_valid", "reason")
    return pd.DataFrame([{name: result[name] for name in columns}])
