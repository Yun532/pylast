This is the standard analysis science tool for LACT Project. 
# INSTALLATION
```bash
pip install .
```

## Support Database
pip install . --config-settings=cmake.args="-DWITH_EXT_REC=ON"

## Required Package
LightGBM Library: Used to load and use the `lightgbm` model
Set the LIGHTGBM_LIBRARY to path of liblightgbm.so

## Coordinates

The code-level coordinate audit, including the LACT ROOT camera mapping, is in
[`docs/coordinate_code_audit_zh.md`](docs/coordinate_code_audit_zh.md).

## LACT_sim data levels

The mapping from LACT_sim ROOT/HDF5 fields to pyLAST data levels and plotting
interfaces is documented in
[`docs/lact_sim_data_levels_zh.md`](docs/lact_sim_data_levels_zh.md).

## LACT event reconstruction notebooks

- [`notebooks/lact_event_reconstruction_no_nsb.ipynb`](notebooks/lact_event_reconstruction_no_nsb.ipynb): complete LACT_sim-style event visualization and reconstruction without NSB.
- [`notebooks/lact_event_reconstruction_with_nsb.ipynb`](notebooks/lact_event_reconstruction_with_nsb.ipynb): the same complete flow after pyLAST Poisson NSB addition and mean-pedestal subtraction.

The same two notebooks are stored in both LACT_sim and pyLAST. Set `INPUT_FILE`
and `EVENT_ID` in Cell 1, then run all cells.

## Optional mono and hybrid reconstruction

The original C++ stereo chain is unchanged. Optional Python stages add frozen
image-only mono direction/energy and stereo-first exact-one-image hybrid routing.
See [the deployment and usage guide](docs/mono_hybrid_zh.md).

Install with `pip install ".[mono]"` in a compatible ROOT/C++ environment.
Trained models and calibrated cut tables are **external private assets**, supplied
separately; they are not included in this repository or its distribution packages.

The main development branch is `lact_sim`; `mono-hybrid` preserves the original
standalone mono/hybrid version.



