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



