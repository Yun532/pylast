# Atmosphere input and compatibility

`LactEventSource` automatically reads a one-entry `cfg/atmosphere_model` tree
from new LACTsim ROOT files. `RootEventSource` uses the same reader for pylast
output. Its four `ROOT::VecOps::RVec<double>` branches are `alt_km` (km above sea
level), `rho` (g/cm3), `thick` (vertical g/cm2), and `refidx_m1` (dimensionless).
The profile is validated before constructing the height-to-depth spline.
A present but malformed tree is an error.

Old LACTsim files without this tree remain supported. For them only, the Python
API allows an explicitly matched four-column external table:

```python
from pylast.io import LactEventSource
source = LactEventSource("old_lact.root")
source.load_atmosphere_model("matched_atmprof.dat")
```

**An input atmosphere always forbids an external profile, even if identical.**
This rule applies to `LactEventSource`, `RootEventSource`, and the native
atmosphere of `SimtelEventSource`. It is enforced by the common C++ API, not just
the command-line wrapper.

For LACTsim reconstruction, use the Python entry point with your reconstruction
configuration:

```bash
# New file: automatic embedded profile; do not specify an external profile.
python -m pylast.ulities.hillas_reco --input-format lact \
  -i new_lact.root -o reconstructed.root -c reconstruction.json

# Old file: optional external profile, matched to its original CORSIKA run.
python -m pylast.ulities.hillas_reco --input-format lact \
  -i old_lact.root -o reconstructed.root -c reconstruction.json \
  --atmosphere-profile matched_atmprof.dat
```

This LACT CLI automatically writes the four-column atmosphere into its output
when available. Direct `DataWriter` callers should set
`write_atmosphere_model=true`. The LACTsim provenance and optical-transmission
trees stay in the original simulation ROOT; this minimal change does not copy
those additional trees into pylast's reconstruction output.

Simtelarray keeps its existing reader and calibration path. Use the existing
executable interface, or `--input-format simtel` for the Python loop.

Without an atmosphere, height-to-depth conversion returns NaN. Valid direction,
core position, and energy estimates remain usable; opening a new input resets
the previous input's global atmosphere. The current depth conversion is a
**vertical column depth**, not a validated slant-depth Xmax estimator. Embedding
a profile does not change the reconstruction method or model training.
