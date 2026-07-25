from ..helper import LactEventSource as _NativeLactEventSource


def _event_id(event_or_id):
    if isinstance(event_or_id, int):
        return event_or_id
    event_id = getattr(event_or_id, "event_id", None)
    return None if event_id is None else int(event_id)


def _unique(values):
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _event_shower_ids(event_or_id):
    candidates = []
    shower = getattr(getattr(event_or_id, "simulation", None), "shower", None)
    if shower is not None:
        for name in ("shower_event_id", "corsika_event_id", "event_id"):
            value = getattr(shower, name, None)
            if value is not None:
                try:
                    candidates.append(int(value))
                except Exception:
                    pass
    return _unique(candidates)


def _native_triggered_tels(event):
    simulation = getattr(event, "simulation", None)
    triggered = getattr(simulation, "triggered_tels", None)
    if triggered is not None:
        triggered_tels = tuple(sorted(int(tel_id) for tel_id in triggered))
        if triggered_tels:
            return triggered_tels
    return ()


class LactEventSource:
    """Python adapter around the native LACT ROOT event source.

    Exposes a uniform ``get_triggered_tels`` helper. LACT ROOT readout images
    are detector-level data, so the native adapter does not need to put them in
    ``event.simulation`` just to carry trigger state. The observations tree is
    used as the source-level trigger table.
    """

    def __init__(self, *args, **kwargs):
        self._source = _NativeLactEventSource(*args, **kwargs)
        self._triggered_tels_by_event_id = {}
        self._trigger_timing_by_event_id = {}
        self._trigger_timing_loaded = False
        self._waveform_timing_loaded = False
        self._waveform_time_centers_ns = ()
        self._waveform_time_reference = ""
        self._waveform_reference_by_event_id = {}
        self._ground_counts_by_event_id = {}
        self._input_filename = getattr(self._source, "input_filename", None)
        if self._input_filename is None and args:
            self._input_filename = str(args[0])
        self._event_id_mode = None

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __iter__(self):
        for event in self._source:
            self._remember_triggered_tels(event)
            yield event

    def __getitem__(self, item):
        event = self._source[item]
        self._remember_triggered_tels(event)
        return event

    def __len__(self):
        return len(self._source)

    def _read_root_triggered_tels(self, event_id):
        filename = self._input_filename
        if filename is None or event_id is None or not str(filename).endswith(".root"):
            return ()
        try:
            import ROOT

            root_file = ROOT.TFile.Open(str(filename))
            if not root_file or root_file.IsZombie():
                return ()
            tree = root_file.Get("observations")
            required = ("event_id", "telescope_id", "triggered")
            if tree is None or any(tree.GetBranch(name) is None for name in required):
                root_file.Close()
                return ()
            triggered = []
            for entry in range(tree.GetEntries()):
                tree.GetEntry(entry)
                if int(tree.event_id) == int(event_id) and bool(tree.triggered):
                    triggered.append(int(tree.telescope_id))
            root_file.Close()
            return tuple(sorted(set(triggered)))
        except Exception:
            return ()

    def _read_root_event_id_mode(self, root_file):
        if self._event_id_mode is not None:
            return self._event_id_mode
        self._event_id_mode = ""
        try:
            tree = root_file.Get("config")
            if tree is not None and tree.GetBranch("event_id_mode") is not None and tree.GetEntries() > 0:
                tree.GetEntry(0)
                self._event_id_mode = str(tree.event_id_mode).strip()
        except Exception:
            self._event_id_mode = ""
        return self._event_id_mode

    def _load_root_trigger_timing(self):
        """Load LACT observation trigger timing once for event-level plots."""

        if self._trigger_timing_loaded:
            return
        self._trigger_timing_loaded = True
        filename = self._input_filename
        if filename is None or not str(filename).lower().endswith(".root"):
            return
        root_file = None
        try:
            import ROOT

            root_file = ROOT.TFile.Open(str(filename))
            if not root_file or root_file.IsZombie():
                return
            tree = root_file.Get("observations")
            required = (
                "event_id",
                "telescope_id",
                "triggered",
                "trigger_time_ns",
            )
            if tree is None or any(
                tree.GetBranch(name) is None for name in required
            ):
                return
            has_geometric_delay = (
                tree.GetBranch("geometric_delay_ns") is not None
            )
            has_coincidence_time = (
                tree.GetBranch("coincidence_time_ns") is not None
            )
            has_first_trigger_time = (
                tree.GetBranch("trigger_first_time_ns") is not None
            )
            has_max_multiplicity_time = (
                tree.GetBranch("trigger_max_multiplicity_time_ns") is not None
            )
            for entry in range(tree.GetEntries()):
                tree.GetEntry(entry)
                if not bool(tree.triggered):
                    continue
                event_id = int(tree.event_id)
                telescope_id = int(tree.telescope_id)
                raw_time = float(tree.trigger_time_ns)
                first_trigger_time = (
                    float(tree.trigger_first_time_ns)
                    if has_first_trigger_time else raw_time
                )
                max_multiplicity_time = (
                    float(tree.trigger_max_multiplicity_time_ns)
                    if has_max_multiplicity_time else raw_time
                )
                geometric_delay = (
                    float(tree.geometric_delay_ns)
                    if has_geometric_delay else float("nan")
                )
                coincidence_time = (
                    float(tree.coincidence_time_ns)
                    if has_coincidence_time else float("nan")
                )
                if not has_coincidence_time and has_geometric_delay:
                    coincidence_time = raw_time + geometric_delay
                self._trigger_timing_by_event_id.setdefault(event_id, {})[
                    telescope_id
                ] = {
                    "trigger_time_ns": raw_time,
                    "trigger_first_time_ns": first_trigger_time,
                    "trigger_max_multiplicity_time_ns": max_multiplicity_time,
                    "trigger_diagnostics_available":
                        has_first_trigger_time and has_max_multiplicity_time,
                    "geometric_delay_ns": geometric_delay,
                    "coincidence_time_ns": coincidence_time,
                }
        except Exception:
            self._trigger_timing_by_event_id.clear()
        finally:
            if root_file:
                root_file.Close()

    def _load_root_waveform_timing(self):
        """Load the common waveform time axis and per-image reference times."""

        if self._waveform_timing_loaded:
            return
        self._waveform_timing_loaded = True
        filename = self._input_filename
        if filename is None or not str(filename).lower().endswith(".root"):
            return
        root_file = None
        try:
            import ROOT

            root_file = ROOT.TFile.Open(str(filename))
            if not root_file or root_file.IsZombie():
                return
            config = root_file.Get("waveform_config")
            if config is None or config.GetEntries() <= 0:
                return
            if config.GetBranch("time_centers_ns") is None:
                return
            config.GetEntry(0)
            self._waveform_time_centers_ns = tuple(
                float(value) for value in config.time_centers_ns
            )
            if config.GetBranch("time_reference") is not None:
                self._waveform_time_reference = str(
                    config.time_reference
                ).strip()

            observations = root_file.Get("observations")
            reference_branch = {
                "image_first": "time_first_ns",
                "image_mean": "time_mean_ns",
            }.get(self._waveform_time_reference)
            required = ("event_id", "telescope_id")
            if (
                observations is None
                or reference_branch is None
                or observations.GetBranch(reference_branch) is None
                or any(
                    observations.GetBranch(name) is None for name in required
                )
            ):
                return
            for entry in range(observations.GetEntries()):
                observations.GetEntry(entry)
                event_id = int(observations.event_id)
                telescope_id = int(observations.telescope_id)
                reference_time = float(
                    getattr(observations, reference_branch)
                )
                self._waveform_reference_by_event_id.setdefault(
                    event_id, {}
                )[telescope_id] = reference_time
        except Exception:
            self._waveform_time_centers_ns = ()
            self._waveform_time_reference = ""
            self._waveform_reference_by_event_id.clear()
        finally:
            if root_file:
                root_file.Close()

    def _read_root_ground_counts(self, event_or_id):
        filename = self._input_filename
        event_id = _event_id(event_or_id)
        if filename is None or event_id is None or not str(filename).endswith(".root"):
            return {}
        try:
            import ROOT

            root_file = ROOT.TFile.Open(str(filename))
            if not root_file or root_file.IsZombie():
                return {}
            tree = root_file.Get("corsika_events")
            required = ("event_id", "ground_gammas", "ground_electrons", "ground_hadrons", "ground_muons")
            if tree is None or any(tree.GetBranch(name) is None for name in required):
                root_file.Close()
                return {}
            event_id_mode = self._read_root_event_id_mode(root_file)
            has_shower_event_id = tree.GetBranch("shower_event_id") is not None

            def counts_from_current_entry():
                return {
                    "ground_gammas": float(tree.ground_gammas),
                    "ground_electrons": float(tree.ground_electrons),
                    "ground_hadrons": float(tree.ground_hadrons),
                    "ground_muons": float(tree.ground_muons),
                }

            # 1. Exact match on the ROOT output event id. This is the canonical
            # match when pylast.event_id and corsika_events.event_id are the same.
            for entry in range(tree.GetEntries()):
                tree.GetEntry(entry)
                if int(tree.event_id) == int(event_id):
                    counts = counts_from_current_entry()
                    root_file.Close()
                    return counts

            # 2. Match through shower_event_id. For event_array100, LACT_sim
            # output event ids are shower_event * 100 + array_id.
            shower_ids = list(_event_shower_ids(event_or_id))
            if event_id_mode == "event_array100" and abs(int(event_id)) >= 100:
                shower_ids.append(int(event_id) // 100)
            # Compatibility fallback: some native readers expose the CORSIKA
            # shower id as event.event_id, while the ROOT output event id is
            # event_array100 encoded.
            shower_ids.append(int(event_id))
            shower_ids = _unique(shower_ids)
            if has_shower_event_id and shower_ids:
                for entry in range(tree.GetEntries()):
                    tree.GetEntry(entry)
                    if int(tree.shower_event_id) in shower_ids:
                        counts = counts_from_current_entry()
                        root_file.Close()
                        return counts
            root_file.Close()
        except Exception:
            return {}
        return {}

    def _remember_triggered_tels(self, event):
        event_id = _event_id(event)
        if event_id is None:
            return ()
        triggered_tels = _native_triggered_tels(event)
        # The native LACT reader populates this list from the observations
        # tree, including the valid empty-trigger case. Do not reopen and scan
        # the complete ROOT tree once per event.
        self._triggered_tels_by_event_id[event_id] = triggered_tels
        return triggered_tels

    def get_triggered_tels(self, event_or_id):
        event_id = _event_id(event_or_id)
        if event_id is None:
            return []
        if not isinstance(event_or_id, int):
            triggered_tels = _native_triggered_tels(event_or_id)
            self._triggered_tels_by_event_id[event_id] = triggered_tels
            return list(triggered_tels)
        if event_id not in self._triggered_tels_by_event_id:
            triggered_tels = self._read_root_triggered_tels(event_id)
            if triggered_tels:
                self._triggered_tels_by_event_id[event_id] = triggered_tels
        return list(self._triggered_tels_by_event_id.get(event_id, ()))

    def get_ground_counts(self, event_or_id):
        event_id = _event_id(event_or_id)
        if event_id is None:
            return {}
        if event_id not in self._ground_counts_by_event_id:
            counts = self._read_root_ground_counts(event_or_id)
            if counts:
                self._ground_counts_by_event_id[event_id] = counts
        return dict(self._ground_counts_by_event_id.get(event_id, {}))

    def get_trigger_timing(self, event_or_id):
        """Return per-triggered-telescope LACT timing fields for one event.

        The mapping is empty for non-LACT inputs and older ROOT files without
        ``trigger_time_ns``. New geometrically corrected LACT ROOT files also
        provide ``geometric_delay_ns`` and ``coincidence_time_ns``.
        """

        event_id = _event_id(event_or_id)
        if event_id is None:
            return {}
        self._load_root_trigger_timing()
        return {
            telescope_id: dict(values)
            for telescope_id, values in self._trigger_timing_by_event_id.get(
                event_id, {}
            ).items()
        }

    def get_waveform_timing(self, event_or_id, telescope_id):
        """Return the LACT waveform axis and its time-reference metadata.

        ``time_centers_ns`` is the axis stored in ``waveform_config``. For
        ``image_first`` and ``image_mean`` references, ``absolute_time_ns``
        adds the corresponding observation-level reference time.
        """

        event_id = _event_id(event_or_id)
        if event_id is None:
            return {}
        self._load_root_waveform_timing()
        if not self._waveform_time_centers_ns:
            return {}
        reference_time = self._waveform_reference_by_event_id.get(
            event_id, {}
        ).get(int(telescope_id))
        if reference_time is None:
            reference_time = (
                float("nan")
                if self._waveform_time_reference in {
                    "image_first", "image_mean"
                }
                else 0.0
            )
        return {
            "time_centers_ns": list(self._waveform_time_centers_ns),
            "time_reference": self._waveform_time_reference,
            "reference_time_ns": float(reference_time),
            "absolute_time_ns": [
                float(reference_time + center)
                for center in self._waveform_time_centers_ns
            ],
        }
