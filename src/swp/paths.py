"""Where data outside the repository lives.

Scripts, the GUI and study analyses used to hard-code these as absolute ``D:/...`` strings, and
when the 2026-08-04 voltage sweep was moved under ``Claude/MI estimation/`` every one of them broke
silently. They now read the locations from here. Each can be overridden by an environment variable,
and the whole tree can be relocated at once with ``SWP_DATA_ROOT``:

=====================  ==========================  ==================================================
constant               environment variable        default
=====================  ==========================  ==================================================
``DATA_ROOT``          ``SWP_DATA_ROOT``           ``D:/Luuk van Knippenberg/Claude``
``VOLTAGE_SWEEP``      ``SWP_VOLTAGE_SWEEP``       ``<DATA_ROOT>/MI estimation/2026_08_04 voltage sweep``
``METRIC_EXPERIMENT``  ``SWP_METRIC_EXPERIMENT``   ``<VOLTAGE_SWEEP>/metric_experiment``
``CAENEN_SWE``         ``SWP_CAENEN_DIR``          ``<DATA_ROOT>/Data Caenen/SWE_results``
``INVIVO_SW``          ``SWP_INVIVO_SW``           ``<DATA_ROOT>/invivo_sw``
``INVIVO_0818``        ``SWP_INVIVO_0818``         ``D:/swp_iv`` (R-peak triggered 41 el / 61 el pair)
``RAW_DATA``           ``SWP_RAW_DATA``            ``Z:/raw_data`` (the 44-folder in-vivo study)
=====================  ==========================  ==================================================

All values are plain ``str`` so existing ``os.path.join`` / f-string code keeps working.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


DATA_ROOT = _env("SWP_DATA_ROOT", "D:/Luuk van Knippenberg/Claude")
VOLTAGE_SWEEP = _env("SWP_VOLTAGE_SWEEP",
                     os.path.join(DATA_ROOT, "MI estimation", "2026_08_04 voltage sweep"))
METRIC_EXPERIMENT = _env("SWP_METRIC_EXPERIMENT", os.path.join(VOLTAGE_SWEEP, "metric_experiment"))
CAENEN_SWE = _env("SWP_CAENEN_DIR", os.path.join(DATA_ROOT, "Data Caenen", "SWE_results"))
INVIVO_SW = _env("SWP_INVIVO_SW", os.path.join(DATA_ROOT, "invivo_sw"))
INVIVO_0818 = _env("SWP_INVIVO_0818", "D:/swp_iv")
RAW_DATA = _env("SWP_RAW_DATA", "Z:/raw_data")


def describe() -> str:
    """One line per location, marking the ones that do not exist on this machine."""
    rows = []
    for k in ("DATA_ROOT", "VOLTAGE_SWEEP", "METRIC_EXPERIMENT", "CAENEN_SWE", "INVIVO_SW",
              "INVIVO_0818", "RAW_DATA"):
        v = globals()[k]
        rows.append(f"{k:18s} {'ok     ' if os.path.isdir(v) else 'MISSING'} {v}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
