"""Automatic RR-interval assessment of the acquisition trigger record.

Runs on every passive workflow when ECG data is present, and warns when the R-peak record is not
good enough to trust a cardiac phase - because a wrong phase turns an MVC into an AK and
silently falsifies the shear-wave-speed analysis that follows.

**Trigger record only.** This module never looks at the logged ``Signal`` waveform. That trace is
a secondary display stream, sampled by the Arduino at 100 Hz off the ECG monitor's *analog*
output (``ArduinoReadECG_request_micros.ino``), so it carries the monitor's own filter delay and
10 ms quantisation. ``ECG_trigger`` by contrast is a hardware interrupt timestamped with
``micros()``. Comparing the two makes the trigger look tens of milliseconds early when it is not;
see "Do not re-detect R-peaks from the ``Signal`` trace" in ``docs/ecg_timing.md``, where this was
established, and then re-derived and withdrawn a second time on 2026-09-21.

What is tolerated and what is not
---------------------------------
A **single** spurious trigger is recoverable: ``triggerlog.clean_r_peaks`` drops any trigger
arriving sooner than 0.6 x the reference RR, and one dropped beat out of ~50 leaves the reference
RR and every event phase intact. What is *not* recoverable is a record in which the R-peaks
themselves are in doubt - a fixed-rate pulse train instead of an ECG, or so many spurious or
missing triggers that the reference RR is no longer defined by real beats. Those must warn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .triggerlog import RR_PLAUSIBLE_MS, buffer_timing, clean_r_peaks, read_log

# An interval this far from the record's own median is an artefact or an ectopic beat.
RR_OUTLIER_FRAC = 0.30
# Up to this many corrected triggers is routine housekeeping, not a reason to warn.
MAX_CORRECTABLE = 1
# Above this fraction of short intervals the record is a pulse train, not an ECG.
FIXED_RATE_FRAC = 0.30
# Beat-to-beat variation above this is flagged for inspection (arrhythmia or a noisy lead).
RR_CV_WARN = 0.15


@dataclass
class RRCheck:
    folder: str = ""
    status: str = "no-ecg"        # ok / corrected / warn / unusable / no-ecg
    quality: str = "unknown"      # clean / extra triggers / variable rhythm / fixed-rate / sparse
    messages: list = field(default_factory=list)
    n_triggers: int = 0
    n_rr: int = 0
    n_corrected: int = 0
    n_rr_short: int = 0
    n_rr_long: int = 0
    n_rr_outliers: int = 0
    frac_rr_short: float = float("nan")
    short_rr_median_ms: float = float("nan")
    rr_median_ms: float = float("nan")
    rr_sd_ms: float = float("nan")
    rr_cv: float = float("nan")
    rr_min_ms: float = float("nan")
    rr_max_ms: float = float("nan")
    rmssd_ms: float = float("nan")
    hr_bpm: float = float("nan")
    gating_error_us: float = float("nan")   # passive block start vs its closest R-peak
    # --- the beats the passive block actually spans (what the phases are measured from) ---
    n_rr_local: int = 0
    n_local_bad: int = 0
    local_rr_min_ms: float = float("nan")
    local_rr_max_ms: float = float("nan")

    @property
    def trustworthy(self):
        """True when cardiac phases derived from this record can be believed."""
        return self.status in ("ok", "corrected")

    def as_row(self):
        d = asdict(self)
        d["messages"] = "; ".join(self.messages)
        return d

    def report(self, prefix="  [ECG] "):
        """One-line summary plus any warnings, for the workflow log."""
        out = []
        if self.status == "no-ecg":
            return [prefix + "no ECG trigger record in this folder - phases unavailable"]
        out.append(prefix + f"RR {self.rr_median_ms:.0f} ms ({self.hr_bpm:.0f} bpm), "
                            f"SD {self.rr_sd_ms:.0f} ms, CV {self.rr_cv:.3f}, "
                            f"n={self.n_rr} intervals, quality: {self.quality}")
        if np.isfinite(self.gating_error_us):
            out.append(prefix + f"passive block starts {self.gating_error_us:+.1f} us "
                                f"from its closest R-peak")
        for m in self.messages:
            out.append(prefix + m)
        return out


def assess_rr(folder, buffer=4):
    """Assess the trigger record of one acquisition folder -> :class:`RRCheck`."""
    res = RRCheck(folder=str(folder))
    log = read_log(folder)
    if log is None:
        res.messages.append("no trigger log")
        return res
    r = np.sort(np.asarray(log[0], float))
    r = r[r > 0]                      # unfilled circular-buffer slots read back as zero
    res.n_triggers = int(r.size)
    if r.size < 3:
        res.status, res.quality = "unusable", "sparse"
        res.messages.append(f"WARNING: only {r.size} ECG trigger(s) logged - "
                            f"cardiac phases cannot be assigned")
        return res

    rr = np.diff(r)
    short = rr < RR_PLAUSIBLE_MS[0]
    long_ = rr > RR_PLAUSIBLE_MS[1]
    res.n_rr_short, res.n_rr_long = int(short.sum()), int(long_.sum())
    res.frac_rr_short = float(short.sum() / rr.size)
    if short.any():
        res.short_rr_median_ms = float(np.median(rr[short]))
    good = rr[~short & ~long_]
    res.n_rr = int(good.size)
    if good.size < 3:
        res.status, res.quality = "unusable", "sparse"
        res.messages.append(f"WARNING: only {good.size} plausible RR interval(s) among "
                            f"{rr.size} - the trigger record is not an ECG")
        return res

    med = float(np.median(good))
    res.rr_median_ms = med
    res.hr_bpm = 60000.0 / med
    res.rr_sd_ms = float(good.std())
    res.rr_cv = float(good.std() / good.mean())
    res.rr_min_ms, res.rr_max_ms = float(good.min()), float(good.max())
    res.n_rr_outliers = int((np.abs(good - med) > RR_OUTLIER_FRAC * med).sum())
    if good.size > 1:
        res.rmssd_ms = float(np.sqrt(np.mean(np.diff(good) ** 2)))

    # how many triggers clean_r_peaks - the path the pipeline actually uses - has to discard
    kept, _ = clean_r_peaks(r)
    res.n_corrected = int(r.size - kept.size) if kept is not None else int(r.size)

    # --- classify ---------------------------------------------------------------------------
    # Every condition is evaluated independently and the worst one decides the status; an early
    # `elif` here would let a high-variability record pass merely because its single spurious
    # trigger was correctable.
    RANK = {"ok": 0, "corrected": 1, "warn": 2, "unusable": 3}
    status, quality = "ok", "clean"

    def raise_to(new_status, new_quality, message):
        nonlocal status, quality
        res.messages.append(message)
        if RANK[new_status] > RANK[status]:
            status, quality = new_status, new_quality

    if res.frac_rr_short > FIXED_RATE_FRAC:
        raise_to("unusable", "fixed-rate",
                 f"WARNING: {res.n_rr_short} of {rr.size} intervals are shorter than "
                 f"{RR_PLAUSIBLE_MS[0]:.0f} ms (median {res.short_rr_median_ms:.0f} ms). This is "
                 f"a fixed-rate pulse train, not an ECG - every cardiac phase from this folder "
                 f"is meaningless and the shear-wave results must not be labelled MVC/AVC.")
    elif res.n_corrected > MAX_CORRECTABLE:
        raise_to("warn", "extra triggers",
                 f"WARNING: {res.n_corrected} triggers had to be discarded as spurious. One is "
                 f"routine; this many means the R-peaks are uncertain, so event phases - and any "
                 f"MVC/AVC label built on them - may be wrong.")
    elif res.n_corrected:
        raise_to("corrected", "extra triggers",
                 f"{res.n_corrected} spurious trigger discarded (recoverable; reference RR and "
                 f"phases are unaffected).")

    if res.n_rr_long:
        raise_to("warn", "missed beats",
                 f"WARNING: {res.n_rr_long} interval(s) longer than "
                 f"{RR_PLAUSIBLE_MS[1]:.0f} ms - a beat was missed, so an event may be timed "
                 f"against the wrong R-peak.")
    if res.rr_cv > RR_CV_WARN:
        raise_to("warn", "variable rhythm",
                 f"WARNING: beat-to-beat variation is high (CV {res.rr_cv:.2f}, RR "
                 f"{res.rr_min_ms:.0f}-{res.rr_max_ms:.0f} ms). Phases are measured against the "
                 f"preceding R-peak so remain valid, but QS2 - and therefore the AVC label - is "
                 f"derived from the median heart rate and is unreliable here.")
    if res.n_rr_outliers:
        raise_to("warn", "variable rhythm",
                 f"WARNING: {res.n_rr_outliers} RR interval(s) differ from the median by more "
                 f"than {RR_OUTLIER_FRAC:.0%} - check for ectopic beats before trusting AVC "
                 f"labels.")
    res.status, res.quality = status, quality

    # --- gating, and the LOCAL beats that actually matter -----------------------------------
    bt = buffer_timing(folder, buffer)
    if bt is not None and r.size:
        d = r - bt.t0_ms
        res.gating_error_us = float(d[int(np.argmin(np.abs(d)))] * 1e3)
        _assess_local(res, r, rr, bt, med)
    return res


def _assess_local(res, r, rr, bt, med):
    """Judge only the beats the passive block spans, and let that decide the verdict.

    The trigger record covers ~40 s while the ultrafast block is ~1.2 s - about 1.5 beats. An
    ectopic beat or a dropped trigger 30 s before the acquisition says nothing about the phase of
    the events actually measured, so judging the whole record excludes folders for arrhythmia that
    never touches the data. The global statistics stay in the result as context; the **verdict**
    is decided here, on the beat containing the block plus one beat either side.
    """
    block_ms = bt.n_frames * bt.frame_ms
    lo = bt.t0_ms - 1.5 * med
    hi = bt.t0_ms + block_ms + 1.5 * med
    starts = r[:-1]
    sel = (starts >= lo) & (starts <= hi)
    local = rr[sel]
    res.n_rr_local = int(local.size)
    if local.size == 0:
        return
    res.local_rr_min_ms, res.local_rr_max_ms = float(local.min()), float(local.max())
    bad_short = local < RR_PLAUSIBLE_MS[0]
    bad_long = local > RR_PLAUSIBLE_MS[1]
    bad_outlier = np.abs(local - med) > RR_OUTLIER_FRAC * med
    bad = bad_short | bad_long | bad_outlier
    res.n_local_bad = int(bad.sum())

    # A record that is not an ECG at all stays unusable however clean the local beats look.
    if res.status == "unusable":
        return
    if res.n_local_bad == 0:
        res.status = "corrected" if res.n_corrected else "ok"
        res.quality = "clean" if not res.n_corrected else "extra triggers"
        res.messages = [m for m in res.messages if not m.startswith("WARNING")]
        res.messages.append(
            f"{res.n_rr_local} beat(s) span the passive block and all are plausible "
            f"({res.local_rr_min_ms:.0f}-{res.local_rr_max_ms:.0f} ms); "
            f"record-level irregularities elsewhere do not affect these phases.")
    else:
        res.status = "warn"
        res.quality = "irregular at the block"
        res.messages.append(
            f"WARNING: {res.n_local_bad} of {res.n_rr_local} beat(s) spanning the passive block "
            f"are implausible or more than {RR_OUTLIER_FRAC:.0%} off the median "
            f"({res.local_rr_min_ms:.0f}-{res.local_rr_max_ms:.0f} ms vs {med:.0f} ms) - "
            f"the phases of these events may be wrong.")
