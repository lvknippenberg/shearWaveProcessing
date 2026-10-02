"""View sorting (swp.views): protocol block fit, voters/consensus on synthetic features, labels table."""
import numpy as np
import pandas as pd

from swp.views import acq_time
from swp.views.review import LABEL_COLUMNS, read_labels, save_subject
from swp.views.vote import block_fit, fit_head, table, vote
from swp.views.echoprime import COARSE_VIEWS, clip_indices


def test_block_fit_single_change_point():
    s = [1, 1, -1, 1, 1, -1, -1, 1, -1, -1]          # noisy PLAX block, then PSAX block
    assert list(block_fit(s)) == ["PLAX"] * 5 + ["PSAX"] * 5
    assert list(block_fit([-1, -1])) == ["PSAX", "PSAX"]


def test_clip_indices_span_loop():
    for n in (26, 32):
        for idx in clip_indices(n):
            assert len(idx) == 16 and idx[0] >= 0 and idx[-1] == n - 1 and np.all(np.diff(idx) >= 0)


def _synthetic(n_subj=6, rng=np.random.default_rng(0)):
    """Two well-separated views per subject + a subject offset; EchoPrime leans the right way."""
    dims = 1024
    proto = {v: rng.normal(size=dims) for v in ("PLAX", "PSAX")}
    vproto = {v: rng.normal(size=512) for v in ("PLAX", "PSAX")}
    items, truth = [], {}
    for s in range(n_subj):
        offset = rng.normal(size=dims) * 2
        views = ["PLAX"] * 6 + ["PSAX"] * 9
        for k, v in enumerate(views):
            folder = f"X_SW_data_1-Jan-2026_10-{k:02d}-00"
            ff = proto[v] + offset + rng.normal(size=(26, dims)) * 0.3
            prob = np.full((26, len(COARSE_VIEWS)), 0.01)
            prob[:, COARSE_VIEWS.index("Parasternal_Long" if v == "PLAX" else "Parasternal_Short")] = 0.9
            items.append((f"C{s:09d}", folder, {"frame_feat": ff.astype(np.float16), "frame_prob": prob,
                                                 "video_emb": vproto[v] + rng.normal(size=(4, 512)) * 0.3}))
            truth[(f"C{s:09d}", folder)] = v
    return items, truth


def test_vote_unanimous_on_separable_data():
    items, truth = _synthetic()
    df, F, V = table(items)
    y = np.array([1 if truth[(s, f)] == "PLAX" else 0 for s, f in zip(df.subject, df.folder)])
    head = fit_head(F, V, df.subject.to_numpy(), y)
    v = vote(df, F, V, head)
    assert not v.needs_review.any()
    assert (v.proposed == [truth[(s, f)] for s, f in zip(v.subject, v.folder)]).all()
    assert list(df.folder[df.subject == df.subject[0]]) == sorted(df.folder[df.subject == df.subject[0]], key=acq_time)


def test_save_subject_replaces_rows(tmp_path):
    p = tmp_path / "labels.csv"
    sub = pd.DataFrame({"subject": ["C1", "C1"], "folder": ["a", "b"], "proposed": ["PLAX", "PSAX"],
                        "needs_review": [False, True], "votes": ["L L L L", "S L L S"]})
    save_subject(p, sub, ["PLAX", "PSAX"])
    save_subject(p, sub, ["PLAX", "Unclear"], source="review")
    d = read_labels(p)
    assert list(d.columns) == LABEL_COLUMNS and len(d) == 2
    assert d.set_index("folder").label.to_dict() == {"a": "PLAX", "b": "Unclear"}
    assert d.set_index("folder").changed.to_dict() == {"a": False, "b": True}
