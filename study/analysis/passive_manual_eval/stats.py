"""Small statistics helpers (no statsmodels in the zea env): nested variance components by REML,
cluster bootstrap, within-cluster SD."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def reml_nested(y, subj, acq):
    """REML fit of y = mu + s_subj + a_acq(subj) + e  (all random intercepts, Gaussian).

    Returns dict(mu, var_subject, var_acq, var_resid, n, n_subj, n_acq). ``acq`` labels must be
    unique across subjects (e.g. the folder name)."""
    y = np.asarray(y, float)
    subj = np.asarray(subj)
    acq = np.asarray(acq)
    blocks = []
    for s in np.unique(subj):
        m = subj == s
        a = acq[m]
        ua, ia = np.unique(a, return_inverse=True)
        Za = np.zeros((m.sum(), len(ua)))
        Za[np.arange(m.sum()), ia] = 1
        blocks.append((y[m], Za))
    n = len(y)

    def nll(theta):
        vs, va, ve = np.exp(theta)
        ll, XtVX, XtVy = 0.0, 0.0, 0.0
        parts = []
        for yb, Za in blocks:
            nb = len(yb)
            V = vs * np.ones((nb, nb)) + va * Za @ Za.T + ve * np.eye(nb)
            L = np.linalg.cholesky(V)
            Vi1 = np.linalg.solve(V, np.ones(nb))
            Viy = np.linalg.solve(V, yb)
            XtVX += Vi1.sum()
            XtVy += Viy.sum()
            parts.append((yb, V, L))
            ll += 2 * np.log(np.diag(L)).sum()
        mu = XtVy / XtVX
        q = 0.0
        for yb, V, L in parts:
            rres = yb - mu
            q += rres @ np.linalg.solve(V, rres)
        return 0.5 * (ll + q + np.log(XtVX)), mu

    v0 = np.var(y) / 3 + 1e-6
    best = None
    for start in ([v0, v0, v0], [2 * v0, v0 / 4, v0], [v0 / 4, v0 / 4, 2 * v0]):
        res = minimize(lambda th: nll(th)[0], np.log(start), method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
        if best is None or res.fun < best.fun:
            best = res
    vs, va, ve = np.exp(best.x)
    mu = nll(best.x)[1]
    return dict(mu=float(mu), var_subject=float(vs), var_acq=float(va), var_resid=float(ve), n=n,
                n_subj=len(blocks), n_acq=len(np.unique(acq)))


def cluster_bootstrap(df, cluster, fn, n=500, seed=0):
    """Percentile CI of fn(df) resampling whole clusters (subjects) -> (est, lo, hi) arrays."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    est = np.asarray(fn(df), float)
    boots = []
    import pandas as pd
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        parts = []
        for j, k in enumerate(pick):
            g = groups[keys[k]].copy()
            g[cluster] = f"{keys[k]}#{j}"            # resampled twice = two different subjects
            if "folder" in g:
                g["folder"] = g["folder"] + f"#{j}"
            parts.append(g)
        try:
            boots.append(np.asarray(fn(pd.concat(parts)), float))
        except Exception:                             # noqa: BLE001 - degenerate resample
            continue
    b = np.array(boots)
    return est, np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)


def within_sd(values, groups):
    """Pooled within-group SD (groups with >= 2 values) and its degrees of freedom."""
    import pandas as pd
    s = pd.DataFrame(dict(v=values, g=groups)).groupby("g").v
    ss = s.apply(lambda x: ((x - x.mean()) ** 2).sum())
    dfree = s.count() - 1
    m = dfree > 0
    tot = dfree[m].sum()
    return (float(np.sqrt(ss[m].sum() / tot)) if tot else np.nan), int(tot)


def pct(sd_log):
    """SD on the natural-log scale -> approximate CV in percent."""
    return 100.0 * float(np.sqrt(np.exp(sd_log ** 2) - 1))
