# -*- coding: utf-8 -*-
"""30_identify.py  2026-11-25
Identification and robustness work both papers need before submission.

A1  adjustment ladder with within-country R^2 at each step -> Oster (2019) delta
    for the bound beta = 0, so the coefficient-stability claim is computed and
    not merely cited.
A2  what the null excludes: CI, SD units, and a formal equivalence test.
A3  SPLIT-SCHOOL meta-regression.  The compliance gap and the achievement slope
    are computed on DISJOINT halves of each country's schools, so the two axes
    of the meta-regression share no sampling noise.  Repeated over 20 random
    splits, both directions each.
A4  country inclusion audit: which of the countries drop out, at which step.

B1  all six IC179 items, one ban coefficient each, ONE sample, with a BRR
    covariance matrix over the six so differences between items get a standard
    error and a Wald test.  This is the acquiescence test.
B2  the same six coefficients on person-centred (ipsatised) outcomes, which
    removes individual response style entirely.
B3  standardised effect sizes.
B4  did schools have discretion?  country ban prevalence as a moderator.
"""
import pandas as pd, numpy as np, pyreadstat, json, time, math, os
t0 = time.time()
RW     = [f"W_FSTURWT{i}" for i in range(1, 81)]
MATHPV = [f"PV{i}MATH" for i in range(1, 11)]
REG    = ["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
ITEMS  = ["IC179Q01JA","IC179Q02JA","IC179Q03JA","IC179Q04JA","IC179Q05JA","IC179Q06JA"]
ILAB   = {"IC179Q01JA":"Phones should not be brought to class",
          "IC179Q02JA":"Laptops should not be brought to class",
          "IC179Q03JA":"Students should help decide the device rules",
          "IC179Q04JA":"The school should filter social media",
          "IC179Q05JA":"The school should filter online games",
          "IC179Q06JA":"Teachers should monitor what students do on laptops"}
# how close the attitude object sits to the rule the school actually imposes
PROX   = {"IC179Q01JA":"the rule itself","IC179Q02JA":"adjacent device",
          "IC179Q03JA":"voice, not restriction","IC179Q04JA":"different instrument",
          "IC179Q05JA":"different instrument","IC179Q06JA":"different instrument"}
CTRL5  = ["ESCS","DISCLIM","sch_escs","private","schsize"]
CTRL6  = CTRL5 + ["female"]
def pval(z): return math.erfc(abs(z)/math.sqrt(2))

CACHE = "out/prep30.pkl"
if os.path.exists(CACHE):
    d = pd.read_pickle(CACHE); print(f"cache {len(d):,} rows | {time.time()-t0:.0f}s", flush=True)
else:
    BASE = ["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM","IC170Q02JA"] + ITEMS
    d, _ = pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE + MATHPV + RW)
    print(f"  stu {len(d):,} | {time.time()-t0:.0f}s", flush=True)
    print("\n=== raw item codes BEFORE any masking (checking for out-of-range) ===", flush=True)
    for it in ITEMS:
        vc = d[it].value_counts(dropna=False).sort_index()
        print(f"  {it}: " + "  ".join(f"{k}:{v:,}" for k, v in vc.items()), flush=True)
    s, _ = pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
            usecols=["CNTSCHID","SC013Q01TA","SC002Q01TA","SC002Q02TA"] + REG)
    d = d.merge(s, on="CNTSCHID", how="inner")
    d = d[d.IC170Q02JA.notna() & (d.IC170Q02JA <= 5)]
    def yesno(c): return c.map({1.0:1.0, 2.0:0.0})
    for c in REG: d["p_"+c] = yesno(d[c])
    d["ban"]      = d["p_SC190Q02JA"]
    d["private"]  = d.SC013Q01TA.map({1.0:0.0, 2.0:1.0})
    d["schsize"]  = np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1, min_count=1))
    d["sch_escs"] = d.groupby("CNTSCHID").ESCS.transform("mean")
    d["female"]   = d.ST004D01T.map({1.0:1.0, 2.0:0.0})
    d["heavy"]    = (d.IC170Q02JA >= 5).astype(float)
    d["math"]     = d[MATHPV].mean(axis=1)
    for it in ITEMS: d[it] = d[it].where(d[it] <= 4)
    d.to_pickle(CACHE)
    print(f"prepared {len(d):,} rows | {time.time()-t0:.0f}s", flush=True)

out = {}

# ---------- shared weighted-LS machinery -------------------------------------
def fit(M, Y, w, codes=None, G=None, within=True, r2=False):
    """weighted LS. Y may be (n,) or (n,m) -> coefficients (p,) or (p,m)."""
    Y2 = Y[:, None] if Y.ndim == 1 else Y
    if within:
        den = np.bincount(codes, weights=w, minlength=G)
        def dm(A):
            o = np.empty_like(A)
            for j in range(A.shape[1]):
                o[:, j] = A[:, j] - (np.bincount(codes, weights=w*A[:, j], minlength=G)/den)[codes]
            return o
        X = dm(M); Yc = dm(Y2)
    else:
        X = np.column_stack([np.ones(len(M)), M]); Yc = Y2
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X*sw[:, None], Yc*sw[:, None], rcond=None)
    if not r2:
        return (b[:, 0] if Y.ndim == 1 else b)
    res = Yc - X@b
    ssr = (w[:, None]*res**2).sum(axis=0); sst = (w[:, None]*Yc**2).sum(axis=0)
    return (b[:, 0] if Y.ndim == 1 else b), 1.0 - ssr/sst

print("\n" + "="*78, flush=True)
print("A1  adjustment ladder with R^2, and the Oster delta for beta = 0", flush=True)
print("="*78, flush=True)
dA = d.dropna(subset=["ban","math"]+CTRL5).copy()
codesA, _ = pd.factorize(dA.CNT.values); GA = codesA.max()+1
wA = dA.W_FSTUWT.values.astype(float); yA = dA.math.values.astype(float)
Mban = dA[["ban"]].values.astype(float)
Mfull = dA[["ban"]+CTRL5].values.astype(float)
b0, r0 = fit(Mban,  yA, wA, codesA, GA, True, r2=True)
b1, r1 = fit(Mfull, yA, wA, codesA, GA, True, r2=True)
beta_unc, R_unc = float(b0[0]), float(r0[0])
beta_ctl, R_ctl = float(b1[0]), float(r1[0])
print(f"  within country, ban only          beta {beta_unc:+7.3f}   R2 {R_unc:.5f}", flush=True)
print(f"  within country, ban + 5 controls  beta {beta_ctl:+7.3f}   R2 {R_ctl:.5f}", flush=True)
print(f"  observables account for {100*(1-beta_ctl/beta_unc):.1f}% of the within-country association", flush=True)
osters = {}
for tag, Rmax in [("1.3x", min(1.0, 1.3*R_ctl)), ("2.0x", min(1.0, 2.0*R_ctl)), ("1.0", 1.0)]:
    num = beta_ctl*(R_ctl - R_unc); den = (beta_unc - beta_ctl)*(Rmax - R_ctl)
    osters[tag] = dict(R_max=Rmax, delta=float(num/den) if den != 0 else None)
    print(f"  Oster delta, R_max = {tag:<5} ({Rmax:.4f}):  delta = {num/den:+.3f}", flush=True)
out["A_oster"] = dict(beta_uncontrolled=beta_unc, R2_uncontrolled=R_unc,
                      beta_controlled=beta_ctl, R2_controlled=R_ctl,
                      share_explained_pct=100*(1-beta_ctl/beta_unc),
                      n=len(dA), deltas=osters)

print("\n" + "="*78, flush=True)
print("A2  what the null excludes", flush=True)
print("="*78, flush=True)
sdm = float(np.sqrt(np.average((yA - np.average(yA, weights=wA))**2, weights=wA)))
BETA, SE = -2.6300, 1.9300          # Table 1 row 5, BRR + Rubin, from fixed_20261124.json
lo, hi = BETA-1.96*SE, BETA+1.96*SE
print(f"  weighted SD of mathematics in the analytic sample: {sdm:.1f} points", flush=True)
print(f"  Table 1 row 5: {BETA:+.2f} (SE {SE:.2f}), 95% CI [{lo:+.2f}, {hi:+.2f}]", flush=True)
print(f"                 = {BETA/sdm:+.4f} SD, CI [{lo/sdm:+.4f}, {hi/sdm:+.4f}] SD", flush=True)
eq = {}
def Phi(x): return 0.5*math.erfc(-x/math.sqrt(2))
for lab, sesoi in [("5 points", 5.0), ("10 points", 10.0),
                   ("0.1 SD", 0.1*sdm), ("0.2 SD", 0.2*sdm)]:
    p1 = Phi(-(BETA + sesoi)/SE)      # reject beta <= -sesoi
    p2 = Phi((BETA - sesoi)/SE)       # reject beta >= +sesoi
    ptost = max(p1, p2)
    eq[lab] = dict(sesoi=sesoi, p_tost=float(ptost), equivalent=bool(ptost < 0.05))
    print(f"  equivalence vs +/-{lab:<10} ({sesoi:5.1f} pts): TOST p = {ptost:.4g}"
          f"   {'EQUIVALENT' if ptost<0.05 else 'not established'}", flush=True)
out["A_equivalence"] = dict(beta=BETA, se=SE, ci=[lo, hi], sd_math=sdm,
                            beta_sd=BETA/sdm, ci_sd=[lo/sdm, hi/sdm], tost=eq)
print(f"\n  [A1+A2 done {time.time()-t0:.0f}s]", flush=True)

print("\n" + "="*78, flush=True)
print("A3  split-school meta-regression: the two axes share no sampling noise", flush=True)
print("="*78, flush=True)
def mm(X, y, v):
    t2 = 0.0
    for _ in range(200):
        Wt = 1/(v+t2); XtWX = X.T@(Wt[:, None]*X)
        b = np.linalg.solve(XtWX, X.T@(Wt*y)); res = y - X@b
        P = np.diag(Wt) - (Wt[:, None]*X)@np.linalg.solve(XtWX, (X*Wt[:, None]).T)
        t2n = max(0.0, (float((Wt*res**2).sum()) - (len(y)-X.shape[1]))/np.trace(P))
        if abs(t2n-t2) < 1e-12: t2 = t2n; break
        t2 = t2n
    Wt = 1/(v+t2); XtWX = X.T@(Wt[:, None]*X)
    b = np.linalg.solve(XtWX, X.T@(Wt*y)); cov = np.linalg.inv(XtWX)
    return b, np.sqrt(np.diag(cov)), t2

dS = d.dropna(subset=["ban","math","heavy"]+CTRL5).copy()
pack = {}
for c in sorted(dS.CNT.unique()):
    g = dS[dS.CNT == c]
    if g.ban.nunique() < 2: continue
    pack[c] = dict(sch=g.CNTSCHID.values,
                   X=g[["ban"]+CTRL5].values.astype(float),
                   y=g.math.values.astype(float),
                   heavy=g.heavy.values.astype(float),
                   w0=g.W_FSTUWT.values.astype(float),
                   rw=g[RW].values.astype(float))
print(f"  {len(pack)} countries usable, {len(dS):,} students", flush=True)

def gap_of(P, mask, w):
    m1 = mask & (P["X"][:, 0] == 1); m0 = mask & (P["X"][:, 0] == 0)
    if w[m1].sum() <= 0 or w[m0].sum() <= 0: return None
    return 100.0*(np.average(P["heavy"][m1], weights=w[m1]) -
                  np.average(P["heavy"][m0], weights=w[m0]))

def beta_of(P, mask, w):
    Xm = P["X"][mask]
    if len(np.unique(Xm[:, 0])) < 2: return None
    Xi = np.column_stack([np.ones(mask.sum()), Xm]); sw = np.sqrt(w[mask])
    b, *_ = np.linalg.lstsq(Xi*sw[:, None], P["y"][mask]*sw, rcond=None)
    return float(b[1])

# baseline on the same PV-mean outcome, no split, for a like-for-like comparison
base_rows = []
for c, P in pack.items():
    full = np.ones(len(P["y"]), bool)
    g = gap_of(P, full, P["w0"]); b = beta_of(P, full, P["w0"])
    if g is None or b is None: continue
    reps = [beta_of(P, full, P["rw"][:, k]) for k in range(80)]
    reps = [x for x in reps if x is not None]
    se = math.sqrt(sum((x-b)**2 for x in reps)/20.0)
    base_rows.append((c, g, b, se))
gb = np.array([r[1] for r in base_rows]); bb = np.array([r[2] for r in base_rows])
vb = np.array([r[3] for r in base_rows])**2
Bb, Sb, T2b = mm(np.column_stack([np.ones(len(gb)), gb]), bb, vb)
print(f"  baseline, NO split, PV-mean outcome: slope {Bb[1]:+.4f} (SE {Sb[1]:.4f}, "
      f"p {pval(Bb[1]/Sb[1]):.3g}), k={len(gb)}", flush=True)
out["A_split_baseline"] = dict(slope=float(Bb[1]), se=float(Sb[1]),
                               p=float(pval(Bb[1]/Sb[1])), k=len(gb), tau2=float(T2b))

R_SPLITS = 20
slopes = []
for s_ in range(R_SPLITS):
    rng = np.random.default_rng(10_000 + s_)
    halves = {}
    for c, P in pack.items():
        sch = P["sch"]; ban0 = P["X"][:, 0]
        usch, idx = np.unique(sch, return_index=True)
        ubn = ban0[idx]
        assign = np.empty(len(usch), int)
        for val in (0.0, 1.0):
            w_ = np.where(ubn == val)[0]; rng.shuffle(w_)
            assign[w_[:len(w_)//2]] = 0; assign[w_[len(w_)//2:]] = 1
        halves[c] = pd.Series(assign, index=usch).reindex(sch).values
    for direction in (0, 1):
        rows = []
        for c, P in pack.items():
            mg = halves[c] == direction          # gap half
            mb = halves[c] == (1-direction)      # beta half
            g = gap_of(P, mg, P["w0"]); b = beta_of(P, mb, P["w0"])
            if g is None or b is None: continue
            reps = [beta_of(P, mb, P["rw"][:, k]) for k in range(80)]
            reps = [x for x in reps if x is not None]
            if len(reps) < 40: continue
            se = math.sqrt(sum((x-b)**2 for x in reps)/20.0)
            rows.append((c, g, b, se))
        gx = np.array([r[1] for r in rows]); by = np.array([r[2] for r in rows])
        vy = np.array([r[3] for r in rows])**2
        B_, S_, T_ = mm(np.column_stack([np.ones(len(gx)), gx]), by, vy)
        slopes.append(dict(split=s_, direction=direction, k=len(gx),
                           slope=float(B_[1]), se=float(S_[1]),
                           p=float(pval(B_[1]/S_[1])), tau2=float(T_)))
    print(f"  split {s_+1:2d}/{R_SPLITS}: slopes {slopes[-2]['slope']:+.4f} "
          f"(p {slopes[-2]['p']:.3g}) / {slopes[-1]['slope']:+.4f} "
          f"(p {slopes[-1]['p']:.3g})   [{time.time()-t0:.0f}s]", flush=True)
sl = np.array([x["slope"] for x in slopes]); pp = np.array([x["p"] for x in slopes])
print(f"\n  {len(sl)} split-half refits: median slope {np.median(sl):+.4f}  "
      f"range {sl.min():+.4f} .. {sl.max():+.4f}", flush=True)
print(f"  significant at 5%: {int((pp<0.05).sum())}/{len(pp)};  positive: {int((sl>0).sum())}/{len(sl)}", flush=True)
out["A_split"] = dict(n_refits=len(sl), median=float(np.median(sl)),
                      lo=float(sl.min()), hi=float(sl.max()),
                      n_sig=int((pp < 0.05).sum()), n_pos=int((sl > 0).sum()),
                      detail=slopes)

print("\n" + "="*78, flush=True)
print("A4  country inclusion audit", flush=True)
print("="*78, flush=True)
au, _ = pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV",
          usecols=["CNT","CNTSCHID","W_FSTUWT","IC170Q02JA","ESCS","PV1MATH"]+ITEMS)
tot = au.CNT.nunique()
has_ict = au.groupby("CNT").IC170Q02JA.apply(lambda x: x.notna().mean())
has_att = au.groupby("CNT")[ITEMS[0]].apply(lambda x: x.notna().mean())
inA = set(pack.keys()); inB = set()
print(f"  countries in the student file: {tot}", flush=True)
print(f"  with the optional ICT module (>=50% answering phone-use): "
      f"{int((has_ict>=0.5).sum())}", flush=True)
print(f"  with the attitude item (>=50%): {int((has_att>=0.5).sum())}", flush=True)
print(f"  in Paper A's analytic sample: {len(inA)}", flush=True)
drop = sorted(set(au.CNT.unique()) - inA)
print(f"  dropped ({len(drop)}): {' '.join(drop)}", flush=True)
cmp_ = []
for lab, sel in [("included", lambda x: x in inA), ("dropped", lambda x: x not in inA)]:
    g = au[au.CNT.map(sel)]
    cmp_.append(dict(group=lab, n_countries=int(g.CNT.nunique()),
                     mean_math=float(np.average(g.PV1MATH.dropna(),
                        weights=g.loc[g.PV1MATH.notna(), "W_FSTUWT"])),
                     mean_escs=float(np.average(g.ESCS.dropna(),
                        weights=g.loc[g.ESCS.notna(), "W_FSTUWT"]))))
    print(f"  {lab:<9} k={cmp_[-1]['n_countries']:>3}  mean math "
          f"{cmp_[-1]['mean_math']:6.1f}  mean ESCS {cmp_[-1]['mean_escs']:+.3f}", flush=True)
out["A_audit"] = dict(total_countries=int(tot), n_ict=int((has_ict >= 0.5).sum()),
                      n_att=int((has_att >= 0.5).sum()), n_analytic=len(inA),
                      dropped=drop, compare=cmp_)
del au

print("\n" + "="*78, flush=True)
print("B1  all six attitude items, one sample, BRR covariance over the six", flush=True)
print("     -> does the ban raise every item equally?  (acquiescence test)", flush=True)
print("="*78, flush=True)
dB = d.dropna(subset=["ban"]+CTRL6+ITEMS).copy()
codesB, _ = pd.factorize(dB.CNT.values); GB = codesB.max()+1
MB  = dB[["ban"]+CTRL6].values.astype(float)
YB  = dB[ITEMS].values.astype(float)
w0B = dB.W_FSTUWT.values.astype(float)
RWB = dB[RW].values.astype(float)
print(f"  sample {len(dB):,} students, {dB.CNT.nunique()} countries, "
      f"{dB.CNTSCHID.nunique():,} schools (all six items answered)", flush=True)

def ban_coefs(Y, w):
    return fit(MB, Y, w, codesB, GB, True)[0, :]

def brr_cov(Y):
    b = ban_coefs(Y, w0B)
    reps = np.empty((80, Y.shape[1]))
    for k in range(80):
        reps[k, :] = ban_coefs(Y, RWB[:, k])
    dv = reps - b
    V = (dv[:, :, None]*dv[:, None, :]).sum(axis=0)/20.0
    return b, V

bR, VR = brr_cov(YB)
seR = np.sqrt(np.diag(VR))
sdI = np.array([math.sqrt(np.average((YB[:, j]-np.average(YB[:, j], weights=w0B))**2,
                                     weights=w0B)) for j in range(6)])
print(f"\n  {'item':<54}{'beta':>9}{'SE':>8}{'p':>11}{'SD units':>10}", flush=True)
B_items = []
for j, it in enumerate(ITEMS):
    row = dict(item=it, label=ILAB[it], proximity=PROX[it], beta=float(bR[j]),
               se=float(seR[j]), p=float(pval(bR[j]/seR[j])),
               sd=float(sdI[j]), beta_sd=float(bR[j]/sdI[j]))
    B_items.append(row)
    print(f"  {ILAB[it][:52]:<54}{bR[j]:+9.4f}{seR[j]:8.4f}{row['p']:11.3g}"
          f"{row['beta_sd']:+10.4f}", flush=True)
out["B_items6"] = B_items

# differences against the item that names the actual rule
print(f"\n  differences against Q01 (the rule itself), BRR SE on the difference:", flush=True)
diffs = []
for j in range(1, 6):
    dlt = bR[0]-bR[j]
    sd_ = math.sqrt(max(VR[0, 0]+VR[j, j]-2*VR[0, j], 1e-18))
    diffs.append(dict(vs=ITEMS[j], label=ILAB[ITEMS[j]], diff=float(dlt),
                      se=float(sd_), p=float(pval(dlt/sd_))))
    print(f"    Q01 - {ITEMS[j]}  {dlt:+.4f}  SE {sd_:.4f}  p {pval(dlt/sd_):.3g}", flush=True)
out["B_item_diffs"] = diffs

# Wald test: are all six ban coefficients equal?
C = np.zeros((5, 6))
for j in range(5): C[j, 0] = 1.0; C[j, j+1] = -1.0
Cb = C@bR; CVC = C@VR@C.T
W = float(Cb@np.linalg.solve(CVC, Cb))
from math import lgamma
def chi2_sf(x, k):
    # regularised upper incomplete gamma Q(k/2, x/2), series/CF
    a, xx = k/2.0, x/2.0
    if xx < a+1:
        s, term = 1.0/a, 1.0/a
        for n in range(1, 500):
            term *= xx/(a+n); s += term
            if abs(term) < abs(s)*1e-15: break
        return 1.0 - s*math.exp(-xx+a*math.log(xx)-lgamma(a))
    FPMIN = 1e-300                       # c starts at 1/FPMIN, not FPMIN
    b_, c_, dd = xx+1.0-a, 1.0/FPMIN, 1.0/(xx+1.0-a)
    h = dd
    for i in range(1, 500):
        an = -i*(i-a); b_ += 2.0
        dd = an*dd+b_;  dd = FPMIN if abs(dd) < FPMIN else dd
        c_ = b_+an/c_;  c_ = FPMIN if abs(c_) < FPMIN else c_
        dd = 1.0/dd; de = dd*c_; h *= de
        if abs(de-1.0) < 1e-15: break
    return h*math.exp(-xx+a*math.log(xx)-lgamma(a))
pW = chi2_sf(W, 5)
print(f"\n  Wald test, all six ban coefficients equal: chi2 = {W:.2f} on 5 df, p = {pW:.4g}", flush=True)
print("  -> a uniform acquiescence shift is " +
      ("REJECTED" if pW < 0.05 else "NOT rejected"), flush=True)
out["B_wald_equal"] = dict(chi2=W, df=5, p=float(pW), rejects_uniform=bool(pW < 0.05))

print("\n" + "="*78, flush=True)
print("B2  person-centred (ipsatised) outcomes: response style removed entirely", flush=True)
print("="*78, flush=True)
pm = YB.mean(axis=1)
YI = YB - pm[:, None]
bI, VI = brr_cov(YI)
seI = np.sqrt(np.diag(VI))
print(f"  {'item':<54}{'beta':>9}{'SE':>8}{'p':>11}", flush=True)
B_ips = []
for j, it in enumerate(ITEMS):
    B_ips.append(dict(item=it, label=ILAB[it], proximity=PROX[it],
                      beta=float(bI[j]), se=float(seI[j]), p=float(pval(bI[j]/seI[j]))))
    print(f"  {ILAB[it][:52]:<54}{bI[j]:+9.4f}{seI[j]:8.4f}{pval(bI[j]/seI[j]):11.3g}", flush=True)
out["B_ipsatised"] = B_ips
# the person mean itself: does the ban shift general agreement?
bPM = fit(MB, pm, w0B, codesB, GB, True)[0]
rPM = [fit(MB, pm, RWB[:, k], codesB, GB, True)[0] for k in range(80)]
sPM = math.sqrt(sum((x-bPM)**2 for x in rPM)/20.0)
print(f"\n  the person's MEAN across all six items: {bPM:+.4f} SE {sPM:.4f} "
      f"p {pval(bPM/sPM):.3g}", flush=True)
out["B_personmean"] = dict(beta=float(bPM), se=float(sPM), p=float(pval(bPM/sPM)))

print("\n" + "="*78, flush=True)
print("B4  did schools have discretion?  country ban prevalence as a moderator", flush=True)
print("="*78, flush=True)
rows = []
for c in sorted(dB.CNT.unique()):
    g = dB[dB.CNT == c]
    if g.ban.nunique() < 2 or len(g) < 200: continue
    Xg = g[["ban"]+CTRL6].values.astype(float); yg = g[ITEMS[0]].values.astype(float)
    wg = g.W_FSTUWT.values.astype(float); rg = g[RW].values.astype(float)
    def bof(w):
        Xi = np.column_stack([np.ones(len(Xg)), Xg]); sw = np.sqrt(w)
        bb, *_ = np.linalg.lstsq(Xi*sw[:, None], yg*sw, rcond=None); return float(bb[1])
    b_ = bof(wg); se_ = math.sqrt(sum((bof(rg[:, k])-b_)**2 for k in range(80))/20.0)
    sch = g.groupby("CNTSCHID").ban.first()
    prev = float(sch.mean())
    rows.append(dict(CNT=c, beta=b_, se=se_, prev=prev,
                     discretion=prev*(1-prev), n=len(g), n_sch=int(len(sch))))
rr = pd.DataFrame(rows)
print(f"  {len(rr)} countries;  ban prevalence median {rr.prev.median():.3f}, "
      f"range {rr.prev.min():.3f} .. {rr.prev.max():.3f}", flush=True)
yv = rr.beta.values; vv = rr.se.values**2
B0, S0, t0_ = mm(np.column_stack([np.ones(len(rr))]), yv, vv)
print(f"  pooled random-effects mean: {B0[0]:+.4f} (SE {S0[0]:.4f}, "
      f"p {pval(B0[0]/S0[0]):.3g}), tau2 {t0_:.5f}", flush=True)
mods = {}
for nm, xv in [("prevalence", rr.prev.values), ("discretion", rr.discretion.values)]:
    Bm, Sm, tm = mm(np.column_stack([np.ones(len(rr)), xv]), yv, vv)
    mods[nm] = dict(slope=float(Bm[1]), se=float(Sm[1]),
                    p=float(pval(Bm[1]/Sm[1])), tau2=float(tm),
                    R2=float(100*(1-tm/t0_)) if t0_ > 0 else None)
    print(f"  moderator {nm:<12} slope {Bm[1]:+.4f} (SE {Sm[1]:.4f}, "
          f"p {pval(Bm[1]/Sm[1]):.3g})   tau2 {tm:.5f}", flush=True)
# high-prevalence stratum: schools had least room to choose
for lab, m in [("prevalence >= 0.80", rr.prev >= 0.80),
               ("prevalence 0.30-0.70", (rr.prev >= 0.30) & (rr.prev <= 0.70)),
               ("prevalence <= 0.20", rr.prev <= 0.20)]:
    if m.sum() < 3: print(f"  {lab:<22} k={int(m.sum())} (too few)", flush=True); continue
    Bs, Ss, _ = mm(np.column_stack([np.ones(int(m.sum()))]), yv[m.values], vv[m.values])
    mods[lab] = dict(k=int(m.sum()), beta=float(Bs[0]), se=float(Ss[0]),
                     p=float(pval(Bs[0]/Ss[0])))
    print(f"  {lab:<22} k={int(m.sum()):>3}  mean beta {Bs[0]:+.4f} "
          f"(SE {Ss[0]:.4f}, p {pval(Bs[0]/Ss[0]):.3g})", flush=True)
out["B_discretion"] = dict(pooled=dict(beta=float(B0[0]), se=float(S0[0]), tau2=float(t0_)),
                           moderators=mods, countries=rows)

json.dump(out, open("out/ident_20261125.json", "w"), indent=1)
print(f"\nDONE {time.time()-t0:.0f}s -> out/ident_20261125.json", flush=True)
