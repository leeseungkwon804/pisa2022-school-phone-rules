# -*- coding: utf-8 -*-
"""31_acq.py  2026-11-25
The acquiescence test Paper B needs.

30_identify.py showed that 76% of the ban-endorsement association on the phone
item is a shift in the student's MEAN across the whole six-item device battery
(+0.0583 of +0.0767).  Two readings compete:
  (a) response style - students in ban schools simply agree with more statements;
  (b) substance      - they really are more pro-regulation about devices.
These are separable, because the PISA student questionnaire carries four more
agree/disagree batteries in the SAME four-point format that have nothing to do
with devices, and each MIXES forward- and reverse-keyed items.

Pure acquiescence raises agreement on forward AND reverse items alike, so it
shows up as a positive coefficient on both, and on the unreversed mean.
A substantive difference moves them in OPPOSITE directions and leaves the
unreversed mean near zero.
"""
import pandas as pd, numpy as np, pyreadstat, json, time, math
t0 = time.time()
RW    = [f"W_FSTURWT{i}" for i in range(1, 81)]
REG   = ["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
ITEMS = ["IC179Q01JA","IC179Q02JA","IC179Q03JA","IC179Q04JA","IC179Q05JA","IC179Q06JA"]
CTRL6 = ["ESCS","DISCLIM","sch_escs","private","schsize","female"]
def pval(z): return math.erfc(abs(z)/math.sqrt(2))

# battery: (name, forward-keyed, reverse-keyed)
BAT = [
 ("Perseverance (ST307)",
  ["ST307Q01JA","ST307Q02JA","ST307Q03JA","ST307Q05JA","ST307Q08JA","ST307Q09JA"],
  ["ST307Q04JA","ST307Q06JA","ST307Q07JA","ST307Q10JA"]),
 ("Self-control (ST309)",
  ["ST309Q01JA","ST309Q04JA","ST309Q05JA","ST309Q06JA","ST309Q08JA","ST309Q10JA"],
  ["ST309Q02JA","ST309Q03JA","ST309Q07JA","ST309Q09JA"]),
 ("Curiosity (ST301)",
  ["ST301Q01JA","ST301Q02JA","ST301Q04JA","ST301Q05JA","ST301Q06JA","ST301Q07JA"],
  ["ST301Q03JA"]),
 ("Sense of belonging (ST034)",
  ["ST034Q02TA","ST034Q03TA","ST034Q05TA"],
  ["ST034Q01TA","ST034Q04TA","ST034Q06TA"]),
]
STALL = sorted({c for _, f, r in BAT for c in f+r})

BASE = ["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM","IC170Q02JA"]
d, _ = pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+ITEMS+STALL+RW)
s, _ = pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
        usecols=["CNTSCHID","SC013Q01TA","SC002Q01TA","SC002Q02TA"]+REG)
d = d.merge(s, on="CNTSCHID", how="inner")
d = d[d.IC170Q02JA.notna() & (d.IC170Q02JA <= 5)]
d["ban"]      = d["SC190Q02JA"].map({1.0:1.0, 2.0:0.0})
d["private"]  = d.SC013Q01TA.map({1.0:0.0, 2.0:1.0})
d["schsize"]  = np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1, min_count=1))
d["sch_escs"] = d.groupby("CNTSCHID").ESCS.transform("mean")
d["female"]   = d.ST004D01T.map({1.0:1.0, 2.0:0.0})
for c in ITEMS+STALL: d[c] = d[c].where(d[c] <= 4)
print(f"loaded {len(d):,} | {time.time()-t0:.0f}s", flush=True)
print("\n=== code ranges of the placebo batteries ===", flush=True)
for c in STALL[:3]+STALL[-2:]:
    print(f"  {c}: min {d[c].min()} max {d[c].max()} n {d[c].notna().sum():,}", flush=True)

# PISA 2022 rotates questionnaire forms ITEM BY ITEM inside a battery: each
# student answers only some items of a construct, so requiring a whole battery
# leaves nothing (ST034 complete-case within the ICT subset = 0 students, while
# each single item has about 257,000).  Every placebo item is therefore estimated
# on its OWN complete-case sample, and compared at the item level.
def one_item(col):
    sub = d.dropna(subset=["ban", col]+CTRL6)
    if len(sub) < 5000 or sub.ban.nunique() < 2: return None
    codes, _ = pd.factorize(sub.CNT.values); G = codes.max()+1
    M = sub[["ban"]+CTRL6].values.astype(float)
    y = sub[col].values.astype(float)
    den_cache = {}
    def f(w):
        den = np.bincount(codes, weights=w, minlength=G)
        A = np.column_stack([M, y])
        o = np.empty_like(A)
        for j in range(A.shape[1]):
            o[:, j] = A[:, j] - (np.bincount(codes, weights=w*A[:, j], minlength=G)/den)[codes]
        sw = np.sqrt(w)
        b, *_ = np.linalg.lstsq(o[:, :-1]*sw[:, None], o[:, -1]*sw, rcond=None)
        return float(b[0])
    w0 = sub.W_FSTUWT.values.astype(float); RWv = sub[RW].values.astype(float)
    b = f(w0)
    se = math.sqrt(sum((f(RWv[:, k])-b)**2 for k in range(80))/20.0)
    sd = math.sqrt(np.average((y-np.average(y, weights=w0))**2, weights=w0))
    return dict(beta=b, se=se, p=pval(b/se), n=int(len(sub)),
                n_countries=int(sub.CNT.nunique()), sd=sd)

rows = []
print(f"\n{'item':<16}{'battery':<28}{'key':<9}{'beta':>9}{'SE':>8}{'p':>11}{'n':>10}", flush=True)
print("-"*92, flush=True)
# the device battery first, item by item, on the same footing as the placebos
for it in ITEMS:
    r = one_item(it)
    if r is None: continue
    r.update(item=it, battery="Device rules (IC179)", key="n/a")
    rows.append(r)
    print(f"{it:<16}{'Device rules (IC179)':<28}{'n/a':<9}{r['beta']:+9.4f}"
          f"{r['se']:8.4f}{r['p']:11.3g}{r['n']:10,}", flush=True)
print(flush=True)
for nm, fw, rv in BAT:
    for key, cols in (("forward", fw), ("reverse", rv)):
        for c in cols:
            r = one_item(c)
            if r is None:
                print(f"{c:<16}{nm[:27]:<28}{key:<9}  skipped (too few)", flush=True); continue
            r.update(item=c, battery=nm, key=key)
            rows.append(r)
            print(f"{c:<16}{nm[:27]:<28}{key:<9}{r['beta']:+9.4f}"
                  f"{r['se']:8.4f}{r['p']:11.3g}{r['n']:10,}", flush=True)
    print(flush=True)

print("="*92, flush=True)
print("SUMMARY  mean ban coefficient by battery and keying", flush=True)
print("="*92, flush=True)
summ = []
for nm in ["Device rules (IC179)"] + [b[0] for b in BAT]:
    for key in ["n/a", "forward", "reverse"]:
        g = [r for r in rows if r["battery"] == nm and r["key"] == key]
        if not g: continue
        mb = float(np.mean([r["beta"] for r in g]))
        npos = sum(1 for r in g if r["beta"] > 0)
        nsig = sum(1 for r in g if r["p"] < 0.05)
        summ.append(dict(battery=nm, key=key, k_items=len(g), mean_beta=mb,
                         n_positive=npos, n_sig=nsig))
        print(f"  {nm[:28]:<30}{key:<9}k={len(g):>2}  mean beta {mb:+.4f}  "
              f"positive {npos}/{len(g)}  p<0.05 {nsig}/{len(g)}", flush=True)

print("\n--- how to read this ---", flush=True)
print("Pure acquiescence: forward AND reverse means both clearly positive.", flush=True)
print("Substantive difference: forward and reverse means have OPPOSITE signs.", flush=True)
print("Device-specific endorsement: device items positive, placebo items near zero.", flush=True)
json.dump(dict(items=rows, summary=summ), open("out/acq_20261125.json", "w"), indent=1)
print(f"\nDONE {time.time()-t0:.0f}s -> out/acq_20261125.json", flush=True)
