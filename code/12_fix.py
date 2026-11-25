# -*- coding: utf-8 -*-
"""Re-estimate everything both papers report, on ONE consistent sample, with
two bugs fixed. 2026-11-24.

BUG 1  `(col == 1).astype(float)` turns a MISSING policy answer into 0 = "No".
       06b_fast.py and 09_paperB2.py did this, so ~1,750 schools that did not
       answer were counted as schools without the policy. For SC190Q06JA the
       genuine "No" group is 1,417 schools, so the fabricated No's outnumbered
       the real ones. Here every policy indicator keeps NaN as NaN.
BUG 2  `2*(1-norm.cdf(z))` underflows to exactly 0 above z ~ 8.2. Uses erfc.
"""
import pandas as pd, numpy as np, pyreadstat, json, time, math
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
SUBJ={s:[f"PV{i}{s}" for i in range(1,11)] for s in ("MATH","READ","SCIE")}
ALLPV=[p for v in SUBJ.values() for p in v]
def pval(z): return math.erfc(abs(z)/math.sqrt(2))

BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM",
      "IC170Q02JA","IC179Q01JA","IC179Q02JA","IC179Q03JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+ALLPV+RW)
REG=["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
REGLAB=["Written statement on general device use","Cell phone use not allowed on the premises",
        "Formal guidelines for devices in teaching","Teachers set rules for device use in class",
        "Rules set in collaboration with students","Programme on responsible device use"]
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
      usecols=["CNTSCHID","SC013Q01TA","SC002Q01TA","SC002Q02TA"]+REG)
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5)]

def yesno(col):
    """1=Yes -> 1.0, 2=No -> 0.0, anything else -> NaN.  NEVER NaN -> 0."""
    return col.map({1.0:1.0, 2.0:0.0})
for c,lab in zip(REG,REGLAB): d["p_"+c]=yesno(d[c])
d["ban"]=d["p_SC190Q02JA"]
d["private"]=d.SC013Q01TA.map({1.0:0.0, 2.0:1.0})
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d["female"]=d.ST004D01T.map({1.0:1.0, 2.0:0.0})
d["heavy"]=(d.IC170Q02JA>=5).astype(float)
d["reg_n"]=d[["p_"+c for c in REG]].sum(axis=1, min_count=6)   # NaN if ANY item missing
CTRL5=["ESCS","DISCLIM","sch_escs","private","schsize"]
CTRL6=CTRL5+["female"]
print(f"loaded {len(d):,} students | {time.time()-t0:.0f}s",flush=True)

def est(sub, treats, ycols, ctrl, within=True, rubin=True):
    """weighted LS; `within` demeans by country. Returns one dict per treat."""
    cols=list(treats)+list(ycols)+list(ctrl)
    sub=sub.dropna(subset=cols)
    n=len(sub); codes,_=pd.factorize(sub.CNT.values); G=codes.max()+1
    M=np.column_stack([sub[t].values.astype(float) for t in treats]
                      +[sub[c].values.astype(float) for c in ctrl])
    W0=sub.W_FSTUWT.values.astype(float); RWv=sub[RW].values.astype(float)
    Y=sub[list(ycols)].values.astype(float)
    def dm(A,w):
        if not within: return A
        den=np.bincount(codes,weights=w,minlength=G); o=np.empty_like(A)
        for j in range(A.shape[1]):
            o[:,j]=A[:,j]-(np.bincount(codes,weights=w*A[:,j],minlength=G)/den)[codes]
        return o
    def beta(w,y):
        X=dm(M,w)
        if not within: X=np.column_stack([np.ones(len(X)),X])
        yc=dm(y[:,None],w)[:,0]; sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(X*sw[:,None], yc*sw, rcond=None)
        return b[(0 if within else 1):(0 if within else 1)+len(treats)]
    th=[];samp=[]
    for j in range(Y.shape[1]):
        b0=beta(W0,Y[:,j]); th.append(b0)
        samp.append(sum((beta(RWv[:,k],Y[:,j])-b0)**2 for k in range(80))/20.0)
    th=np.array(th); U=np.mean(samp,axis=0)
    B=th.var(axis=0,ddof=1) if (rubin and Y.shape[1]>1) else np.zeros(len(treats))
    se=np.sqrt(U+1.1*B); mu=th.mean(axis=0)
    return [dict(beta=float(mu[i]),se=float(se[i]),p=float(pval(mu[i]/se[i])),n=n,
                 samp_se=float(np.sqrt(U[i])),imp_se=float(np.sqrt(1.1*B[i])))
            for i in range(len(treats))]

out={}
# ---------------- Paper A, Table 1: five rows, ONE sample
print("\n=== A Table 1: the adjustment ladder, one sample, one script ===",flush=True)
dA=d.dropna(subset=["ban"]+CTRL5+SUBJ["MATH"]).copy()
print(f"  sample {len(dA):,} students, {dA.CNTSCHID.nunique():,} schools, {dA.CNT.nunique()} countries",flush=True)
ladder=[("Pooled, no adjustment", [], False),
        ("Pooled, student socio-economic status", ["ESCS"], False),
        ("Within country", [], True),
        ("Within country, student status", ["ESCS"], True),
        ("Within country, student status, school composition, type, size, disciplinary climate", CTRL5, True)]
T1=[]
for lab,ctrl,wi in ladder:
    r=est(dA,["ban"],SUBJ["MATH"],ctrl,within=wi)[0]; r['label']=lab; T1.append(r)
    print(f"  {lab[:62]:<62} {r['beta']:+7.2f}  SE {r['se']:5.2f}  p {r['p']:.3g}",flush=True)
out['A_table1']=T1

# ---------------- Paper A, Table 2: three domains, full specification
print("\n=== A Table 2: three domains ===",flush=True)
T2=[]
for sname,pvs in SUBJ.items():
    sub=d.dropna(subset=["ban"]+CTRL5+pvs)
    r=est(sub,["ban"],pvs,CTRL5,within=True)[0]; r['label']=sname; T2.append(r)
    print(f"  {sname:<8} {r['beta']:+7.2f}  SE {r['se']:5.2f}  p {r['p']:.4g}  n={r['n']:,}",flush=True)
out['A_table2']=T2

# ---------------- Paper A, Table 3: six policies, separately AND mutually adjusted
print("\n=== A Table 3: six policies, each on its own ===",flush=True)
T3=[]
for c,lab in zip(REG,REGLAB):
    sub=d.dropna(subset=["p_"+c]+CTRL5+SUBJ["MATH"])
    r=est(sub,["p_"+c],SUBJ["MATH"],CTRL5,within=True)[0]; r['label']=lab; T3.append(r)
    print(f"  {lab:<46} {r['beta']:+7.2f}  SE {r['se']:5.2f}  p {r['p']:.4g}  n={r['n']:,}",flush=True)
out['A_table3_separate']=T3

print("\n=== A new: six policies, mutually adjusted (all six in one model) ===",flush=True)
subM=d.dropna(subset=["p_"+c for c in REG]+CTRL5+SUBJ["MATH"])
rM=est(subM,["p_"+c for c in REG],SUBJ["MATH"],CTRL5,within=True)
for lab,r in zip(REGLAB,rM):
    r['label']=lab
    print(f"  {lab:<46} {r['beta']:+7.2f}  SE {r['se']:5.2f}  p {r['p']:.4g}",flush=True)
print(f"  (mutually adjusted sample n={rM[0]['n']:,})",flush=True)
out['A_table3_mutual']=rM

# ---------------- Paper B
print("\n=== B Table 1 ===",flush=True)
dB=d.dropna(subset=["ban","IC179Q01JA"]+CTRL6).copy()
dB["support_hi"]=(dB.IC179Q01JA>=3).astype(float)
print(f"  sample {len(dB):,} students, {dB.CNT.nunique()} countries",flush=True)
B1=[]
for lab,treat,y,extra in [
    ("Ban school, endorsement on the four-point scale","ban","IC179Q01JA",[]),
    ("Ban school, proportion agreeing","ban","support_hi",[]),
    ("Ban school, own phone use held constant","ban","IC179Q01JA",["heavy"]),
    ("Own heavy phone use, endorsement","heavy","IC179Q01JA",[])]:
    r=est(dB,[treat],[y],CTRL6+extra,within=True,rubin=False)[0]; r['label']=lab; B1.append(r)
    print(f"  {lab:<48} {r['beta']:+.4f}  SE {r['se']:.4f}  p {r['p']:.4g}",flush=True)
out['B_table1']=B1

print("\n=== B dose response, regulation count 0-6 (all six items answered) ===",flush=True)
dD=d.dropna(subset=["reg_n","IC179Q01JA"]+CTRL6).copy()
print(f"  sample {len(dD):,} students, {dD.CNT.nunique()} countries",flush=True)
BD=[]
for lab,extra in [("Number of device rules, per rule",[]),
                  ("Number of device rules, own use held constant",["heavy"])]:
    r=est(dD,["reg_n"],["IC179Q01JA"],CTRL6+extra,within=True,rubin=False)[0]; r['label']=lab; BD.append(r)
    print(f"  {lab:<48} {r['beta']:+.4f}  SE {r['se']:.4f}  p {r['p']:.4g}",flush=True)
out['B_dose']=BD

print("\n=== B other attitude items ===",flush=True)
BI=[]
for lab,y in [("Phones should not be brought to class","IC179Q01JA"),
              ("Laptops should not be brought to class","IC179Q02JA"),
              ("Students should help set the device rules","IC179Q03JA")]:
    sub=d.dropna(subset=["ban",y]+CTRL6)
    r=est(sub,["ban"],[y],CTRL6,within=True,rubin=False)[0]; r['label']=lab; BI.append(r)
    print(f"  {lab:<46} {r['beta']:+.4f}  SE {r['se']:.4f}  p {r['p']:.4g}  n={r['n']:,}",flush=True)
out['B_items']=BI

print("\n=== B raw weighted distribution of the four-point item ===",flush=True)
dist={}
for nm,g in (("ban",dB[dB.ban==1]),("no_ban",dB[dB.ban==0])):
    w=g.W_FSTUWT.values; tot=w.sum()
    dist[nm]={str(int(v)): float(w[g.IC179Q01JA.values==v].sum()/tot*100) for v in (1,2,3,4)}
    dist[nm].update(n=int(len(g)), mean=float(np.average(g.IC179Q01JA,weights=w)))
    print(f"  {nm:7s} n={len(g):>7,} "+" ".join(f"{k}:{dist[nm][k]:5.2f}%" for k in '1234')+f"  mean {dist[nm]['mean']:.4f}",flush=True)
out['B_distribution']=dist

print("\n=== B observed means by number of rules ===",flush=True)
BM=[]
for k in range(7):
    g=dD[dD.reg_n==k]
    if len(g)<200: BM.append(dict(reg_n=k,n=int(len(g)),mean=None,se=None)); continue
    w=g.W_FSTUWT.values; m=float(np.average(g.IC179Q01JA,weights=w)); RWv=g[RW].values
    se=math.sqrt(sum((np.average(g.IC179Q01JA,weights=RWv[:,j])-m)**2 for j in range(80))/20.0)
    BM.append(dict(reg_n=k,n=int(len(g)),mean=m,se=se))
    print(f"  rules={k} n={len(g):>7,} mean {m:.4f} SE {se:.4f}",flush=True)
out['B_dose_observed']=BM

print("\n=== B benchmark on the same outcome ===",flush=True)
BB=[]
for lab,var,mk in [("Attending a school that bans phones","ban",None),
                   ("School mean socio-economic status, per SD","sch_escs_sd","sch_escs"),
                   ("Disciplinary climate, per SD","DISCLIM_sd","DISCLIM"),
                   ("Own heavy phone use","heavy",None),
                   ("Female","female",None),
                   ("Private school","private",None)]:
    dd=dB.copy()
    if mk: dd[var]=(dd[mk]-dd[mk].mean())/dd[mk].std()
    ctrl=[c for c in CTRL6 if c!=var and c!=mk]
    r=est(dd,[var],["IC179Q01JA"],ctrl,within=True,rubin=False)[0]; r['label']=lab; BB.append(r)
    print(f"  {lab:<44} {r['beta']:+.4f}  SE {r['se']:.4f}",flush=True)
out['B_benchmark']=BB

print("\n=== B country estimates with BRR SE ===",flush=True)
rows=[]
for cnt,g in dB.groupby("CNT"):
    if g.ban.nunique()<2 or len(g)<800: continue
    if (g.ban==1).sum()<150 or (g.ban==0).sum()<150: continue
    X=np.column_stack([np.ones(len(g)), g.ban.values]+[g[c].values.astype(float) for c in CTRL6])
    y=g.IC179Q01JA.values.astype(float); W0=g.W_FSTUWT.values; RWv=g[RW].values
    def b_(w):
        sw=np.sqrt(w); b,*_=np.linalg.lstsq(X*sw[:,None], y*sw, rcond=None); return b[1]
    b0=b_(W0); se=math.sqrt(sum((b_(RWv[:,k])-b0)**2 for k in range(80))/20.0)
    rows.append(dict(CNT=cnt,n=int(len(g)),beta=float(b0),se=float(se)))
rc=pd.DataFrame(rows); rc.to_csv("out/paperB_country_brr.csv",index=False)
print(f"  {len(rc)} countries | positive {int((rc.beta>0).sum())} | median {rc.beta.median():+.4f}",flush=True)
print(f"  significant positive {int(((rc.beta-1.96*rc.se)>0).sum())} | significant negative {int(((rc.beta+1.96*rc.se)<0).sum())}",flush=True)
out['B_country_n']=len(rc)

print("\n=== B variance components ===",flush=True)
g=dB; sm=g.groupby("CNTSCHID").IC179Q01JA.transform("mean")
out['B_icc']=dict(total_var=float(g.IC179Q01JA.var()), between_school_var=float(sm.var()),
                  icc=float(sm.var()/g.IC179Q01JA.var()), n=int(len(g)), n_schools=int(g.CNTSCHID.nunique()))
print(f"  ICC {out['B_icc']['icc']:.3f} over {out['B_icc']['n_schools']:,} schools",flush=True)

json.dump(out, open("out/fixed_20261124.json","w"), ensure_ascii=False, indent=1)
print(f"\nDONE {time.time()-t0:.0f}s -> out/fixed_20261124.json",flush=True)
