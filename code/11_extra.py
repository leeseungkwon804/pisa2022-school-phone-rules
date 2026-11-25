# -*- coding: utf-8 -*-
"""Extra analyses demanded by a hard review. 2026-11-24.
A1  six device policies, MUTUALLY adjusted (all six in one model)
B1  raw weighted distribution of the 4-point endorsement item by school policy
B2  observed weighted means of endorsement by number of device rules (0-6)
B3  country-specific Paper B estimates WITH BRR standard errors
B4  effect-size benchmark: ban vs other school-level predictors, same outcome
B5  between-school share of variance in the endorsement item
p-values use erfc, never 1-norm.cdf (that underflows to exactly 0 above z~8.2).
"""
import pandas as pd, numpy as np, pyreadstat, json, time, math
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
def pval(z): return math.erfc(abs(z)/math.sqrt(2))

BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM",
      "IC170Q02JA","IC179Q01JA","IC179Q02JA","IC179Q03JA"]
PV=[f"PV{i}MATH" for i in range(1,11)]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+PV+RW)
SC=["CNTSCHID","SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA",
    "SC013Q01TA","SC002Q01TA","SC002Q02TA"]
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV", usecols=SC)
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5) & d.SC190Q02JA.notna()]
d["ban"]=(d.SC190Q02JA==1).astype(float)
d["private"]=(d.SC013Q01TA==2).astype(float)
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d["female"]=(d.ST004D01T==1).astype(float)
d["heavy"]=(d.IC170Q02JA>=5).astype(float)
REG=["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
REGLAB=["Written statement on device use","Cell phone use not allowed on the premises",
        "Formal guidelines for devices in teaching","Teachers set rules for device use in class",
        "Rules set in collaboration with students","Programme on responsible device use"]
for c in REG: d["p_"+c]=(d[c]==1).astype(float)
d["reg_n"]=sum(d["p_"+c] for c in REG)
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize","female"]
print(f"merged {len(d):,} students, {d.CNT.nunique()} countries | {time.time()-t0:.0f}s",flush=True)

codes_all,_=pd.factorize(d.CNT.values)
def wls_within(sub, treats, y, extra=()):
    """within-country weighted LS; returns coefficient(s) on `treats` with BRR SE"""
    cols=list(treats)+[y]+CTRL+list(extra)
    sub=sub.dropna(subset=cols)
    n=len(sub); codes,_=pd.factorize(sub.CNT.values); G=codes.max()+1
    M=np.column_stack([sub[t].values.astype(float) for t in treats]
        +[sub[c].values.astype(float) for c in CTRL+list(extra)])
    yv=sub[y].values.astype(float); W0=sub.W_FSTUWT.values.astype(float); RWv=sub[RW].values.astype(float)
    def dm(A,w):
        den=np.bincount(codes,weights=w,minlength=G); out=np.empty_like(A)
        for j in range(A.shape[1]):
            out[:,j]=A[:,j]-(np.bincount(codes,weights=w*A[:,j],minlength=G)/den)[codes]
        return out
    def beta(w):
        Xc=dm(M,w); yc=dm(yv[:,None],w)[:,0]; sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(Xc*sw[:,None], yc*sw, rcond=None); return b[:len(treats)]
    b0=beta(W0)
    U=sum((beta(RWv[:,k])-b0)**2 for k in range(80))/20.0
    se=np.sqrt(U)
    return [dict(beta=float(b0[i]), se=float(se[i]), p=float(pval(b0[i]/se[i])), n=n) for i in range(len(treats))]

out={}

# ---- A1 six policies, mutually adjusted, outcome = mathematics (PV-averaged for speed, BRR for SE)
print("\n=== A1 six device policies, mutually adjusted ===",flush=True)
dA=d.dropna(subset=["ESCS","DISCLIM","sch_escs","private","schsize"]+PV).copy()
dA["pvbar"]=dA[PV].mean(axis=1)
treats=["p_"+c for c in REG]
rA=wls_within(dA, treats, "pvbar")
# imputation variance from the 10 PVs, Rubin
codes,_=pd.factorize(dA.CNT.values); G=codes.max()+1
M=np.column_stack([dA[t].values.astype(float) for t in treats]+[dA[c].values.astype(float) for c in CTRL[:-1]])
W0=dA.W_FSTUWT.values.astype(float)
def dm(A,w,codes=codes,G=G):
    den=np.bincount(codes,weights=w,minlength=G); out=np.empty_like(A)
    for j in range(A.shape[1]):
        out[:,j]=A[:,j]-(np.bincount(codes,weights=w*A[:,j],minlength=G)/den)[codes]
    return out
Xc=dm(M,W0); sw=np.sqrt(W0)
th=[]
for p in PV:
    yc=dm(dA[p].values.astype(float)[:,None],W0)[:,0]
    b,*_=np.linalg.lstsq(Xc*sw[:,None], yc*sw, rcond=None); th.append(b[:6])
th=np.array(th); B=th.var(axis=0,ddof=1)
A1=[]
for i,(lab,r) in enumerate(zip(REGLAB,rA)):
    se=math.sqrt(r['se']**2 + 1.1*B[i])
    A1.append(dict(label=lab, beta=float(th[:,i].mean()), se=se, p=float(pval(th[:,i].mean()/se)), n=r['n']))
    print(f"  {lab:<48} {A1[-1]['beta']:+7.2f}  SE {se:5.2f}  p {A1[-1]['p']:.3g}",flush=True)
out['A1_policies_mutual']=A1

# ---- B1 raw weighted distribution of IC179Q01JA by ban
print("\n=== B1 endorsement distribution by school policy ===",flush=True)
dB=d.dropna(subset=["IC179Q01JA","ban"]).copy()
dist={}
for nm,g in (("ban",dB[dB.ban==1]),("no_ban",dB[dB.ban==0])):
    w=g.W_FSTUWT.values; tot=w.sum()
    dist[nm]={str(int(v)): float(w[(g.IC179Q01JA.values==v)].sum()/tot*100) for v in (1,2,3,4)}
    dist[nm]['n']=int(len(g)); dist[nm]['mean']=float(np.average(g.IC179Q01JA,weights=w))
    print(f"  {nm:7s} n={len(g):>7,}  "+"  ".join(f"{k}:{dist[nm][k]:5.2f}%" for k in '1234')+f"   mean {dist[nm]['mean']:.3f}",flush=True)
out['B1_distribution']=dist

# ---- B2 observed weighted means by regulation count
print("\n=== B2 observed means by number of device rules ===",flush=True)
B2=[]
for k in range(7):
    g=dB[dB.reg_n==k]
    if len(g)<200: 
        B2.append(dict(reg_n=k,n=int(len(g)),mean=None,se=None)); continue
    w=g.W_FSTUWT.values; m=float(np.average(g.IC179Q01JA,weights=w))
    RWv=g[RW].values
    se=math.sqrt(sum((np.average(g.IC179Q01JA,weights=RWv[:,j])-m)**2 for j in range(80))/20.0)
    B2.append(dict(reg_n=k,n=int(len(g)),mean=m,se=se))
    print(f"  rules={k}  n={len(g):>7,}  mean {m:.4f}  SE {se:.4f}",flush=True)
out['B2_dose_observed']=B2

# ---- B3 country estimates with BRR SE
print("\n=== B3 country-specific estimates with BRR SE ===",flush=True)
rows=[]
for cnt,g in dB.groupby("CNT"):
    g2=g.dropna(subset=["ban","IC179Q01JA"]+CTRL)
    if g2.ban.nunique()<2 or len(g2)<800: continue
    if (g2.ban==1).sum()<150 or (g2.ban==0).sum()<150: continue
    X=np.column_stack([np.ones(len(g2)), g2.ban.values]+[g2[c].values.astype(float) for c in CTRL])
    y=g2.IC179Q01JA.values.astype(float); W0=g2.W_FSTUWT.values; RWv=g2[RW].values
    def b_(w):
        sw=np.sqrt(w); b,*_=np.linalg.lstsq(X*sw[:,None], y*sw, rcond=None); return b[1]
    b0=b_(W0); se=math.sqrt(sum((b_(RWv[:,k])-b0)**2 for k in range(80))/20.0)
    rows.append(dict(CNT=cnt,n=int(len(g2)),beta=float(b0),se=float(se)))
    print(f"  {cnt} n={len(g2):>6,} beta {b0:+.4f} SE {se:.4f}",flush=True)
pd.DataFrame(rows).to_csv("out/paperB_country_brr.csv",index=False)
out['B3_n_countries']=len(rows)

# ---- B4 effect-size benchmark on the same outcome
print("\n=== B4 benchmark: ban against other school-level predictors ===",flush=True)
bench=[]
for lab,var in [("Attending a school that bans phones","ban"),
                ("School mean socio-economic status (per SD)","sch_escs_sd"),
                ("Disciplinary climate (per SD)","DISCLIM_sd"),
                ("Private school","private"),
                ("Own heavy phone use","heavy"),
                ("Female","female")]:
    dd=dB.copy()
    if var.endswith("_sd"):
        base=var[:-3]; dd[var]=(dd[base]-dd[base].mean())/dd[base].std()
    ctrl=[c for c in CTRL if c!=var and not var.startswith(c)]
    cols=[var,"IC179Q01JA"]+ctrl
    sub=dd.dropna(subset=cols)
    codes2,_=pd.factorize(sub.CNT.values); G2=codes2.max()+1
    M2=np.column_stack([sub[var].values.astype(float)]+[sub[c].values.astype(float) for c in ctrl])
    yv=sub.IC179Q01JA.values.astype(float); W=sub.W_FSTUWT.values; RWv=sub[RW].values
    def dm2(A,w):
        den=np.bincount(codes2,weights=w,minlength=G2); o=np.empty_like(A)
        for j in range(A.shape[1]):
            o[:,j]=A[:,j]-(np.bincount(codes2,weights=w*A[:,j],minlength=G2)/den)[codes2]
        return o
    def bb(w):
        Xc=dm2(M2,w); yc=dm2(yv[:,None],w)[:,0]; sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(Xc*sw[:,None], yc*sw, rcond=None); return b[0]
    b0=bb(W); se=math.sqrt(sum((bb(RWv[:,k])-b0)**2 for k in range(80))/20.0)
    bench.append(dict(label=lab,beta=float(b0),se=se,p=float(pval(b0/se)),n=int(len(sub))))
    print(f"  {lab:<44} {b0:+.4f}  SE {se:.4f}",flush=True)
out['B4_benchmark']=bench

# ---- B5 between-school share of variance
print("\n=== B5 variance components ===",flush=True)
g=dB.dropna(subset=["IC179Q01JA"])
sm=g.groupby("CNTSCHID").IC179Q01JA.transform("mean")
tot=g.IC179Q01JA.var(); betw=sm.var()
out['B5_icc']=dict(total_var=float(tot), between_school_var=float(betw), icc=float(betw/tot),
                   n=int(len(g)), n_schools=int(g.CNTSCHID.nunique()))
print(f"  total {tot:.4f} | between-school {betw:.4f} | ICC {betw/tot:.3f} | schools {g.CNTSCHID.nunique():,}",flush=True)

json.dump(out, open("out/extra_20261124.json","w"), ensure_ascii=False, indent=1)
print(f"\nDONE {time.time()-t0:.0f}s -> out/extra_20261124.json",flush=True)
