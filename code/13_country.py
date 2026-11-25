# -*- coding: utf-8 -*-
"""Paper A country-level re-run with the missing-value bug fixed, plus the
compliance quantities Section 4.1 reports and the within-country dose means
Paper B needs.  2026-11-24."""
import pandas as pd, numpy as np, pyreadstat, json, time, math
t0=time.time()
PV=[f"PV{i}MATH" for i in range(1,11)]
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
def pval(z): return math.erfc(abs(z)/math.sqrt(2))
REG=["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM","IC170Q02JA","IC179Q01JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+PV+RW)
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
      usecols=["CNTSCHID","SC013Q01TA","SC002Q01TA","SC002Q02TA"]+REG)
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5)]
yn=lambda c: c.map({1.0:1.0, 2.0:0.0})
for c in REG: d["p_"+c]=yn(d[c])
d["ban"]=d["p_SC190Q02JA"]
d["private"]=d.SC013Q01TA.map({1.0:0.0, 2.0:1.0})
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d["female"]=d.ST004D01T.map({1.0:1.0,2.0:0.0})
d["heavy"]=(d.IC170Q02JA>=5).astype(float)
d["reg_n"]=d[["p_"+c for c in REG]].sum(axis=1,min_count=6)
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize"]
out={}

# ---- compliance, pooled and within country (Section 4.1)
dc=d.dropna(subset=["ban","heavy"]).copy()
w=dc.W_FSTUWT.values
pa=np.average(dc.heavy[dc.ban==1],weights=dc.W_FSTUWT[dc.ban==1])
pc=np.average(dc.heavy[dc.ban==0],weights=dc.W_FSTUWT[dc.ban==0])
never_a=np.average((dc.IC170Q02JA[dc.ban==1]==1),weights=dc.W_FSTUWT[dc.ban==1])
never_c=np.average((dc.IC170Q02JA[dc.ban==0]==1),weights=dc.W_FSTUWT[dc.ban==0])
print(f"pooled heavy use  ban {pa*100:.2f}%  no-ban {pc*100:.2f}%  gap {(pa-pc)*100:+.2f} pp",flush=True)
print(f"pooled never use  ban {never_a*100:.2f}%  no-ban {never_c*100:.2f}%",flush=True)
rows=[]
for cnt,g in dc.groupby("CNT"):
    if g.ban.nunique()<2: continue
    ga,gc=g[g.ban==1],g[g.ban==0]
    if len(ga)<80 or len(gc)<80: continue
    gap=(np.average(ga.heavy,weights=ga.W_FSTUWT)-np.average(gc.heavy,weights=gc.W_FSTUWT))*100
    rows.append(dict(CNT=cnt,n=int(len(g)),gap=float(gap)))
cg=pd.DataFrame(rows); cg.to_csv("out/compliance_fixed.csv",index=False)
out['compliance']=dict(pooled_ban=float(pa*100), pooled_noban=float(pc*100), pooled_gap=float((pa-pc)*100),
   never_ban=float(never_a*100), never_noban=float(never_c*100),
   n_countries=len(cg), median=float(cg.gap.median()),
   q25=float(cg.gap.quantile(.25)), q75=float(cg.gap.quantile(.75)),
   lo=float(cg.gap.min()), hi=float(cg.gap.max()), n_reducing=int((cg.gap<0).sum()))
print(f"within country: {len(cg)} countries, median {cg.gap.median():+.2f}, "
      f"IQR {cg.gap.quantile(.25):+.2f}..{cg.gap.quantile(.75):+.2f}, "
      f"range {cg.gap.min():+.2f}..{cg.gap.max():+.2f}, reducing {int((cg.gap<0).sum())}",flush=True)

# ---- country-specific achievement difference, full spec, PV10 + BRR80
print("\n=== country-specific achievement difference ===",flush=True)
dA=d.dropna(subset=["ban"]+CTRL+PV).copy()
def wb(X,y,w,i=1):
    sw=np.sqrt(w); b,*_=np.linalg.lstsq(X*sw[:,None], y*sw, rcond=None); return b[i]
rows=[]
for cnt,g in dA.groupby("CNT"):
    if g.ban.nunique()<2: continue
    na,nc=int((g.ban==1).sum()),int((g.ban==0).sum())
    if na<80 or nc<80: continue
    n=len(g)
    X=np.column_stack([np.ones(n), g.ban.values]+[g[c].values.astype(float) for c in CTRL])
    if np.linalg.matrix_rank(X)<X.shape[1]: continue
    w0=g.W_FSTUWT.values; RWv=g[RW].values
    th=[];samp=[]
    for p in PV:
        y=g[p].values; b0=wb(X,y,w0); th.append(b0)
        samp.append(sum((wb(X,y,RWv[:,k])-b0)**2 for k in range(80))/20.0)
    th=np.array(th); se=math.sqrt(np.mean(samp)+1.1*th.var(ddof=1))
    ga,gc=g[g.ban==1],g[g.ban==0]
    gap=(np.average(ga.heavy,weights=ga.W_FSTUWT)-np.average(gc.heavy,weights=gc.W_FSTUWT))*100
    # robustness variants for the correlation checks
    def var_(cols):
        Xv=np.column_stack([np.ones(n), g.ban.values]+[g[c].values.astype(float) for c in cols])
        return float(np.mean([wb(Xv,g[p].values,w0) for p in PV]))
    rows.append(dict(CNT=cnt,n=n,n_ban=na,beta=float(th.mean()),se=se,gap=float(gap),
                     b_escs=var_(["ESCS"]), b_disc=var_(["ESCS","DISCLIM"]), b_full=float(th.mean())))
    print(f"  {cnt} n={n:>6,} beta {th.mean():+7.2f} SE {se:5.2f} gap {gap:+6.1f}",flush=True)
r=pd.DataFrame(rows); r.to_csv("out/country_fixed.csv",index=False)
print(f"\n{len(r)} countries",flush=True)
out['n_countries_achievement']=len(r)

# ---- Paper B: within-country adjusted mean endorsement by rule count
print("\n=== within-country adjusted endorsement by number of rules ===",flush=True)
dD=d.dropna(subset=["reg_n","IC179Q01JA","ban","female"]+CTRL).copy()
codes,_=pd.factorize(dD.CNT.values); G=codes.max()+1
W0=dD.W_FSTUWT.values.astype(float)
den=np.bincount(codes,weights=W0,minlength=G)
cmean=(np.bincount(codes,weights=W0*dD.IC179Q01JA.values,minlength=G)/den)[codes]
grand=np.average(dD.IC179Q01JA,weights=W0)
dD["adj"]=dD.IC179Q01JA.values-cmean+grand          # country-centred, re-centred on the grand mean
RWv=dD[RW].values
BM=[]
for k in range(7):
    m_=dD.reg_n==k
    if m_.sum()<200: BM.append(dict(reg_n=k,n=int(m_.sum()),mean=None,se=None)); continue
    g=dD[m_]; mu=float(np.average(g.adj,weights=g.W_FSTUWT))
    def rep(j):
        wj=RWv[:,j]; dn=np.bincount(codes,weights=wj,minlength=G)
        cm=(np.bincount(codes,weights=wj*dD.IC179Q01JA.values,minlength=G)/dn)[codes]
        a=dD.IC179Q01JA.values-cm+np.average(dD.IC179Q01JA,weights=wj)
        return float(np.average(a[m_.values],weights=wj[m_.values]))
    se=math.sqrt(sum((rep(j)-mu)**2 for j in range(80))/20.0)
    BM.append(dict(reg_n=k,n=int(m_.sum()),mean=mu,se=se))
    print(f"  rules={k} n={int(m_.sum()):>7,} adj.mean {mu:.4f} SE {se:.4f}",flush=True)
out['B_dose_within']=BM
json.dump(out,open("out/country_fixed.json","w"),ensure_ascii=False,indent=1)
print(f"\nDONE {time.time()-t0:.0f}s",flush=True)
