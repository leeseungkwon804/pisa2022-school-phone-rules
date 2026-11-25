# -*- coding: utf-8 -*-
"""Dose-response done properly: regulation count entered as INDICATORS, not as a
line, so the figure and the slope are the same object.  Reference = 0 rules.
Also the same test for Paper A's achievement outcome.  2026-11-24."""
import pandas as pd, numpy as np, pyreadstat, json, time, math
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
PV=[f"PV{i}MATH" for i in range(1,11)]
def pval(z): return math.erfc(abs(z)/math.sqrt(2))
REG=["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM","IC170Q02JA","IC179Q01JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+PV+RW)
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
      usecols=["CNTSCHID","SC013Q01TA","SC002Q01TA","SC002Q02TA"]+REG)
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5)]
yn=lambda c: c.map({1.0:1.0,2.0:0.0})
for c in REG: d["p_"+c]=yn(d[c])
d["ban"]=d["p_SC190Q02JA"]
d["private"]=d.SC013Q01TA.map({1.0:0.0,2.0:1.0})
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d["female"]=d.ST004D01T.map({1.0:1.0,2.0:0.0})
d["heavy"]=(d.IC170Q02JA>=5).astype(float)
d["reg_n"]=d[["p_"+c for c in REG]].sum(axis=1,min_count=6)
CTRL6=["ESCS","DISCLIM","sch_escs","private","schsize","female"]

def est(sub,treats,ycols,ctrl,rubin=False):
    sub=sub.dropna(subset=list(treats)+list(ycols)+list(ctrl))
    n=len(sub); codes,_=pd.factorize(sub.CNT.values); G=codes.max()+1
    M=np.column_stack([sub[t].values.astype(float) for t in treats]
                      +[sub[c].values.astype(float) for c in ctrl])
    W0=sub.W_FSTUWT.values.astype(float); RWv=sub[RW].values.astype(float)
    Y=sub[list(ycols)].values.astype(float)
    def dm(A,w):
        den=np.bincount(codes,weights=w,minlength=G); o=np.empty_like(A)
        for j in range(A.shape[1]):
            o[:,j]=A[:,j]-(np.bincount(codes,weights=w*A[:,j],minlength=G)/den)[codes]
        return o
    def beta(w,y):
        X=dm(M,w); yc=dm(y[:,None],w)[:,0]; sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(X*sw[:,None], yc*sw, rcond=None); return b[:len(treats)]
    th=[];samp=[]
    for j in range(Y.shape[1]):
        b0=beta(W0,Y[:,j]); th.append(b0)
        samp.append(sum((beta(RWv[:,k],Y[:,j])-b0)**2 for k in range(80))/20.0)
    th=np.array(th); U=np.mean(samp,axis=0)
    B=th.var(axis=0,ddof=1) if (rubin and Y.shape[1]>1) else np.zeros(len(treats))
    se=np.sqrt(U+1.1*B); mu=th.mean(axis=0)
    return n,[dict(beta=float(mu[i]),se=float(se[i]),p=float(pval(mu[i]/se[i]))) for i in range(len(treats))]

out={}
dD=d.dropna(subset=["reg_n","IC179Q01JA"]+CTRL6).copy()
for k in range(1,7): dD[f"r{k}"]=(dD.reg_n==k).astype(float)
print("=== B dose as indicators, reference = 0 rules (endorsement) ===",flush=True)
n,rr=est(dD,[f"r{k}" for k in range(1,7)],["IC179Q01JA"],CTRL6)
B=[dict(reg_n=0,beta=0.0,se=0.0,p=None,n=int((dD.reg_n==0).sum()))]
for k,r in zip(range(1,7),rr):
    r.update(reg_n=k,n=int((dD.reg_n==k).sum())); B.append(r)
    print(f"  {k} rules vs 0   {r['beta']:+.4f}  SE {r['se']:.4f}  p {r['p']:.3g}  n={r['n']:,}",flush=True)
out['B_dose_indicators']=B; out['B_dose_n']=n

print("\n=== B dose, linear slope on the same sample ===",flush=True)
n2,r2=est(dD,["reg_n"],["IC179Q01JA"],CTRL6)
print(f"  per rule        {r2[0]['beta']:+.4f}  SE {r2[0]['se']:.4f}  p {r2[0]['p']:.3g}  n={n2:,}",flush=True)
out['B_dose_linear']=r2[0]
# monotone test: Spearman of the adjusted steps against rule count
from math import sqrt
steps=[b['beta'] for b in B]
rk=np.argsort(np.argsort(steps))
rho=np.corrcoef(np.arange(7),rk)[0,1]
out['B_dose_spearman']=float(rho)
print(f"  Spearman of adjusted step against rule count: {rho:+.3f}",flush=True)

print("\n=== A dose as indicators (mathematics) ===",flush=True)
dA=d.dropna(subset=["reg_n"]+CTRL6[:-1]+PV).copy()
for k in range(1,7): dA[f"r{k}"]=(dA.reg_n==k).astype(float)
nA,rA=est(dA,[f"r{k}" for k in range(1,7)],PV,CTRL6[:-1],rubin=True)
A=[dict(reg_n=0,beta=0.0,se=0.0,p=None,n=int((dA.reg_n==0).sum()))]
for k,r in zip(range(1,7),rA):
    r.update(reg_n=k,n=int((dA.reg_n==k).sum())); A.append(r)
    print(f"  {k} rules vs 0   {r['beta']:+7.2f}  SE {r['se']:5.2f}  p {r['p']:.3g}  n={r['n']:,}",flush=True)
out['A_dose_indicators']=A; out['A_dose_n']=nA
json.dump(out,open("out/dose_20261124.json","w"),ensure_ascii=False,indent=1)
print(f"\nDONE {time.time()-t0:.0f}s",flush=True)
