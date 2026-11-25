# -*- coding: utf-8 -*-
"""Is the dose-response anything more than the ban restated?
The ban is one of the six rules, so reg_n and ban are mechanically correlated.
Tests:  (1) slope of the FIVE non-ban rules, holding the ban constant
        (2) the same within ban schools and within no-ban schools separately
        (3) departure from linearity (Wald on indicator deviations)
2026-11-24."""
import pandas as pd, numpy as np, pyreadstat, json, time, math
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
def pval(z): return math.erfc(abs(z)/math.sqrt(2))
REG=["SC190Q01JA","SC190Q02JA","SC190Q05JA","SC190Q06JA","SC190Q07JA","SC190Q08JA"]
OTHER=[c for c in REG if c!="SC190Q02JA"]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM","IC170Q02JA","IC179Q01JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+RW)
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
d["reg_other"]=d[["p_"+c for c in OTHER]].sum(axis=1,min_count=5)
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize","female"]

def est(sub,treats,y,ctrl):
    sub=sub.dropna(subset=list(treats)+[y]+list(ctrl))
    n=len(sub); codes,_=pd.factorize(sub.CNT.values); G=codes.max()+1
    M=np.column_stack([sub[t].values.astype(float) for t in treats]
                      +[sub[c].values.astype(float) for c in ctrl])
    W0=sub.W_FSTUWT.values.astype(float); RWv=sub[RW].values.astype(float)
    yv=sub[y].values.astype(float)
    def dm(A,w):
        den=np.bincount(codes,weights=w,minlength=G); o=np.empty_like(A)
        for j in range(A.shape[1]):
            o[:,j]=A[:,j]-(np.bincount(codes,weights=w*A[:,j],minlength=G)/den)[codes]
        return o
    def beta(w):
        X=dm(M,w); yc=dm(yv[:,None],w)[:,0]; sw=np.sqrt(w)
        b,*_=np.linalg.lstsq(X*sw[:,None], yc*sw, rcond=None); return b[:len(treats)]
    b0=beta(W0)
    reps=np.array([beta(RWv[:,k]) for k in range(80)])
    cov=((reps-b0)[:,:,None]*(reps-b0)[:,None,:]).sum(axis=0)/20.0
    se=np.sqrt(np.diag(cov))
    return n,b0,se,cov

out={}
dD=d.dropna(subset=["reg_n","reg_other","ban","IC179Q01JA"]+CTRL).copy()
print(f"sample {len(dD):,} students, {dD.CNT.nunique()} countries",flush=True)
print(f"corr(reg_n, ban) = {np.corrcoef(dD.reg_n,dD.ban)[0,1]:+.3f}",flush=True)

print("\n=== 1) five non-ban rules, ban held constant ===",flush=True)
n,b,se,_=est(dD,["reg_other","ban"],"IC179Q01JA",CTRL)
for lab,i in (("Number of the five other rules, per rule",0),("Attending a ban school",1)):
    print(f"  {lab:<46} {b[i]:+.4f}  SE {se[i]:.4f}  p {pval(b[i]/se[i]):.4g}",flush=True)
out['five_other_with_ban']=dict(slope=float(b[0]),se=float(se[0]),p=float(pval(b[0]/se[0])),
                                ban=float(b[1]),ban_se=float(se[1]),n=n)

print("\n=== 2) within each policy group ===",flush=True)
for nm,sub in (("ban schools",dD[dD.ban==1]),("no-ban schools",dD[dD.ban==0])):
    n2,b2,se2,_=est(sub,["reg_other"],"IC179Q01JA",CTRL)
    print(f"  {nm:<16} per other rule {b2[0]:+.4f}  SE {se2[0]:.4f}  p {pval(b2[0]/se2[0]):.4g}  n={n2:,}",flush=True)
    out['within_'+nm.split()[0]]=dict(slope=float(b2[0]),se=float(se2[0]),p=float(pval(b2[0]/se2[0])),n=n2)

print("\n=== 3) departure from linearity (reg_n) ===",flush=True)
for k in range(1,7): dD[f"r{k}"]=(dD.reg_n==k).astype(float)
nL,bL,seL,covL=est(dD,[f"r{k}" for k in range(1,7)],"IC179Q01JA",CTRL)
# contrast: indicator effects minus the linear prediction  (k * slope)
nS,bS,seS,_=est(dD,["reg_n"],"IC179Q01JA",CTRL)
slope=bS[0]
dev=bL-np.arange(1,7)*slope
# Wald on the 5 degrees of freedom of departure, using the indicator covariance
C=np.eye(6)[1:]-np.eye(6)[:-1]          # successive differences
dif=C@bL; V=C@covL@C.T
W=float(dif@np.linalg.solve(V,dif))
from scipy import stats as st
print(f"  linear slope {slope:+.4f} SE {seS[0]:.4f}",flush=True)
print(f"  indicator steps vs linear prediction: {np.round(dev,4)}",flush=True)
print(f"  Wald on the 5 successive differences: {W:.2f} on 5 df, p {st.chi2.sf(W,5):.4g}",flush=True)
out['linearity']=dict(slope=float(slope),slope_se=float(seS[0]),wald=W,df=5,p=float(st.chi2.sf(W,5)),
                      steps=[float(x) for x in bL],step_se=[float(x) for x in seL])
json.dump(out,open("out/dose2_20261124.json","w"),ensure_ascii=False,indent=1)
print(f"\nDONE {time.time()-t0:.0f}s",flush=True)
