import pandas as pd, numpy as np, pyreadstat, json, time
from scipy import stats
t0=time.time()
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
BASE=["CNTSCHID","CNT","W_FSTUWT","ESCS","ST004D01T","DISCLIM",
      "IC170Q02JA","IC179Q01JA","IC179Q02JA","IC179Q03JA"]
d,m=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+RW)
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
    usecols=["CNTSCHID","SC190Q02JA","SC013Q01TA","SC002Q01TA","SC002Q02TA"])
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.IC170Q02JA.notna() & (d.IC170Q02JA<=5) & d.SC190Q02JA.notna() & d.IC179Q01JA.notna()]
d["ban"]=(d.SC190Q02JA==1).astype(float)
d["private"]=(d.SC013Q01TA==2).astype(float)
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d["female"]=(d.ST004D01T==1).astype(float)
# attitude: 1 strongly disagree .. 4 strongly agree (higher = more endorsement)
d["support"]=d.IC179Q01JA.astype(float)
d["support_hi"]=(d.IC179Q01JA>=3).astype(float)
d["heavy"]=(d.IC170Q02JA>=5).astype(float)
d=d.dropna(subset=["ESCS","DISCLIM","sch_escs","schsize","private","support"])
print(f"sample {len(d):,} · schools {d.CNTSCHID.nunique():,} · countries {d.CNT.nunique()} | {time.time()-t0:.0f}s",flush=True)
w=d.W_FSTUWT
print()
print("=== descriptives: endorsement of a ban ===",flush=True)
for b,lab in ((1,"ban schools"),(0,"no-ban schools")):
    g=d[d.ban==b]; ww=g.W_FSTUWT
    print(f"  {lab}: endorse (agree or strongly agree) {np.average(g.support_hi,weights=ww)*100:5.1f}%  mean {np.average(g.support,weights=ww):.3f}  n={len(g):,}",flush=True)
a=d[d.ban==1]; c=d[d.ban==0]
print(f"  difference {np.average(a.support_hi,weights=a.W_FSTUWT)*100-np.average(c.support_hi,weights=c.W_FSTUWT)*100:+.1f}pp",flush=True)
CTRL=["ESCS","DISCLIM","sch_escs","private","schsize","female"]
def beta(X,y,wv):
    sw=np.sqrt(wv); b,*_=np.linalg.lstsq(X*sw[:,None], y*sw, rcond=None); return b[1]
def est(df, treat, y, label, extra=()):
    df=df.dropna(subset=[treat,y]+list(CTRL)+list(extra))
    n=len(df)
    X=np.column_stack([np.ones(n), df[treat].values.astype(float)]
        +[df[c].values.astype(float) for c in list(CTRL)+list(extra)]
        +[pd.get_dummies(df.CNT,drop_first=True).values.astype(float)])
    yv=df[y].values; w0=df.W_FSTUWT.values; RWv=df[RW].values
    b0=beta(X,yv,w0)
    U=sum((beta(X,yv,RWv[:,k])-b0)**2 for k in range(80))/20.0
    se=np.sqrt(U); p=2*(1-stats.norm.cdf(abs(b0/se)))
    print(f"  {label:<48} {b0:+7.4f}  SE {se:.4f}  p {p:.4g}  n={n:,}",flush=True)
    return dict(label=label,beta=float(b0),se=float(se),p=float(p),n=n)
res=[]
print("\n=== policy to attitude (countriesFE + full controls, BRR80) ===",flush=True)
res.append(est(d,"ban","support","ban school -> endorsement(1-4)"))
res.append(est(d,"ban","support_hi","ban school -> endorsement(binary)"))
print("\n=== adding the behaviour control ===",flush=True)
res.append(est(d,"ban","support","ban -> endorsement, own use held constant",extra=("heavy",)))
print("\n=== how own use relates to the attitude ===",flush=True)
res.append(est(d,"heavy","support","heavy use -> endorsement"))
json.dump(res,open("out/paperB.json","w"),ensure_ascii=False,indent=1)
print(f"\n{time.time()-t0:.0f}s",flush=True)
