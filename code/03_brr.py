import pandas as pd, numpy as np, pyreadstat, json, time
t0=time.time()
PV=[f"PV{i}MATH" for i in range(1,11)]
RW=[f"W_FSTURWT{i}" for i in range(1,81)]
BASE=["CNTSCHID","CNTSTUID","CNT","W_FSTUWT","ESCS","DISCLIM","IC170Q02JA","IC179Q01JA"]
d,_=pyreadstat.read_sav("data/CY08MSP_STU_QQQ.SAV", usecols=BASE+PV+RW)
s,_=pyreadstat.read_sav("data/CY08MSP_SCH_QQQ.SAV",
      usecols=["CNTSCHID","SC190Q02JA","SC013Q01TA","SC002Q01TA","SC002Q02TA"])
d=d.merge(s,on="CNTSCHID",how="inner")
d=d[d.SC190Q02JA.notna() & d.IC170Q02JA.notna() & (d.IC170Q02JA<=5)]
d["ban"]=(d.SC190Q02JA==1).astype(float)
d["private"]=(d.SC013Q01TA==2).astype(float)
d["schsize"]=np.log1p(d[["SC002Q01TA","SC002Q02TA"]].sum(axis=1,min_count=1))
d["sch_escs"]=d.groupby("CNTSCHID").ESCS.transform("mean")
d=d.dropna(subset=["ESCS","DISCLIM","sch_escs","schsize","private"]+PV)
print(f"sample {len(d):,} · schools {d.CNTSCHID.nunique():,} · countries {d.CNT.nunique()} | {time.time()-t0:.0f}s",flush=True)

def wls_beta(X, y, w, idx=1):
    sw=np.sqrt(w)
    Xw=X*sw[:,None]; yw=y*sw
    b,*_=np.linalg.lstsq(Xw, yw, rcond=None)
    return b[idx]

def pisa_estimate(df, cols, label, cnt_fe=False):
    n=len(df)
    X=np.column_stack([np.ones(n), df["ban"].values]+[df[c].values.astype(float) for c in cols])
    if cnt_fe:
        cd=pd.get_dummies(df.CNT, drop_first=True).values.astype(float)
        X=np.hstack([X,cd])
    w0=df.W_FSTUWT.values
    RWv=df[RW].values
    th=[]; samp=[]
    for p in PV:
        y=df[p].values
        b0=wls_beta(X,y,w0)
        th.append(b0)
        v=sum((wls_beta(X,y,RWv[:,g])-b0)**2 for g in range(80))/20.0
        samp.append(v)
    th=np.array(th); U=np.mean(samp)
    B=th.var(ddof=1)
    V=U+(1+1/10)*B
    se=np.sqrt(V)
    df_=1  # normal approximation for reporting
    print(f"  {label:<34} beta {th.mean():+7.2f}  SE {se:5.2f}  "
          f"[{th.mean()-1.96*se:+7.2f}, {th.mean()+1.96*se:+7.2f}]  "
          f"p {2*(1-__import__(chr(115)+chr(99)+chr(105)+chr(112)+chr(121)).stats.norm.cdf(abs(th.mean()/se))):.3g}  "
          f"(samp {np.sqrt(U):.2f} / imp {np.sqrt((1+1/10)*B):.2f})",flush=True)
    return dict(label=label,beta=float(th.mean()),se=float(se),U=float(U),B=float(B))

print("\n=== ban and mathematics achievement: PV10 + BRR80 ===",flush=True)
res=[]
res.append(pisa_estimate(d,[],"1) pooled, no adjustment"))
res.append(pisa_estimate(d,["ESCS"],"2) pooled + student SES"))
res.append(pisa_estimate(d,[],"3) within country",cnt_fe=True))
res.append(pisa_estimate(d,["ESCS"],"4) within country + student SES",cnt_fe=True))
res.append(pisa_estimate(d,["ESCS","DISCLIM","sch_escs","private","schsize"],"5) within country + full controls",cnt_fe=True))
json.dump(res,open("out/brr_pooled.json","w"),ensure_ascii=False,indent=1)
print(f"\ntotal {time.time()-t0:.0f}s",flush=True)
