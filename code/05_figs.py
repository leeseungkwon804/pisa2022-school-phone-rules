import pandas as pd, numpy as np, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.2,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "savefig.dpi":820,"figure.dpi":820,"legend.frameon":False,"axes.grid":True,
 "grid.color":"#E8E8E8","grid.linewidth":0.5,"axes.axisbelow":True})
BLUE="#0072B2"; VERM="#D55E00"; GREY="#8C8C8C"; GREEN="#009E73"; CM=1/2.54
R="figs"
r=pd.read_csv("out/country_brr.csv")
pooled=json.load(open("out/brr_pooled.json")) if os.path.exists("out/brr_pooled.json") else []

# ---- Figure 1: pooled estimate shrinks as comparison tightens ----
labs=["Pooled\nraw","Pooled\n+ SES","Country\nFE","Country FE\n+ SES"]
bet=[-13.35,-5.08,-6.81,-5.43]; ses=[2.54,1.95,2.67,2.23]
fig,ax=plt.subplots(1,2,figsize=(16*CM,7.6*CM))
a=ax[0]
x=np.arange(len(bet))
a.errorbar(x,bet,yerr=[1.96*s for s in ses],fmt="o",ms=6,lw=0,elinewidth=1.4,capsize=3,color=VERM)
for i,(b,s) in enumerate(zip(bet,ses)):
    a.annotate(f"{b:.2f}",(i,b),xytext=(13,-3),textcoords="offset points",ha="left",fontsize=7.2,color=VERM)
a.axhline(0,color=GREY,lw=1.0)
a.set_xticks(x); a.set_xticklabels(labs,fontsize=7.0)
a.set_ylabel("Ban minus no-ban, mathematics points")
a.set_xlim(-0.5,3.8); a.set_ylim(-21,4); a.yaxis.set_major_locator(MultipleLocator(5))
a.set_title("Most of the pooled estimate is composition",pad=5,fontsize=8)
# ---- right: country forest-ish scatter ordered ----
b=ax[1]
rs=r.sort_values("beta").reset_index(drop=True)
yy=np.arange(len(rs))
sig=(rs.beta-1.96*rs.se>0)|(rs.beta+1.96*rs.se<0)
b.errorbar(rs.beta,yy,xerr=1.96*rs.se,fmt="none",elinewidth=0.7,color="#BBBBBB")
b.scatter(rs.beta[~sig],yy[~sig],s=11,color=GREY,label=f"not significant (n={int((~sig).sum())})",zorder=3)
b.scatter(rs.beta[sig],yy[sig],s=15,color=VERM,label=f"significant (n={int(sig.sum())})",zorder=3)
b.axvline(0,color="#444444",lw=1.0)
b.axvline(rs.beta.median(),color=BLUE,lw=1.2,ls=(0,(4,2)))
b.set_xlim(-32,36)
b.annotate(f"median {rs.beta.median():.2f}",(rs.beta.median(),len(rs)-1),
           xytext=(-5,-2),textcoords="offset points",fontsize=6.9,color=BLUE,va="top",ha="right")
b.set_yticks([]); b.set_xlabel("Ban minus no-ban, mathematics points")
b.set_ylabel(f"{len(rs)} countries, ordered")
b.set_title("Country by country, most differences are not\ndistinguishable from zero",pad=4,fontsize=8)
b.legend(loc="lower right",handlelength=1.0)
fig.savefig(f"{R}/Figure1.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# ---- Figure 2: compliance explains the heterogeneity ----
fig,ax=plt.subplots(figsize=(16*CM,9.6*CM))
se=r.se.values; tau=4.80
wv=1/(se**2+tau**2)
ax.errorbar(r.gap,r.beta,yerr=1.96*se,fmt="none",elinewidth=0.6,color="#CCCCCC",zorder=1)
sc=ax.scatter(r.gap,r.beta,s=14+np.sqrt(r.n)/8,color=BLUE,alpha=.85,zorder=3)
X=np.column_stack([np.ones(len(r)),r.gap.values]); sw=np.sqrt(wv)
bb=np.linalg.lstsq(X*sw[:,None], r.beta.values*sw, rcond=None)[0]
xs=np.linspace(r.gap.min()-1,r.gap.max()+1,50)
ax.plot(xs,bb[0]+bb[1]*xs,color=VERM,lw=1.8,zorder=4)
ax.axhline(0,color=GREY,lw=1.0); ax.axvline(0,color=GREY,lw=1.0)
OFF={"CZE":(-8,7),"CHE":(7,-2),"BRN":(7,6),"KOR":(6,5),"FIN":(7,-3),"MAR":(-8,6)}
HA ={"CZE":"right","CHE":"left","BRN":"left","KOR":"left","FIN":"left","MAR":"right"}
for cc,off in OFF.items():
    s=r[r.CNT==cc]
    if len(s): ax.annotate(cc,(s.gap.iloc[0],s.beta.iloc[0]),xytext=off,textcoords="offset points",
                           fontsize=6.6,color="#333333",ha=HA[cc])
ax.set_xlabel("Compliance gap: heavy phone use in ban minus no-ban schools, percentage points")
ax.set_ylabel("Ban minus no-ban, mathematics points")
ax.set_ylim(-34,42)
ax.annotate(f"meta-regression slope {bb[1]:+.3f} (SE 0.136, p = 0.0008)\nexplains 25.6% of the between-country variance",
            (0.985,0.025),xycoords="axes fraction",fontsize=7,color=VERM,va="bottom",ha="right")
ax.set_title("Where the ban binds, the achievement gap is larger",pad=6)
fig.savefig(f"{R}/Figure2.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)
print("written", os.listdir(R))
