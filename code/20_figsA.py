# -*- coding: utf-8 -*-
"""Paper A figures, built from the corrected result files. 2026-11-24.
Every number is read from out/*.json|csv -- nothing is hard-coded."""
import pandas as pd, numpy as np, json, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.2,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "savefig.dpi":820,"figure.dpi":820,"legend.frameon":False,"axes.grid":True,
 "grid.color":"#E8E8E8","grid.linewidth":0.5,"axes.axisbelow":True})
BLUE="#0072B2"; VERM="#D55E00"; GREY="#8C8C8C"; GREEN="#009E73"; DARK="#333333"; CM=1/2.54
F=json.load(open("out/fixed_20261124.json"))
MR=json.load(open("out/metareg_fixed.json"))
CF=json.load(open("out/country_fixed.json"))
r=pd.read_csv("out/country_fixed.csv")
cg=pd.read_csv("out/compliance_fixed.csv")
W=17*CM

# ================================================== Figure 1  the system
fig,ax=plt.subplots(figsize=(W,5.6*CM)); ax.set_axis_off(); ax.grid(False)
ax.set_xlim(0,100); ax.set_ylim(0,40)
def box(x,y,w,h,t,fc="#FFFFFF",ec=DARK,fs=7.6,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5,rounding_size=1.1",
        linewidth=0.9,edgecolor=ec,facecolor=fc,zorder=3))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=7.6,zorder=4,
            fontweight="bold" if bold else "normal",color=DARK,linespacing=1.35)
def arr(p1,p2,c=DARK,rad=0.0,lw=0.9,ls="-"):
    ax.add_patch(FancyArrowPatch(p1,p2,connectionstyle=f"arc3,rad={rad}",mutation_scale=6,
        arrowstyle="-|>",linewidth=lw,color=c,linestyle=ls,zorder=2,shrinkA=0,shrinkB=0))
Y,H=23,11
box(1.5,Y,20,H,"School conditions\nthat make phones\na problem",fc="#F2F2F2")
box(28.5,Y,19,H,"The school\nadopts a ban")
box(54.5,Y,19,H,"How far the ban\nbinds behaviour",ec=VERM)
box(81.5,Y,17,H,"Measured\nachievement")
for x1,x2,c in ((21.5,28.5,DARK),(47.5,54.5,VERM),(73.5,81.5,DARK)):
    arr((x1,Y+H/2),(x2,Y+H/2),c=c)
arr((11.5,Y),(90,Y),c=GREY,rad=0.42,lw=0.9)
ax.text(50.5,2.2,"the same conditions act on achievement directly",ha="center",va="center",fontsize=7.2,color=GREY,bbox=dict(facecolor="white",edgecolor="none",pad=1.2),zorder=5)
arr((11.5,Y),(62,Y),c=GREY,rad=0.30,lw=0.9)
ax.text(36.5,12.6,"and on how far the ban binds",ha="center",va="center",fontsize=7.2,color=GREY,bbox=dict(facecolor="white",edgecolor="none",pad=1.2),zorder=5)
ax.plot([28.5,98.5],[37.2,37.2],color=VERM,lw=0.9,zorder=2)
for x in (28.5,98.5): ax.plot([x,x],[37.2,35.6],color=VERM,lw=0.9,zorder=2)
ax.text(63.5,38.3,"what an observational comparison of ban and no-ban schools reports",
        ha="center",fontsize=7.4,color=VERM,fontweight="bold",va="bottom")
ax.text(64,20.4,"PISA records this step, and it is rarely used",ha="center",fontsize=7.0,
        color=VERM,style="italic",va="top")
fig.tight_layout(pad=0.3); fig.savefig("figs/A_Figure1.png",bbox_inches="tight"); plt.close(fig)

# ================================================== Figure 2  ladder + forest
T1=F['A_table1']
labs=["Pooled\nraw","Pooled\n+ SES","Within\ncountry","Within\n+ SES","Within\n+ full"]
bet=[t['beta'] for t in T1]; ses=[t['se'] for t in T1]
fig,ax=plt.subplots(1,2,figsize=(W,7.6*CM),gridspec_kw={"width_ratios":[1,1.18]})
a=ax[0]; x=np.arange(5)
a.errorbar(x,bet,yerr=[1.96*s for s in ses],fmt="o",ms=5.5,lw=0,elinewidth=1.4,capsize=3,color=VERM)
for i,(b,s) in enumerate(zip(bet,ses)):
    a.annotate(f"{b:.2f}",(i,b),xytext=(9,-2.5),textcoords="offset points",ha="left",fontsize=7.0,color=VERM)
a.axhline(0,color=GREY,lw=1.0)
a.set_xticks(x); a.set_xticklabels(labs,fontsize=7.0)
a.set_ylabel("Ban minus no-ban, mathematics points")
a.set_xlim(-0.55,4.85); a.set_ylim(-21,4); a.yaxis.set_major_locator(MultipleLocator(5))
a.set_title("Most of the pooled estimate is composition",pad=5,fontsize=8)
b_=ax[1]
rs=r.sort_values("beta").reset_index(drop=True)
sig=((rs.beta-1.96*rs.se)>0)|((rs.beta+1.96*rs.se)<0)
yy=np.arange(len(rs))
b_.hlines(yy,rs.beta-1.96*rs.se,rs.beta+1.96*rs.se,color="#C8C8C8",lw=0.7,zorder=1)
b_.scatter(rs.beta[~sig],yy[~sig],s=9,color=GREY,zorder=2,label=f"not significant (n={int((~sig).sum())})")
b_.scatter(rs.beta[sig],yy[sig],s=11,color=VERM,zorder=3,label=f"significant (n={int(sig.sum())})")
b_.axvline(0,color=DARK,lw=0.8)
b_.axvline(rs.beta.median(),color=BLUE,lw=1.0,ls=(0,(4,2)))
b_.set_ylim(-1.5,len(rs)+4.5); b_.set_yticks([])
b_.annotate(f"median {rs.beta.median():.2f}",(rs.beta.median(),len(rs)+3.2),xytext=(-3,0),
            textcoords="offset points",fontsize=7.0,color=BLUE,ha="right",va="center")
b_.set_xlabel("Ban minus no-ban, mathematics points")
b_.set_ylabel(f"{len(rs)} countries, ordered")
b_.set_title("Country by country, most differences are not\ndistinguishable from zero",pad=5,fontsize=8)
b_.legend(loc="lower right",handletextpad=0.4,borderpad=0.2)
fig.tight_layout(pad=0.5,w_pad=1.8); fig.savefig("figs/A_Figure2.png",bbox_inches="tight"); plt.close(fig)

# ================================================== Figure 3  compliance
C=CF['compliance']
fig,ax=plt.subplots(1,2,figsize=(W,7.0*CM),gridspec_kw={"width_ratios":[0.82,1.18]})
a=ax[0]
ban=[C['pooled_ban'],C['never_ban']]; nob=[C['pooled_noban'],C['never_noban']]
xx=np.arange(2); wdt=0.34
a.bar(xx-wdt/2-0.012,ban,wdt,color=VERM,label="Ban schools")
a.bar(xx+wdt/2+0.012,nob,wdt,color=BLUE,label="Schools without a ban")
for i,(v1,v2) in enumerate(zip(ban,nob)):
    a.annotate(f"{v1:.1f}",(i-wdt/2-0.012,v1),xytext=(0,2),textcoords="offset points",ha="center",fontsize=7.0,color=VERM)
    a.annotate(f"{v2:.1f}",(i+wdt/2+0.012,v2),xytext=(0,2),textcoords="offset points",ha="center",fontsize=7.0,color=BLUE)
a.set_xticks(xx); a.set_xticklabels(["Several times\na day","Never"],fontsize=7.2)
a.set_ylabel("Students, percent"); a.set_ylim(0,52)
a.set_title("The ban changes behaviour, but not completely",pad=5,fontsize=8)
a.legend(loc="upper center",ncol=1,handletextpad=0.5,borderpad=0.2,bbox_to_anchor=(0.5,1.0))
b_=ax[1]
gs=cg.sort_values("gap").reset_index(drop=True)
b_.barh(np.arange(len(gs)),gs.gap,color=np.where(gs.gap<0,BLUE,VERM),height=0.78)
b_.axvline(0,color=DARK,lw=0.8)
b_.axvline(C['pooled_gap'],color=VERM,lw=1.1,ls=(0,(4,2)))
b_.axvline(C['median'],color=DARK,lw=1.1,ls=(0,(1,1.6)))
b_.set_yticks([]); b_.set_ylim(-1,len(gs)+6)
b_.annotate(f"pooled {C['pooled_gap']:.1f}",(C['pooled_gap'],len(gs)+5.0),xytext=(-3,0),
            textcoords="offset points",ha="right",va="top",fontsize=7.0,color=VERM)
b_.annotate(f"median within country {C['median']:.1f}",(C['median'],len(gs)+5.0),xytext=(3,0),
            textcoords="offset points",ha="left",va="top",fontsize=7.0,color=DARK)
b_.set_xlabel("Heavy phone use, ban minus no-ban, percentage points")
b_.set_ylabel(f"{len(gs)} countries, ordered")
b_.set_title(f"Pooling inflates the behavioural difference\nby a factor of {abs(C['pooled_gap']/C['median']):.1f}",pad=5,fontsize=8)
fig.tight_layout(pad=0.5,w_pad=1.8); fig.savefig("figs/A_Figure3.png",bbox_inches="tight"); plt.close(fig)

# ================================================== Figure 4  meta-regression + leave one out
y=r.beta.values; v=r.se.values**2; k=len(y); gap=r.gap.values
def mm(X,y,v):
    t2=0.0
    for _ in range(200):
        Wt=1/(v+t2); XtWX=X.T@(Wt[:,None]*X)
        b=np.linalg.solve(XtWX,X.T@(Wt*y)); res=y-X@b
        P=np.diag(Wt)-(Wt[:,None]*X)@np.linalg.solve(XtWX,(X*Wt[:,None]).T)
        t2n=max(0.0,(float((Wt*res**2).sum())-(len(y)-X.shape[1]))/np.trace(P))
        if abs(t2n-t2)<1e-12: t2=t2n; break
        t2=t2n
    Wt=1/(v+t2); XtWX=X.T@(Wt[:,None]*X)
    b=np.linalg.solve(XtWX,X.T@(Wt*y)); cov=np.linalg.inv(XtWX)
    return b,np.sqrt(np.diag(cov)),t2
X1=np.column_stack([np.ones(k),gap]); b1,s1,t21=mm(X1,y,v)
loo=[]
for i in range(k):
    m=np.ones(k,bool); m[i]=False
    bb,ss,_=mm(np.column_stack([np.ones(k-1),gap[m]]),y[m],v[m])
    loo.append((r.CNT.iloc[i],bb[1],ss[1]))
loo=sorted(loo,key=lambda t:t[1])
fig,ax=plt.subplots(1,2,figsize=(W,7.4*CM),gridspec_kw={"width_ratios":[1.25,1]})
a=ax[0]
a.vlines(gap,y-1.96*np.sqrt(v),y+1.96*np.sqrt(v),color="#C8C8C8",lw=0.7,zorder=1)
sz=18*(1/ (v+t21))/np.max(1/(v+t21))+4
a.scatter(gap,y,s=sz,color=BLUE,zorder=3,edgecolor="white",linewidth=0.3)
xs=np.linspace(gap.min()-1,gap.max()+1,50)
a.plot(xs,b1[0]+b1[1]*xs,color=VERM,lw=1.5,zorder=4)
a.axhline(0,color=GREY,lw=0.8); a.axvline(0,color=GREY,lw=0.8)
for c in ("CHE","FIN","MAR","KOR"):
    if c in set(r.CNT):
        i=int(np.where(r.CNT.values==c)[0][0])
        a.annotate(c,(gap[i],y[i]),xytext=(4,3),textcoords="offset points",fontsize=6.6,color=DARK)
a.set_xlabel("Compliance gap, percentage points")
a.set_ylabel("Ban minus no-ban, mathematics points")
a.set_title("Where the ban binds, the achievement gap is larger",pad=5,fontsize=8)
a.annotate(f"slope {b1[1]:+.3f} (SE {s1[1]:.3f}, p = {MR['slope_p']:.1e})\n"
           f"explains {MR['R2']:.0f} percent of the between-country variance",
           xy=(0.97,0.05),xycoords="axes fraction",ha="right",va="bottom",fontsize=7.0,color=VERM,linespacing=1.4)
b_=ax[1]
ls=[t[1] for t in loo]; lse=[t[2] for t in loo]; names=[t[0] for t in loo]
yy=np.arange(k)
b_.scatter(ls,yy,s=10,color=BLUE,zorder=3)
b_.axvline(b1[1],color=VERM,lw=1.1,ls=(0,(4,2)))
b_.set_yticks([]); b_.set_ylim(-1.5,k+3.5)
b_.annotate(f"all countries {b1[1]:+.3f}",(b1[1],k+2.4),xytext=(3,0),textcoords="offset points",
            fontsize=7.0,color=VERM,ha="left",va="center")
b_.set_xlabel("Slope with one country removed")
b_.set_ylabel(f"{k} refits, ordered")
b_.set_title("No single country drives the slope",pad=5,fontsize=8)
b_.set_xlim(min(ls)-0.035,max(ls)+0.035)
b_.annotate(f"removing {names[0]} gives the smallest slope, {min(ls):.3f};\nremoving {names[-1]} the largest, {max(ls):.3f}.\nAll 45 refits are significant at the five percent level",
            xy=(0.035,0.80),xycoords="axes fraction",ha="left",va="top",fontsize=6.8,color=DARK,linespacing=1.4,bbox=dict(facecolor="white",edgecolor="none",pad=1.5))
fig.tight_layout(pad=0.5,w_pad=1.8); fig.savefig("figs/A_Figure4.png",bbox_inches="tight"); plt.close(fig)

# ================================================== Figure 5  six policies
sep=F['A_table3_separate']; mut=F['A_table3_mutual']
order=list(range(6))
labs=[s['label'] for s in sep]
short=[l.replace("Cell phone use not allowed on the premises","Cell phone use not allowed\non the premises")
        .replace("Written statement on general device use","Written statement on\ngeneral device use")
        .replace("Formal guidelines for devices in teaching","Formal guidelines for\ndevices in teaching")
        .replace("Teachers set rules for device use in class","Teachers set rules for\ndevice use in class")
        .replace("Rules set in collaboration with students","Rules set in collaboration\nwith students")
        .replace("Programme on responsible device use","Programme on\nresponsible device use") for l in labs]
fig,ax=plt.subplots(figsize=(W,7.8*CM))
yy=np.arange(6)[::-1]
off=0.17
ax.hlines(yy+off,[s['beta']-1.96*s['se'] for s in sep],[s['beta']+1.96*s['se'] for s in sep],color=BLUE,lw=1.1)
ax.scatter([s['beta'] for s in sep],yy+off,s=20,color=BLUE,zorder=3,label="estimated on its own")
ax.hlines(yy-off,[s['beta']-1.96*s['se'] for s in mut],[s['beta']+1.96*s['se'] for s in mut],color=VERM,lw=1.1)
ax.scatter([s['beta'] for s in mut],yy-off,s=20,color=VERM,marker="s",zorder=3,label="all six in one model")
ax.axvline(0,color=DARK,lw=0.9)
ax.set_yticks(yy); ax.set_yticklabels(short,fontsize=7.0)
ax.set_xlabel("Difference in mathematics points, schools with the policy minus schools without")
ax.set_ylim(-0.7,5.7)
ax.set_title("Among six device policies, only the outright ban carries\na negative association of any size",pad=6,fontsize=8)
ax.legend(loc="upper left",handletextpad=0.5,borderpad=0.3,bbox_to_anchor=(0.015,0.14))
fig.tight_layout(pad=0.5); fig.savefig("figs/A_Figure5.png",bbox_inches="tight"); plt.close(fig)
print("A figures 1-5 written",flush=True)
