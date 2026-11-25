# -*- coding: utf-8 -*-
"""Paper B figures from the corrected results. 2026-11-24."""
import pandas as pd, numpy as np, json, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.2,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "savefig.dpi":820,"figure.dpi":820,"legend.frameon":False,"axes.grid":True,
 "grid.color":"#E8E8E8","grid.linewidth":0.5,"axes.axisbelow":True})
BLUE="#0072B2"; VERM="#D55E00"; GREY="#8C8C8C"; GREEN="#009E73"; DARK="#333333"; CM=1/2.54
F=json.load(open("out/fixed_20261124.json"))
D1=json.load(open("out/dose_20261124.json"))
D2=json.load(open("out/dose2_20261124.json"))
cb=pd.read_csv("out/paperB_country_brr.csv")
W=17*CM

# ============================== Figure 1  what the data look like + the ladder
dist=F['B_distribution']; T=F['B_table1']
fig,ax=plt.subplots(1,2,figsize=(W,7.2*CM),gridspec_kw={"width_ratios":[1.15,0.85]})
a=ax[0]
cats=["Strongly\ndisagree","Disagree","Agree","Strongly\nagree"]
ban=[dist['ban'][str(i)] for i in (1,2,3,4)]; nob=[dist['no_ban'][str(i)] for i in (1,2,3,4)]
xx=np.arange(4); wd=0.36
a.bar(xx-wd/2-0.012,ban,wd,color=VERM,label=f"Ban schools (n = {dist['ban']['n']:,})")
a.bar(xx+wd/2+0.012,nob,wd,color=BLUE,label=f"Schools without a ban (n = {dist['no_ban']['n']:,})")
for i,(v1,v2) in enumerate(zip(ban,nob)):
    a.annotate(f"{v1:.1f}",(i-wd/2-0.012,v1),xytext=(0,2),textcoords="offset points",ha="center",fontsize=6.8,color=VERM)
    a.annotate(f"{v2:.1f}",(i+wd/2+0.012,v2),xytext=(0,2),textcoords="offset points",ha="center",fontsize=6.8,color=BLUE)
a.set_xticks(xx); a.set_xticklabels(cats,fontsize=7.0)
a.set_ylabel("Students, percent"); a.set_ylim(0,52)
a.set_title("\"Students should not be allowed to bring\nmobile phones to class\"",pad=5,fontsize=8)
a.legend(loc="upper right",handletextpad=0.5,borderpad=0.2)
b_=ax[1]
rows=[T[0],T[2]]
labs=["Attending\na ban school","Same, holding the\nstudent's own\nphone use constant"]
bb=[t['beta'] for t in rows]; ss=[t['se'] for t in rows]
yy=np.arange(2)[::-1]
b_.hlines(yy,[v-1.96*s for v,s in zip(bb,ss)],[v+1.96*s for v,s in zip(bb,ss)],color=VERM,lw=1.3)
b_.scatter(bb,yy,s=28,color=VERM,zorder=3)
for v,s,y in zip(bb,ss,yy):
    b_.annotate(f"{v:+.4f}",(v,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=7.2,color=VERM)
b_.axvline(0,color=DARK,lw=0.9)
b_.set_yticks(yy); b_.set_yticklabels(labs,fontsize=7.0)
b_.set_ylim(-0.6,1.6)
b_.set_xlabel("Change in endorsement, four-point scale")
pct=100*rows[1]['beta']/rows[0]['beta']
b_.set_title(f"Holding own use constant leaves {pct:.0f} percent\nof the difference",pad=5,fontsize=8)
fig.tight_layout(pad=0.5,w_pad=1.8); fig.savefig("figs/B_Figure1.png",bbox_inches="tight"); plt.close(fig)

# ============================== Figure 2  the dose question, answered honestly
ind=D1['B_dose_indicators']; lin=D1['B_dose_linear']
fig,ax=plt.subplots(1,2,figsize=(W,7.2*CM),gridspec_kw={"width_ratios":[1.1,0.9]})
a=ax[0]
ks=[d['reg_n'] for d in ind]; bs=[d['beta'] for d in ind]; se=[d['se'] for d in ind]
a.errorbar(ks[1:],bs[1:],yerr=[1.96*s for s in se[1:]],fmt="o",ms=5,lw=0,elinewidth=1.2,capsize=3,color=BLUE,
           label="estimated as a separate step")
a.scatter([0],[0],s=26,color=DARK,zorder=4,marker="D",label="reference: no rules")
xs=np.linspace(0,6,20)
a.plot(xs,lin['beta']*xs,color=VERM,lw=1.5,label=f"line forced through the points ({lin['beta']:+.4f} per rule)")
a.axhline(0,color=GREY,lw=0.8)
a.set_xlabel("Number of device rules the school reports")
a.set_ylabel("Change in endorsement vs schools with no rules")
a.set_title("The line is a constraint, not a pattern",pad=5,fontsize=8)
a.legend(loc="upper left",handletextpad=0.5,borderpad=0.2)
a.annotate(f"departure from the line: Wald {D2['linearity']['wald']:.1f} on 5 df, p = {D2['linearity']['p']:.4f}",
           xy=(0.5,0.02),xycoords="axes fraction",ha="center",va="bottom",fontsize=6.9,color=DARK)
b_=ax[1]
items=[("The ban itself",D2['five_other_with_ban']['ban'],D2['five_other_with_ban']['ban_se'],VERM),
       ("Each of the five other rules,\nholding the ban constant",D2['five_other_with_ban']['slope'],D2['five_other_with_ban']['se'],BLUE),
       ("Each other rule,\nwithin ban schools",D2['within_ban']['slope'],D2['within_ban']['se'],BLUE),
       ("Each other rule,\nwithin schools without a ban",D2['within_no-ban']['slope'],D2['within_no-ban']['se'],BLUE)]
yy=np.arange(len(items))[::-1]
for (lab,v,s,c),y in zip(items,yy):
    b_.hlines(y,v-1.96*s,v+1.96*s,color=c,lw=1.3)
    b_.scatter([v],[y],s=24,color=c,zorder=3)
b_.axvline(0,color=DARK,lw=0.9)
b_.set_yticks(yy); b_.set_yticklabels([i[0] for i in items],fontsize=6.9)
b_.set_xlabel("Change in endorsement, four-point scale")
b_.set_title("The dose is the ban",pad=5,fontsize=8)
b_.set_ylim(-0.7,len(items)-0.3)
fig.tight_layout(pad=0.5,w_pad=1.8); fig.savefig("figs/B_Figure2.png",bbox_inches="tight"); plt.close(fig)

# ============================== Figure 3  three items + benchmark
it=F['B_items']; bm=F['B_benchmark']
fig,ax=plt.subplots(1,2,figsize=(W,6.6*CM),gridspec_kw={"width_ratios":[1,1]})
a=ax[0]
labs=["Phones should not be\nbrought to class","Laptops should not be\nbrought to class",
      "Students should help\nset the device rules"]
yy=np.arange(3)[::-1]
cols=[VERM,VERM,GREEN]
for (t,y,c) in zip(it,yy,cols):
    a.hlines(y,t['beta']-1.96*t['se'],t['beta']+1.96*t['se'],color=c,lw=1.3)
    a.scatter([t['beta']],[y],s=26,color=c,zorder=3)
    a.annotate(f"{t['beta']:+.4f}",(t['beta'],y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=7.0,color=c)
a.axvline(0,color=DARK,lw=0.9)
a.set_yticks(yy); a.set_yticklabels(labs,fontsize=7.0); a.set_ylim(-0.7,2.6)
a.set_xlabel("Change in endorsement for students in ban schools")
a.set_title("Students in ban schools endorse more restriction\nand more say in setting it",pad=5,fontsize=8)
b_=ax[1]
bm2=sorted(bm,key=lambda d: d['beta'])
yy=np.arange(len(bm2))
cc=[VERM if d['label'].startswith("Attending") else GREY for d in bm2]
for d,y,c in zip(bm2,yy,cc):
    b_.hlines(y,d['beta']-1.96*d['se'],d['beta']+1.96*d['se'],color=c,lw=1.2)
    b_.scatter([d['beta']],[y],s=22,color=c,zorder=3)
b_.axvline(0,color=DARK,lw=0.9)
b_.set_yticks(yy); b_.set_yticklabels([d['label'] for d in bm2],fontsize=6.9)
b_.set_xlabel("Change in endorsement, four-point scale")
b_.set_title("How large is it? Against other predictors of\nthe same item",pad=5,fontsize=8)
fig.tight_layout(pad=0.5,w_pad=1.8); fig.savefig("figs/B_Figure3.png",bbox_inches="tight"); plt.close(fig)

# ============================== Figure 4  country forest
fig,ax=plt.subplots(figsize=(8.6*CM,10.4*CM))
cs=cb.sort_values("beta").reset_index(drop=True)
sig=((cs.beta-1.96*cs.se)>0)|((cs.beta+1.96*cs.se)<0)
yy=np.arange(len(cs))
ax.hlines(yy,cs.beta-1.96*cs.se,cs.beta+1.96*cs.se,color="#C8C8C8",lw=0.7)
ax.scatter(cs.beta[~sig],yy[~sig],s=8,color=GREY,zorder=3,label=f"not significant ({int((~sig).sum())})")
ax.scatter(cs.beta[sig],yy[sig],s=10,color=VERM,zorder=4,label=f"significantly positive ({int(sig.sum())})")
ax.axvline(0,color=DARK,lw=0.9)
ax.set_yticks(yy); ax.set_yticklabels(cs.CNT,fontsize=5.4)
ax.set_ylim(-1,len(cs))
ax.set_xlabel("Change in endorsement for students in ban schools")
ax.set_title(f"Positive in {int((cs.beta>0).sum())} of {len(cs)} countries,\nsignificantly negative in none",pad=5,fontsize=8)
ax.legend(loc="lower right",handletextpad=0.4,borderpad=0.2)
fig.tight_layout(pad=0.4); fig.savefig("figs/B_Figure4.png",bbox_inches="tight"); plt.close(fig)
print("B figures 1-4 written",flush=True)
