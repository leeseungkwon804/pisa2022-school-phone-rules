import pandas as pd, numpy as np, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.2,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
 "savefig.dpi":820,"figure.dpi":820,"legend.frameon":False,"axes.grid":True,
 "grid.color":"#E8E8E8","grid.linewidth":0.5,"axes.axisbelow":True})
BLUE="#0072B2"; VERM="#D55E00"; GREY="#8C8C8C"; GREEN="#009E73"; CM=1/2.54
R="figs"
b1=json.load(open("out/paperB.json")); b2=json.load(open("out/paperB2.json"))
G={x["label"]:x for x in b1+b2}

# Figure B1: the effect survives controlling behaviour, and is a dose response
fig,ax=plt.subplots(1,2,figsize=(16*CM,6.8*CM))
a=ax[0]
keys=["ban school -> endorsement(1-4)","ban -> endorsement, own use held constant"]
labs=["Ban school","Ban school,\nown phone use\nheld constant"]
bet=[G[k]["beta"] for k in keys]; ses=[G[k]["se"] for k in keys]
a.bar(range(2),bet,0.45,yerr=[1.96*s for s in ses],color=[BLUE,GREEN],
      edgecolor="white",lw=0.8,capsize=4,error_kw=dict(elinewidth=1.2))
for i,(bb,ss) in enumerate(zip(bet,ses)):
    a.annotate(f"{bb:+.4f}",(i,bb+1.96*ss),xytext=(0,5),textcoords="offset points",ha="center",fontsize=7.4)
a.axhline(0,color=GREY,lw=1.0)
a.set_xticks(range(2)); a.set_xticklabels(labs,fontsize=7.0)
a.set_ylabel("Change in support for a ban, 1 to 4 scale")
a.set_ylim(0,0.115); a.yaxis.set_major_locator(MultipleLocator(0.02))
a.set_title("Living under the rule, not using the phone less",pad=5,fontsize=8)
b=ax[1]
k2=["rule count -> endorsement","rule count -> endorsement, own use held constant"]
l2=["All students","Own phone use\nheld constant"]
bt=[G[k]["beta"] for k in k2]; se2=[G[k]["se"] for k in k2]
xs=np.arange(0,7)
for i,(bb,ss,lb,col) in enumerate(zip(bt,se2,l2,[BLUE,GREEN])):
    b.plot(xs,bb*xs,lw=1.9,color=col,label=f"{lb}  ({bb:+.4f} per rule)")
    b.fill_between(xs,(bb-1.96*ss)*xs,(bb+1.96*ss)*xs,color=col,alpha=.15,lw=0)
b.set_xlabel("Number of digital-device rules the school reports (0 to 6)")
b.set_ylabel("Change in support for a ban")
b.set_xticks(xs); b.set_ylim(0,0.135); b.yaxis.set_major_locator(MultipleLocator(0.03))
b.set_title("A dose response, not a one-off difference",pad=5,fontsize=8)
b.legend(loc="upper left",handlelength=1.4)
fig.savefig(f"{R}/FigureB1.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# Figure B2: convergent items + country distribution
fig,ax=plt.subplots(1,2,figsize=(16*CM,6.8*CM))
a=ax[0]
ks=["ban school -> endorsement(1-4)","ban -> laptops should not be brought","ban -> students should help set rules"]
ls=["Phones should not\nbe brought to class","Laptops should not\nbe brought to class","Students should help\nset the rules"]
bb=[G[k]["beta"] for k in ks]; ss=[G[k]["se"] for k in ks]
y=np.arange(3)[::-1]
a.errorbar(bb,y,xerr=[1.96*s for s in ss],fmt="o",ms=6,lw=0,elinewidth=1.4,capsize=3,color=VERM)
for i,(v,s) in enumerate(zip(bb,ss)):
    a.annotate(f"{v:+.4f}",(v,y[i]),xytext=(0,10),textcoords="offset points",ha="center",fontsize=7.2,color=VERM)
a.axvline(0,color=GREY,lw=1.0)
a.set_yticks(y); a.set_yticklabels(ls,fontsize=7.0)
a.set_xlabel("Effect of attending a ban school, 1 to 4 scale")
a.set_xlim(-0.01,0.105); a.set_ylim(-0.6,2.7)
a.set_title("Three attitude items move together",pad=5,fontsize=8)
b=ax[1]
r=pd.read_csv("out/paperB_country.csv")
b.hist(r.beta,bins=16,color=BLUE,alpha=.85)
b.axvline(0,color="#444444",lw=1.2)
b.axvline(r.beta.median(),color=VERM,lw=1.4,ls=(0,(4,2)))
b.annotate(f"median {r.beta.median():+.3f}\n{int((r.beta>0).sum())} of {len(r)} positive",
           (r.beta.median(),b.get_ylim()[1]*0.92),xytext=(6,0),textcoords="offset points",
           fontsize=7,color=VERM,va="top")
b.set_xlabel("Country-specific effect of attending a ban school")
b.set_ylabel("Countries")
b.set_title(f"Consistent across {len(r)} countries",pad=5,fontsize=8)
fig.savefig(f"{R}/FigureB2.png",bbox_inches="tight",pad_inches=0.02); plt.close(fig)
print("written",[f for f in sorted(os.listdir(R)) if f.startswith("FigureB")])
