# -*- coding: utf-8 -*-
"""32_figsB34.py  2026-11-25   Paper B Figures 3 and 4, rebuilt for the new argument."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8,
    "xtick.labelsize":7.4,"ytick.labelsize":7.4,"axes.titlesize":8,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7})
BLUE,VERM,GREY,GREEN,DARK = "#0072B2","#D55E00","#8C8C8C","#009E73","#333333"
CM = 1/2.54; W = 17*CM; DPI = 820
def clean(a):
    for sp in ("top","right"): a.spines[sp].set_visible(False)
    a.grid(True,color="#E8E8E8",lw=0.5,zorder=0); a.set_axisbelow(True)

I = json.load(open("out/ident_20261125.json"))
A = json.load(open("out/acq_20261125.json"))

# ======================================================= Figure 3  specificity
SHORT = {"IC179Q01JA":"Phones not brought to class","IC179Q02JA":"Laptops not brought to class",
         "IC179Q04JA":"School filters social media","IC179Q05JA":"School filters online games",
         "IC179Q06JA":"Teachers monitor laptop use","IC179Q03JA":"Students help set the rules"}
dev = {r["item"]: r for r in I["B_items6"]}
ips = {r["item"]: r for r in I["B_ipsatised"]}
order = ["IC179Q01JA","IC179Q02JA","IC179Q05JA","IC179Q04JA","IC179Q06JA","IC179Q03JA"]

fig, ax = plt.subplots(1, 2, figsize=(W, 8.2*CM), gridspec_kw={"width_ratios":[1.0,1.12]})
a = ax[0]
y = np.arange(len(order))[::-1]
for k, it in enumerate(order):
    r, q = dev[it], ips[it]
    a.plot([r["beta"]-1.96*r["se"], r["beta"]+1.96*r["se"]], [y[k]+0.16]*2, color=BLUE, lw=1.4, solid_capstyle="butt")
    a.plot(r["beta"], y[k]+0.16, "o", ms=4.2, color=BLUE, mec="white", mew=0.5, zorder=3)
    a.plot([q["beta"]-1.96*q["se"], q["beta"]+1.96*q["se"]], [y[k]-0.16]*2, color=VERM, lw=1.4, solid_capstyle="butt")
    a.plot(q["beta"], y[k]-0.16, "o", ms=4.2, color="white", mec=VERM, mew=1.1, zorder=3)
a.axvline(0, color=DARK, lw=0.8, zorder=2)
a.set_yticks(y); a.set_yticklabels([SHORT[i] for i in order])
a.set_xlabel("Association with attending a ban school, scale points")
a.set_title("Six device-regulation items", pad=5)
a.legend(handles=[Line2D([],[],marker="o",color=BLUE,ms=4.2,lw=1.4,label="as reported"),
                  Line2D([],[],marker="o",color=VERM,ms=4.2,lw=1.4,mfc="white",label="own mean removed")],
         loc="upper left", frameon=True, framealpha=1.0, edgecolor="none",
         fontsize=7, handlelength=1.5, borderpad=0.3)
a.set_ylim(-0.75, len(order)-0.1)
clean(a)

b = ax[1]
items = [r for r in A["items"] if r["battery"] != "Device rules (IC179)"]
BATS = ["Perseverance (ST307)","Self-control (ST309)","Curiosity (ST301)","Sense of belonging (ST034)"]
devlo = min(dev[i]["beta"] for i in order); devhi = max(dev[i]["beta"] for i in order)
b.axvspan(devlo, devhi, color=BLUE, alpha=0.10, zorder=0)
pos, ticks, labs = 0, [], []
for nm in BATS:
    g = [r for r in items if r["battery"] == nm]
    g = sorted(g, key=lambda r: (r["key"] != "forward", r["beta"]))
    start = pos
    for r in g:
        c = GREEN if r["key"] == "forward" else VERM
        mfc = c if r["key"] == "forward" else "white"
        b.plot([r["beta"]-1.96*r["se"], r["beta"]+1.96*r["se"]], [pos]*2, color=c, lw=1.0, alpha=0.85, solid_capstyle="butt")
        b.plot(r["beta"], pos, "o", ms=3.2, color=mfc, mec=c, mew=0.9, zorder=3)
        pos -= 1
    ticks.append((start + pos + 1)/2.0); labs.append(nm.split(" (")[0]); pos -= 1.2
b.axvline(0, color=DARK, lw=0.8, zorder=2)
b.set_yticks(ticks); b.set_yticklabels(labs)
b.set_xlabel("Association with attending a ban school, scale points")
b.set_title("33 items that have nothing to do with devices", pad=5)
b.set_ylim(pos-1.9, 1.9)
b.annotate("range of the six\ndevice items", xy=((devlo+devhi)/2, 1.75),
           ha="center", va="top", fontsize=6.8, color=BLUE)
b.legend(handles=[Line2D([],[],marker="o",color=GREEN,ms=3.2,lw=1.0,label="worded positively"),
                  Line2D([],[],marker="o",color=VERM,ms=3.2,lw=1.0,mfc="white",label="worded negatively")],
         loc="lower left", frameon=True, framealpha=1.0, edgecolor="none",
         fontsize=7, handlelength=1.5, borderpad=0.3)
clean(b)
fig.tight_layout(pad=0.5, w_pad=1.6)
fig.savefig("figs/B_Figure3.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)
print("B_Figure3 done", flush=True)

# ======================================================= Figure 4  countries + discretion
cc = pd.DataFrame(I["B_discretion"]["countries"])
fo = pd.read_csv("out/paperB_country_brr.csv").sort_values("beta").reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(W, 8.6*CM), gridspec_kw={"width_ratios":[1.0,1.0]})
a = ax[0]
yy = np.arange(len(fo))
sig = (fo.beta - 1.96*fo.se) > 0
a.hlines(yy, fo.beta-1.96*fo.se, fo.beta+1.96*fo.se, color="#C8C8C8", lw=0.7, zorder=1)
a.scatter(fo.beta[~sig], yy[~sig], s=9, color=GREY, zorder=3, edgecolor="white", linewidth=0.3)
a.scatter(fo.beta[sig], yy[sig], s=11, color=BLUE, zorder=3, edgecolor="white", linewidth=0.3)
a.axvline(0, color=DARK, lw=0.8, zorder=2)
a.set_yticks([]); a.set_ylabel(f"{len(fo)} countries, ordered")
a.set_xlabel("Association with endorsement, scale points")
a.set_title(f"Positive in {int((fo.beta>0).sum())} of {len(fo)}, significantly\n"
            f"positive in {int(sig.sum())}, negative in none", pad=5)
clean(a)

b = ax[1]
b.hlines(cc.beta, cc.prev*0+0, cc.prev*0+0, color="none")
b.vlines(cc.prev, cc.beta-1.96*cc.se, cc.beta+1.96*cc.se, color="#C8C8C8", lw=0.7, zorder=1)
b.scatter(cc.prev, cc.beta, s=16, color=BLUE, zorder=3, edgecolor="white", linewidth=0.3)
b.axhline(0, color=DARK, lw=0.8, zorder=2)
M = I["B_discretion"]["moderators"]
xs = np.linspace(cc.prev.min(), cc.prev.max(), 100)
pooled = I["B_discretion"]["pooled"]["beta"]
icpt = pooled - M["discretion"]["slope"]*float(np.mean(cc.discretion))
b.plot(xs, icpt + M["discretion"]["slope"]*xs*(1-xs), color=VERM, lw=1.3, zorder=4)
for lab, key, col in [("prevalence >= 0.80","prevalence >= 0.80",GREY),
                      ("prevalence 0.30-0.70","prevalence 0.30-0.70",GREEN),
                      ("prevalence <= 0.20","prevalence <= 0.20",GREY)]:
    if key in M:
        v = M[key]
        print(f"  {lab}: k={v['k']} beta={v['beta']:+.4f}", flush=True)
b.set_xlabel("Share of schools in the country reporting a ban")
b.set_ylabel("Association with endorsement, scale points")
b.set_title("Largest where schools were most evenly divided", pad=5)
b.annotate(f"slope on discretion {M['discretion']['slope']:+.3f}\n"
           f"(SE {M['discretion']['se']:.3f}, p = {M['discretion']['p']:.3f})",
           xy=(0.03, 0.03), xycoords="axes fraction", ha="left", va="bottom", fontsize=7,
           bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
hi = M.get("prevalence >= 0.80")
if hi:
    b.axvspan(0.80, 1.02, color=GREY, alpha=0.12, zorder=0)
    b.annotate(f"almost every school bans\n{hi['k']} countries, mean {hi['beta']:+.3f}",
               xy=(0.91, b.get_ylim()[1]), ha="center", va="top", fontsize=6.6, color="#5A5A5A",
               bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    b.set_xlim(right=1.02)
clean(b)
fig.tight_layout(pad=0.5, w_pad=1.8)
fig.savefig("figs/B_Figure4.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)
print("B_Figure4 done", flush=True)
import subprocess
for f in ("B_Figure3.png","B_Figure4.png"):
    print(subprocess.run(["python3","-c",
      f"from PIL import Image;im=Image.open('figs/{f}');print('{f}',im.size,'-> %.0f dpi at 17cm'%(im.size[0]/(17/2.54)))"],
      capture_output=True,text=True).stdout.strip(), flush=True)
print("DONE", flush=True)
