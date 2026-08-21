#!/usr/bin/env python
"""Generate all paper figures directly from results/wm/*.json (single source of truth)."""
import glob, json, math, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=f'{REPO}/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,
    'ytick.labelsize':7,'legend.fontsize':7,'figure.dpi':300,'savefig.dpi':300,
    'axes.spines.top':False,'axes.spines.right':False,'font.family':'serif',
    'mathtext.fontset':'dejavuserif','axes.grid':True,'grid.alpha':.25,'grid.linewidth':.4,
    'savefig.bbox':'tight','savefig.pad_inches':0.01})
C={'ours':'#B2182B','base':'#2166AC','bk':'#4D4D4D','alt':'#1B7837','warn':'#E08214'}

def vals(t,ds,m='NDCG@20'):
    fs=glob.glob(f'{REPO}/results/wm/{t}__{ds}__seed*.json')
    return [json.load(open(f))['metrics'][m] for f in fs if m in json.load(open(f))['metrics']]
def ms(t,ds,m='NDCG@20'):
    v=vals(t,ds,m)
    if not v: return None
    mu=sum(v)/len(v); sd=math.sqrt(sum((x-mu)**2 for x in v)/(len(v)-1)) if len(v)>1 else 0.
    return mu,sd

# ---- Fig 2: geom dose-response (single panel; density panel removed as unsupported) ----
def fig_dose_density():
    import numpy as np
    fig,ax=plt.subplots(figsize=(3.3,2.1))
    ws=[0,1,2]; tags=['XSimGCLg_w00','XSimGCLg_w10','XSimGCLg_w20']  # current harness, 3 seeds each
    for ds,col,lab in [('douban-book',C['ours'],'Douban-Book (sparse)'),('ml-1M',C['alt'],'ML-1M (dense)')]:
        pts=[(w,ms(t,ds)) for w,t in zip(ws,tags) if ms(t,ds)]
        if len(pts)<2: continue
        x=[p[0] for p in pts]; y=[p[1][0] for p in pts]; y0=y[0]
        ax.plot(x,[100*(v-y0)/y0 for v in y],marker='o',ms=3.5,lw=1.3,color=col,label=lab)
    ax.axhline(0,color='k',lw=.7)
    ax.set_xlabel(r'uniform geometric weight $\mu_g$')
    ax.set_ylabel(r'$\Delta$NDCG@20 vs.\ encoder (%)')
    ax.legend(frameon=False,fontsize=6.5,loc='center left')
    fig.savefig(f'{OUT}/dose_density.pdf'); plt.close(fig); print('dose_density.pdf (single panel)')

# ---- Fig 3: inert knobs ----------------------------------------------------
def fig_sensitivity():
    S=[(r'$\mu_{\mathrm{sp}}$ (L2-SP)',[('0',"SW_sp0p0"),('0.002','SW_sp0p002'),('0.02','SW_sp0p02'),('0.2','SW_sp0p2'),('1.0','SW_sp1p0')]),
       (r'$\mu_u$ (uniformity)',[('0','SW_u0p0'),('0.001','SW_u0p001'),('0.01','SW_u0p01'),('0.1','SW_u0p1'),('1.0','SW_u1p0')]),
       (r'$\gamma$ (pop.\ debias)',[('0.1','SW_g0p1'),('0.3','SW_g0p3'),('0.5','SW_g0p5')]),
       ('warm-up epochs',[(v,f'SW_wu{v}') for v in ['0','5','10','20','30']])]
    fig,axes=plt.subplots(1,4,figsize=(7.0,1.75),sharey=True)
    for ax,(name,items) in zip(axes,S):
        xs=[];ys=[]
        for lb,t in items:
            m=ms(t,'douban-book')
            if m: xs.append(lb); ys.append(m[0])
        ax.plot(range(len(xs)),ys,marker='o',ms=3,lw=1.1,color=C['ours'])
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs,rotation=45,ha='right')
        ax.set_title(name); ax.set_ylim(.145,.157)
    axes[0].set_ylabel('NDCG@20')
    fig.savefig(f'{OUT}/sensitivity.pdf'); plt.close(fig); print('sensitivity.pdf')

# ---- Fig 4: accuracy vs exposure -------------------------------------------
def fig_exposure():
    V=[('PT4Rec','PTbase_XSim_nofz',C['base']),('+routing/pop.','OURS_XSim_nofz',C['warn']),
       ('+geometry','OURSgeom_w2',C['ours']),('XSimGCL','XSimGCLg_w00',C['bk']),('DirectAU','DirectAU',C['alt'])]
    fig,axes=plt.subplots(1,3,figsize=(7.0,1.9))
    for ax,(met,lab,inv) in zip(axes,[('ARP@20','avg. rec.\\ popularity $\\downarrow$',1),
                                      ('Gini@20','Gini of exposure $\\downarrow$',1),
                                      ('ItemCoverage@20','item coverage $\\uparrow$',0)]):
        names=[];v=[];cs=[]
        for n,t,c in V:
            m=ms(t,'douban-book',met)
            if m: names.append(n); v.append(m[0]); cs.append(c)
        ax.bar(range(len(v)),v,color=cs,edgecolor='k',linewidth=.4)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=30,ha='right',fontsize=6)
        ax.set_ylabel(lab,fontsize=7)
    fig.savefig(f'{OUT}/exposure.pdf'); plt.close(fig); print('exposure.pdf')

# ---- Fig 5: embedding spectrum ---------------------------------------------
def fig_spectrum():
    def geo(t,key='user_sv_top20'):
        for f in glob.glob(f'{REPO}/results/wm/{t}__douban-book__seed*.json'):
            d=json.load(open(f))
            if d.get('geometry',{}).get(key): return d['geometry']
        return None
    V=[('XSimGCL (backbone)','XSimGCLg_w00',C['bk']),('PT4Rec','PTbase_XSim_nofz',C['base']),
       ('OURS','OURS_XSim_nofz',C['warn']),('OURS-G','OURSgeom_w2',C['ours'])]
    got=[(n,geo(t),c) for n,t,c in V]; got=[g for g in got if g[1]]
    if not got: print('spectrum: no geometry data yet'); return
    fig,ax=plt.subplots(figsize=(3.3,2.0))
    for n,g,c in got:
        sv=np.array(g['user_sv_top20']); sv=sv/sv[0]
        ax.plot(range(1,len(sv)+1),sv,marker='o',ms=2.5,lw=1.1,color=c,
                label=f"{n} (eff.\\ rank {g['user_eff_rank']:.1f})")
    ax.set_yscale('log'); ax.set_xlabel('singular value index'); ax.set_ylabel('normalized singular value')
    ax.legend(frameon=False,fontsize=6)
    fig.savefig(f'{OUT}/spectrum.pdf'); plt.close(fig); print('spectrum.pdf')

for f in (fig_dose_density,fig_sensitivity,fig_exposure,fig_spectrum):
    try: f()
    except Exception as e: print(f.__name__,'FAILED',e)
