"""
net_cost_experiment.py -- net-of-transaction-cost / turnover analysis.

All Sharpes elsewhere are GROSS. This computes per-variant turnover and NET
Sharpe at several costs, per-window then averaged (same basis as gross).

  turnover_ann = sum_t sum_i |w_i,t - w_i,t-1| / years   (NAV traded per year)
  net_sharpe(c) per window = (gross_ann_ret - turnover_ann*c) / gross_vol
  reported = mean of per-window net Sharpe.

Part 1: production variants from saved weights (VP, Variant E, optimiser QP, optimiser scaled).
Part 2: recompute min-variance (lw504, our gross lead) + VP fresh, to test whether
        min-var's gross edge survives trading costs.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from backtest_har_rv import load_daily_returns
from features import get_spy_daily
from daily_pipeline import factor_covariance_daily
from benchmark_vol_parity import vol_parity_weights
from config_daily import OOS_DAYS, SIGNAL_REFRESH_EVERY_D, EXEC_LAG_D, COV_WINDOW_D, MIN_ACTIVE_OBS_D

OFF=[0,501,1002,1503,2004]
COSTS=[1,2,5,10]
VARIANTS=[("volparity","ret_vp","VP"),("variant_e","ret_variant_e","Variant E"),
          ("qp","ret_qp","optimiser QP-only"),("qp_scaled","ret_qp_scaled","optimiser 756-scaled")]

def gstats(r):
    r=r.dropna().values; ar=r.mean()*252; av=r.std()*np.sqrt(252); return ar,av,(ar/av if av>0 else np.nan)

def per_window_net(g_ar,g_av,turn_ann):
    return {b: ((g_ar-turn_ann*(b/10000.0))/g_av if g_av>0 else np.nan) for b in COSTS}

def part1():
    print("=== PART 1: production variants (from saved weights) ===")
    print(f"{'Variant':<16s} {'GrossSh':>7s} {'Turn/yr':>7s}  " + "  ".join(f'net{b}bp' for b in COSTS))
    for stem,retcol,label in VARIANTS:
        gsh=[]; nets={b:[] for b in COSTS}; turns=[]
        for o in OFF:
            suf=f"_oos{o}" if o else ""
            ret=pd.read_csv(f"results/daily_wf_returns{suf}.csv",index_col=0,parse_dates=True)
            W=pd.read_csv(f"results/daily_wf_weights_{stem}{suf}.csv",index_col=0).fillna(0.0)
            ar,av,sh=gstats(ret[retcol]); yrs=len(ret)/252.0
            t=W.diff().abs().sum(axis=1).iloc[1:].sum()/yrs
            gsh.append(sh); turns.append(t)
            nw=per_window_net(ar,av,t)
            for b in COSTS: nets[b].append(nw[b])
        print(f"{label:<16s} {np.mean(gsh):7.3f} {np.mean(turns):7.1f}  " +
              "  ".join(f"{np.mean(nets[b]):6.3f}" for b in COSTS))

def w_minvar(S):
    n=S.shape[0]
    r=minimize(lambda w:w@S@w, (1/np.clip(np.diag(S),1e-12,None))/np.sum(1/np.clip(np.diag(S),1e-12,None)),
               method="SLSQP",bounds=[(0,1)]*n,constraints=({"type":"eq","fun":lambda w:w.sum()-1},),
               options={"maxiter":200,"ftol":1e-11})
    return r.x/r.x.sum() if r.success and r.x.sum()>0 else None

def part2():
    print("\n=== PART 2: min-variance (lw504) vs VP, recomputed with turnover ===")
    dret=load_daily_returns(); spy=get_spy_daily()
    print(f"{'metric':<18s} {'VP':>8s} {'MinVar':>8s}")
    gsh_vp=[]; gsh_mv=[]; t_vp=[]; t_mv=[]; ar_vp=[];av_vp=[];ar_mv=[];av_mv=[]
    for o in OFF:
        d=dret.iloc[:len(dret)-o] if o else dret
        spyr=spy.pct_change().reindex(d.index); cols=list(d.columns); nt=len(d.index)
        OOS=nt-OOS_DAYS; rt=list(range(OOS-EXEC_LAG_D,nt-EXEC_LAG_D,SIGNAL_REFRESH_EVERY_D))
        wprev_vp=None; wprev_mv=None; tv=0;tm=0; rvp=[];rmv=[]
        for t in rt:
            active=np.array([d[c].iloc[max(0,t-COV_WINDOW_D):t].notna().sum()>=MIN_ACTIVE_OBS_D for c in cols])
            cs=d.iloc[max(0,t-COV_WINDOW_D):t]
            cov=factor_covariance_daily(cs,spyr.reindex(cs.index)); sig=np.sqrt(np.maximum(np.diag(cov),0))
            wvp=vol_parity_weights(sig,active,cols)
            act=np.where(active)[0]; wmv=np.zeros(len(cols))
            R=d.iloc[max(0,t-504):t,:].iloc[:,act].dropna(how="any").values
            if R.shape[0]>R.shape[1]+5:
                S=LedoitWolf().fit(R).covariance_*252.0; w=w_minvar(S)
                if w is not None: wmv[act]=w
            if wprev_vp is not None: tv+=np.abs(wvp-wprev_vp).sum()
            if wprev_mv is not None: tm+=np.abs(wmv-wprev_mv).sum()
            wprev_vp=wvp; wprev_mv=wmv
            es=t+EXEC_LAG_D; ee=min(es+SIGNAL_REFRESH_EVERY_D,nt)
            for dd in range(es,ee):
                if dd<OOS: continue
                r=d.iloc[dd].fillna(0).values; rvp.append(wvp@r); rmv.append(wmv@r)
        yrs=len(rvp)/252.0
        a,v,s=gstats(pd.Series(rvp)); gsh_vp.append(s); ar_vp.append(a);av_vp.append(v); t_vp.append(tv/yrs)
        a,v,s=gstats(pd.Series(rmv)); gsh_mv.append(s); ar_mv.append(a);av_mv.append(v); t_mv.append(tm/yrs)
    print(f"{'gross Sharpe':<18s} {np.mean(gsh_vp):8.3f} {np.mean(gsh_mv):8.3f}")
    print(f"{'turnover/yr':<18s} {np.mean(t_vp):8.1f} {np.mean(t_mv):8.1f}")
    for b in COSTS:
        nv=np.mean([(ar_vp[i]-t_vp[i]*b/10000)/av_vp[i] for i in range(len(OFF))])
        nm=np.mean([(ar_mv[i]-t_mv[i]*b/10000)/av_mv[i] for i in range(len(OFF))])
        print(f"{'net Sharpe @'+str(b)+'bp':<18s} {nv:8.3f} {nm:8.3f}   (min-var edge {nm-nv:+.3f})")

if __name__=="__main__":
    part1(); part2()