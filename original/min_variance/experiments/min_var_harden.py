"""
min_var_harden.py -- try to convert the min-variance lead from "promising but
unproven" into something steadier and recent-window-safe.

#1 cov-window ensemble : average min-var weights over cov windows {252,504,756}
   each rebalance (damps covariance estimation error).
#2 VP<->min-var blend  : capital-allocation blend (1-a)*VP + a*minvar_ensemble
   for a in {0.25,0.5,0.75} (mostly VP with a regularized min-var tilt).

Reports per-variant: 5-period mean gross Sharpe, RECENT-window Sharpe (off=0,
the weak spot), turnover/yr, and net Sharpe @2/5bp. Weight-level blend so
turnover/net are consistent.
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

OFF=[0,501,1002,1503,2004]; COSTS=[2,5]; MV_WINDOWS=[252,504,756]

def gstats(r):
    r=np.asarray(r,float); r=r[~np.isnan(r)]; ar=r.mean()*252; av=r.std()*np.sqrt(252)
    return ar,av,(ar/av if av>0 else np.nan)

def w_minvar(S):
    n=S.shape[0]; x0=(1/np.clip(np.diag(S),1e-12,None)); x0/=x0.sum()
    r=minimize(lambda w:w@S@w,x0,method="SLSQP",bounds=[(0,1)]*n,
               constraints=({"type":"eq","fun":lambda w:w.sum()-1},),options={"maxiter":200,"ftol":1e-11})
    return r.x/r.x.sum() if r.success and r.x.sum()>0 else x0

def run():
    dret=load_daily_returns(); spy=get_spy_daily()
    # per-variant per-window: returns list + turnover
    names=["VP","mv252","mv504","mv756","mv_ens","blend0.25","blend0.50","blend0.75"]
    G={n:{} for n in names}     # name -> off -> (ann_ret, ann_vol, sharpe)
    T={n:{} for n in names}     # name -> off -> turnover/yr
    for o in OFF:
        d=dret.iloc[:len(dret)-o] if o else dret
        spyr=spy.pct_change().reindex(d.index); cols=list(d.columns); nt=len(d.index)
        OOS=nt-OOS_DAYS; rt=list(range(OOS-EXEC_LAG_D,nt-EXEC_LAG_D,SIGNAL_REFRESH_EVERY_D))
        prev={n:None for n in names}; turn={n:0.0 for n in names}; ret={n:[] for n in names}
        for t in rt:
            active=np.array([d[c].iloc[max(0,t-COV_WINDOW_D):t].notna().sum()>=MIN_ACTIVE_OBS_D for c in cols])
            cs=d.iloc[max(0,t-COV_WINDOW_D):t]
            cov=factor_covariance_daily(cs,spyr.reindex(cs.index)); sig=np.sqrt(np.maximum(np.diag(cov),0))
            w_vp=vol_parity_weights(sig,active,cols)
            act=np.where(active)[0]
            mv={}
            for win in MV_WINDOWS:
                w=np.zeros(len(cols))
                R=d.iloc[max(0,t-win):t,:].iloc[:,act].dropna(how="any").values
                if R.shape[0]>R.shape[1]+5:
                    S=LedoitWolf().fit(R).covariance_*252.0; w[act]=w_minvar(S)
                mv[win]=w
            mv_ens=np.mean([mv[w] for w in MV_WINDOWS],axis=0)
            W={"VP":w_vp,"mv252":mv[252],"mv504":mv[504],"mv756":mv[756],"mv_ens":mv_ens,
               "blend0.25":0.75*w_vp+0.25*mv_ens,"blend0.50":0.5*w_vp+0.5*mv_ens,"blend0.75":0.25*w_vp+0.75*mv_ens}
            for n in names:
                if prev[n] is not None: turn[n]+=np.abs(W[n]-prev[n]).sum()
                prev[n]=W[n]
            es=t+EXEC_LAG_D; ee=min(es+SIGNAL_REFRESH_EVERY_D,nt)
            for dd in range(es,ee):
                if dd<OOS: continue
                rr=d.iloc[dd].fillna(0).values
                for n in names: ret[n].append(float(W[n]@rr))
        yrs=len(ret["VP"])/252.0
        for n in names:
            G[n][o]=gstats(pd.Series(ret[n])); T[n][o]=turn[n]/yrs

    print(f"{'variant':<11s} {'grossSh':>7s} {'recent':>7s} {'turn/yr':>7s} {'net2bp':>7s} {'net5bp':>7s}")
    for n in names:
        gmean=np.mean([G[n][o][2] for o in OFF])
        recent=G[n][0][2]
        tmean=np.mean([T[n][o] for o in OFF])
        nets={}
        for b in COSTS:
            nets[b]=np.mean([(G[n][o][0]-T[n][o]*b/10000)/G[n][o][1] if G[n][o][1]>0 else np.nan for o in OFF])
        print(f"{n:<11s} {gmean:7.3f} {recent:7.3f} {tmean:7.1f} {nets[2]:7.3f} {nets[5]:7.3f}")
    print("\nVP recent=<real figure removed> is the bar to keep; mv_ens/blends aim to hold the +edge while not losing recent.")

if __name__=="__main__":
    run()