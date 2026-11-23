"""
bootstrap_minvar.py -- significance test for the min-variance ENSEMBLE and the
75% blend vs VP. Recomputes the three daily return streams across all 5 windows,
then reports:
  (1) per-window Sharpe + how many of 5 windows each beats VP (consistency),
  (2) paired stationary block-bootstrap (10k, mean block 21d) on the SHARPE
      DIFFERENCE, pooled daily -> p(diff<=0) and 95% CI.
Honest framing: the daily bootstrap is optimistic (true independent units are
the ~5 regimes), so we report BOTH the bootstrap p AND the 5-window win rate.
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

OFF=[0,501,1002,1503,2004]; MV_WINDOWS=[252,504,756]

def sharpe(r):
    r=np.asarray(r,float); r=r[~np.isnan(r)]
    return (r.mean()*252)/(r.std()*np.sqrt(252)) if len(r) and r.std()>0 else np.nan

def w_minvar(S):
    n=S.shape[0]; x0=(1/np.clip(np.diag(S),1e-12,None)); x0/=x0.sum()
    r=minimize(lambda w:w@S@w,x0,method="SLSQP",bounds=[(0,1)]*n,
               constraints=({"type":"eq","fun":lambda w:w.sum()-1},),options={"maxiter":200,"ftol":1e-11})
    return r.x/r.x.sum() if r.success and r.x.sum()>0 else x0

def streams():
    dret=load_daily_returns(); spy=get_spy_daily()
    out={"VP":{}, "mv_ens":{}, "blend075":{}}
    for o in OFF:
        d=dret.iloc[:len(dret)-o] if o else dret
        spyr=spy.pct_change().reindex(d.index); cols=list(d.columns); nt=len(d.index)
        OOS=nt-OOS_DAYS; rt=list(range(OOS-EXEC_LAG_D,nt-EXEC_LAG_D,SIGNAL_REFRESH_EVERY_D))
        rv={"VP":[],"mv_ens":[],"blend075":[]}
        for t in rt:
            active=np.array([d[c].iloc[max(0,t-COV_WINDOW_D):t].notna().sum()>=MIN_ACTIVE_OBS_D for c in cols])
            cs=d.iloc[max(0,t-COV_WINDOW_D):t]
            cov=factor_covariance_daily(cs,spyr.reindex(cs.index)); sig=np.sqrt(np.maximum(np.diag(cov),0))
            w_vp=vol_parity_weights(sig,active,cols); act=np.where(active)[0]
            mvs=[]
            for win in MV_WINDOWS:
                w=np.zeros(len(cols)); R=d.iloc[max(0,t-win):t,:].iloc[:,act].dropna(how="any").values
                if R.shape[0]>R.shape[1]+5:
                    S=LedoitWolf().fit(R).covariance_*252.0; w[act]=w_minvar(S)
                mvs.append(w)
            mv_ens=np.mean(mvs,axis=0); blend=0.25*w_vp+0.75*mv_ens
            es=t+EXEC_LAG_D; ee=min(es+SIGNAL_REFRESH_EVERY_D,nt)
            for dd in range(es,ee):
                if dd<OOS: continue
                rr=d.iloc[dd].fillna(0).values
                rv["VP"].append(float(w_vp@rr)); rv["mv_ens"].append(float(mv_ens@rr)); rv["blend075"].append(float(blend@rr))
        for k in out: out[k][o]=np.array(rv[k])
    return out

def block_boot_diff(a,b,n=10000,mb=21):
    m=~(np.isnan(a)|np.isnan(b)); a,b=a[m],b[m]; N=len(a); p=1/mb
    rng=np.random.RandomState(2027); diffs=np.empty(n)
    for k in range(n):
        idx=np.empty(N,dtype=int); i=0
        while i<N:
            st=rng.randint(0,N); L=max(1,rng.geometric(p))
            for j in range(L):
                if i>=N: break
                idx[i]=(st+j)%N; i+=1
        diffs[k]=sharpe(a[idx])-sharpe(b[idx])
    return (diffs<=0).mean(), np.percentile(diffs,2.5), np.percentile(diffs,97.5), diffs.mean()

def main():
    S=streams()
    for name in ["mv_ens","blend075"]:
        print(f"\n===== {name} vs VP =====")
        pw=[];
        for o in OFF:
            sv=sharpe(S["VP"][o]); sc=sharpe(S[name][o]); pw.append((o,sv,sc))
        wins=sum(1 for _,sv,sc in pw if sc>sv)
        print("  per-window VP / cand / win:")
        for o,sv,sc in pw: print(f"    off={o:<5d} {sv:6.3f} / {sc:6.3f}  {'WIN' if sc>sv else 'lose'}")
        print(f"  beats VP in {wins}/5 windows;  mean Sharpe edge = {np.mean([sc-sv for _,sv,sc in pw]):+.3f}")
        a=np.concatenate([S[name][o] for o in OFF]); b=np.concatenate([S["VP"][o] for o in OFF])
        p,lo,hi,mn=block_boot_diff(a,b)
        print(f"  block-bootstrap Sharpe-diff: mean {mn:+.3f}  p(diff<=0)={p:.4f}  95%CI[{lo:+.3f},{hi:+.3f}]")
        print(f"  -> {'significant at 5%' if p<0.05 else 'NOT significant at 5%'} (daily bootstrap; small-regime caveat applies)")

if __name__=="__main__":
    main()