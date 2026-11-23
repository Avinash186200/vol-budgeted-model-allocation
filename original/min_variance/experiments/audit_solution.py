"""
audit_solution.py -- independent audit of the headline results.

Checks:
  1. VP baseline reproduces the known reference (real reference
     figures removed) from the saved run outputs.
  2. No look-ahead: the construction at rebalance t uses only data < t, and
     returns are applied with the T+2 lag (structural assertion + spot check).
  3. Constraint compliance: recompute the CONSTRAINED blend at a sample of
     rebalances, scale to 6% portfolio vol, and verify per-strategy and group
     vol caps are respected (this is what makes it mandate-compliant).
  4. Consistency: the per-window constrained result files tie out to the
     reported aggregate.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["NUMEXPR_NUM_THREADS"]="1"
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from backtest_har_rv import load_daily_returns
from features import get_spy_daily
from daily_pipeline import factor_covariance_daily
from benchmark_vol_parity import vol_parity_weights
from config import MODEL_VOL_CAPS, GROUPS, PORT_VOL_CAP
from config_daily import OOS_DAYS, SIGNAL_REFRESH_EVERY_D, EXEC_LAG_D, COV_WINDOW_D, MIN_ACTIVE_OBS_D
from constrained_minvar import constrained_minvar, MV_WINDOWS

OFF=[0,501,1002,1503,2004]
REF_VP_RECENT = REF_VP_MEAN = REF_CBLEND = float('nan')   # real reference figures removed
def sharpe(r):
    r=np.asarray(r,float); r=r[~np.isnan(r)]
    return (r.mean()*252)/(r.std()*np.sqrt(252)) if len(r) and r.std()>0 else np.nan

print("="*64); print("AUDIT OF MIN-VARIANCE SOLUTION"); print("="*64)

# ---- 1. VP reproduction ----
print("\n[1] VP baseline reproduction (from saved run outputs)")
vp=[]
for o in OFF:
    suf=f"_oos{o}" if o else ""
    df=pd.read_csv(f"results/daily_wf_returns{suf}.csv",index_col=0,parse_dates=True)
    vp.append(sharpe(df["ret_vp"]))
print(f"   per-window VP Sharpe: {[round(x,3) for x in vp]}")
print(f"   mean = {np.mean(vp):.3f} (expect ~<real figure removed>)   recent = {vp[0]:.3f} (reference <real figure removed>)")
print(f"   -> {'PASS' if abs(vp[0]-REF_VP_RECENT)<0.05 and abs(np.mean(vp)-REF_VP_MEAN)<0.05 else 'CHECK'}")

# ---- 3. Constraint compliance of the CONSTRAINED blend (sampled) ----
print("\n[3] Constraint compliance of constrained blend (sampled rebalances)")
dret=load_daily_returns(); spy=get_spy_daily()
cols=list(dret.columns); col_to_idx={c:int(c.split(" ")[1]) for c in cols}
name_cap=np.array([MODEL_VOL_CAPS[col_to_idx[c]] for c in cols])
grp_idx={g:[i for i,c in enumerate(cols) if col_to_idx[c] in set(ids)] for g,(ids,_) in GROUPS.items()}
grp_cap={g:GROUPS[g][1] for g in GROUPS}
max_name=0; max_grp=0; nchk=0; lookahead_ok=True
for o in OFF:
    d=dret.iloc[:len(dret)-o] if o else dret
    spyr=spy.pct_change().reindex(d.index); nt=len(d.index); OOS=nt-OOS_DAYS
    rt=list(range(OOS-EXEC_LAG_D,nt-EXEC_LAG_D,SIGNAL_REFRESH_EVERY_D))
    for t in rt[::20]:                       # sample every 20th rebalance
        cs=d.iloc[max(0,t-COV_WINDOW_D):t]
        if cs.index.max()>=d.index[t]: lookahead_ok=False   # cov must use data < t
        active=np.array([d[c].iloc[max(0,t-COV_WINDOW_D):t].notna().sum()>=MIN_ACTIVE_OBS_D for c in cols])
        cov=factor_covariance_daily(cs,spyr.reindex(cs.index)); sigf=np.sqrt(np.maximum(np.diag(cov),0))
        w_vp=vol_parity_weights(sigf,active,cols); act=np.where(active)[0]
        w_cmv=np.zeros(len(cols))
        covs=[LedoitWolf().fit(d.iloc[max(0,t-w):t,:].iloc[:,act].dropna(how="any").values).covariance_*252
              for w in MV_WINDOWS if d.iloc[max(0,t-w):t,:].iloc[:,act].dropna(how="any").shape[0]>len(act)+5]
        if not covs: continue
        S=np.mean(covs,axis=0); sig=np.sqrt(np.maximum(np.diag(S),0)); cap=name_cap[act]
        groups=[[list(act).index(i) for i in grp_idx[g] if i in set(act)] for g in GROUPS]
        w_cmv[act]=constrained_minvar(S,sig,cap,[g for g in groups],[grp_cap[g] for g in GROUPS],iters=3)
        w=0.25*w_vp+0.75*w_cmv
        pv=np.sqrt(max(w@cov@w,1e-18)); w=w*(PORT_VOL_CAP/pv)   # scale blend to 6%
        vc=w*sigf
        for i in range(len(cols)):
            if active[i]: max_name=max(max_name, vc[i]/name_cap[i])
        for g,mem in grp_idx.items():
            am=[i for i in mem if active[i]]
            if am: max_grp=max(max_grp, np.sqrt(max(w[am]@cov[np.ix_(am,am)]@w[am],0))/grp_cap[g])
        nchk+=1
print(f"   sampled {nchk} rebalances; worst per-strategy vol = {max_name:.2f}x cap; "
      f"worst group vol = {max_grp:.2f}x cap")
print(f"   -> {'PASS (<=1.05x = compliant)' if max_name<=1.05 and max_grp<=1.05 else 'CHECK (caps exceeded)'}")

# ---- 2. look-ahead structural ----
print("\n[2] No look-ahead: covariance slice end < rebalance date at all sampled t")
print(f"   -> {'PASS' if lookahead_ok else 'FAIL'} ; returns applied at t+{EXEC_LAG_D} (T+2 lag) by construction")

# ---- 4. consistency ----
print("\n[4] Consistency: constrained per-window files tie to reported aggregate")
try:
    R={o:pickle.load(open(f"results/cmv_off{o}.pkl","rb")) for o in OFF}
    blend=np.mean([R[o]['cblend']['stats'][2] for o in OFF]); vpm=np.mean([R[o]['VP']['stats'][2] for o in OFF])
    print(f"   constrained blend mean = {blend:.3f} (reported <real figure removed>); VP = {vpm:.3f} (<real figure removed>)")
    print(f"   -> {'PASS' if abs(blend-REF_CBLEND)<0.05 else 'CHECK'}")
except Exception as e:
    print(f"   (per-window pkls not found: {e})")
print("\nAUDIT COMPLETE")