"""
audit_minvar_constraints.py -- does the min-variance solution respect the
mandate constraints?

Recomputes the min-var ENSEMBLE and the 75/25 blend each rebalance, scales to
the 6% PORTFOLIO vol cap, then audits against:
  - per-strategy vol cap   (w_i * sigma_i <= cap_i,  ~1%)
  - group vol cap          (sqrt(w_g' Cov_g w_g) <= group_cap)
  - portfolio vol cap      (scaled to 6% by construction)
Reports how often / how badly each binds. If they're mostly slack, the
edge stands as-is; if they bind hard, we need the constrained optimizer.
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
from config import MODEL_VOL_CAPS, GROUPS, PORT_VOL_CAP
from config_daily import OOS_DAYS, SIGNAL_REFRESH_EVERY_D, EXEC_LAG_D, COV_WINDOW_D, MIN_ACTIVE_OBS_D

OFF=[0,501,1002,1503,2004]; MV_WINDOWS=[252,504,756]

def w_minvar(S):
    n=S.shape[0]; x0=(1/np.clip(np.diag(S),1e-12,None)); x0/=x0.sum()
    r=minimize(lambda w:w@S@w,x0,method="SLSQP",bounds=[(0,1)]*n,
               constraints=({"type":"eq","fun":lambda w:w.sum()-1},),options={"maxiter":200,"ftol":1e-11})
    return r.x/r.x.sum() if r.success and r.x.sum()>0 else x0

def main():
    dret=load_daily_returns(); spy=get_spy_daily()
    cols=list(dret.columns); col_to_idx={c:int(c.split(" ")[1]) for c in cols}
    # per-name caps and group memberships (by model index)
    name_cap=np.array([MODEL_VOL_CAPS[col_to_idx[c]] for c in cols])
    grp_members={g:[i for i,c in enumerate(cols) if col_to_idx[c] in set(ids)] for g,(ids,_) in GROUPS.items()}
    grp_cap={g:GROUPS[g][1] for g in GROUPS}

    for variant in ["mv_ens","blend075"]:
        pername_viol=0; pername_tot=0; max_pername_ratio=0
        grp_viol=0; grp_tot=0; max_grp_ratio=0; portvols=[]
        for o in OFF:
            d=dret.iloc[:len(dret)-o] if o else dret
            spyr=spy.pct_change().reindex(d.index); nt=len(d.index)
            OOS=nt-OOS_DAYS; rt=list(range(OOS-EXEC_LAG_D,nt-EXEC_LAG_D,SIGNAL_REFRESH_EVERY_D))
            for t in rt:
                active=np.array([d[c].iloc[max(0,t-COV_WINDOW_D):t].notna().sum()>=MIN_ACTIVE_OBS_D for c in cols])
                cs=d.iloc[max(0,t-COV_WINDOW_D):t]; cov=factor_covariance_daily(cs,spyr.reindex(cs.index))
                sig=np.sqrt(np.maximum(np.diag(cov),0)); w_vp=vol_parity_weights(sig,active,cols)
                act=np.where(active)[0]; mvs=[]
                for win in MV_WINDOWS:
                    w=np.zeros(len(cols)); R=d.iloc[max(0,t-win):t,:].iloc[:,act].dropna(how="any").values
                    if R.shape[0]>R.shape[1]+5:
                        S=LedoitWolf().fit(R).covariance_*252.0; w[act]=w_minvar(S)
                    mvs.append(w)
                mv_ens=np.mean(mvs,axis=0)
                w=mv_ens if variant=="mv_ens" else 0.25*w_vp+0.75*mv_ens
                pv=np.sqrt(max(w@cov@w,1e-18))
                if pv<=0: continue
                w=w*(PORT_VOL_CAP/pv)                     # scale to 6% portfolio vol
                portvols.append(np.sqrt(w@cov@w))
                vc=w*sig                                   # per-name vol contribution
                for i in range(len(cols)):
                    if active[i]:
                        pername_tot+=1
                        if vc[i]>name_cap[i]+1e-9: pername_viol+=1
                        max_pername_ratio=max(max_pername_ratio, vc[i]/name_cap[i])
                for g,mem in grp_members.items():
                    am=[i for i in mem if active[i]]
                    if not am: continue
                    gv=np.sqrt(max(w[am]@cov[np.ix_(am,am)]@w[am],0)); grp_tot+=1
                    if gv>grp_cap[g]+1e-9: grp_viol+=1
                    max_grp_ratio=max(max_grp_ratio, gv/grp_cap[g])
        print(f"\n===== {variant} (scaled to 6% portfolio vol) =====")
        print(f"  portfolio vol: mean {np.mean(portvols)*100:.2f}% (target 6.00%)")
        print(f"  per-strategy cap: {pername_viol}/{pername_tot} name-rebalances exceed cap "
              f"({100*pername_viol/max(pername_tot,1):.1f}%); worst = {max_pername_ratio:.2f}x cap")
        print(f"  group cap:        {grp_viol}/{grp_tot} group-rebalances exceed cap "
              f"({100*grp_viol/max(grp_tot,1):.1f}%); worst = {max_grp_ratio:.2f}x cap")

if __name__=="__main__":
    main()