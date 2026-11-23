"""
constrained_minvar.py -- min-variance ensemble RE-SOLVED under the full
mandate (per-strategy vol caps + group vol caps + 6% portfolio cap), to see if
the edge survives the real constraints.

Per rebalance, using the cov-window-ensemble shrunk covariance S:
  minimize w' S w
  s.t. sum(w)=1, 0<=w_i<=cap_i/(sigma_i*k), group: sqrt(w_g' S_g w_g)<=gcap_g/k
  iterate k = 0.06/portvol so the caps bind at the live 6%-scaled portfolio.
Then realize returns; also the 75/25 blend with VP (VP respects per-name caps).
Compare gross Sharpe (5-period mean + recent) and turnover vs VP.
NOTE: group cap modeled as group sub-portfolio vol <= group cap (correlation-aware).
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

def gstats(r):
    r=np.asarray(r,float); r=r[~np.isnan(r)]; ar=r.mean()*252; av=r.std()*np.sqrt(252)
    return ar,av,(ar/av if av>0 else np.nan)

def constrained_minvar(S, sig, cap, groups, gcaps, port_cap=0.06, iters=4):
    """min-variance over active subset under per-name vol caps, group vol caps,
    6% portfolio cap. Returns weights scaled so portfolio vol = port_cap."""
    n=S.shape[0]; k=1.0; w=None
    base=(1/np.clip(np.diag(S),1e-12,None)); base/=base.sum()
    for _ in range(iters):
        ub=np.minimum(cap/(np.clip(sig,1e-9,None)*k),1.0)
        cons=[{"type":"eq","fun":lambda w:w.sum()-1.0}]
        for gi,members in enumerate(groups):
            if members:
                gc=gcaps[gi]/k
                cons.append({"type":"ineq","fun":(lambda w,m=members,g=gc: g*g - w[m]@S[np.ix_(m,m)]@w[m])})
        x0=np.minimum(base,ub);
        if x0.sum()>0: x0/=x0.sum()
        else: x0=np.ones(n)/n
        r=minimize(lambda w:w@S@w,x0,method="SLSQP",bounds=[(0,float(u)) for u in ub],
                   constraints=cons,options={"maxiter":150,"ftol":1e-10})
        w=r.x if r.success else x0
        w=np.clip(w,0,None);
        if w.sum()>0: w/=w.sum()
        pv=np.sqrt(max(w@S@w,1e-18)); k=port_cap/pv
    return w*k   # scaled to 6% portfolio vol

import pickle, sys

def _setup():
    dret=load_daily_returns(); spy=get_spy_daily()
    cols=list(dret.columns); col_to_idx={c:int(c.split(" ")[1]) for c in cols}
    name_cap_all=np.array([MODEL_VOL_CAPS[col_to_idx[c]] for c in cols])
    grp_idx_all={g:[i for i,c in enumerate(cols) if col_to_idx[c] in set(ids)] for g,(ids,_) in GROUPS.items()}
    grp_cap={g:GROUPS[g][1] for g in GROUPS}
    return dret,spy,cols,name_cap_all,grp_idx_all,grp_cap

def do_window(o):
    dret,spy,cols,name_cap_all,grp_idx_all,grp_cap=_setup()
    d=dret.iloc[:len(dret)-o] if o else dret
    spyr=spy.pct_change().reindex(d.index); nt=len(d.index)
    OOS=nt-OOS_DAYS; rt=list(range(OOS-EXEC_LAG_D,nt-EXEC_LAG_D,SIGNAL_REFRESH_EVERY_D))
    keys=["VP","cmv","cblend"]; prev={k:None for k in keys}; tt={k:0.0 for k in keys}; rr={k:[] for k in keys}
    for t in rt:
        active=np.array([d[c].iloc[max(0,t-COV_WINDOW_D):t].notna().sum()>=MIN_ACTIVE_OBS_D for c in cols])
        cs=d.iloc[max(0,t-COV_WINDOW_D):t]; cov=factor_covariance_daily(cs,spyr.reindex(cs.index))
        sig_f=np.sqrt(np.maximum(np.diag(cov),0)); w_vp=vol_parity_weights(sig_f,active,cols)
        act=np.where(active)[0]; w_cmv=np.zeros(len(cols))
        if len(act)>=5:
            covs=[]
            for win in MV_WINDOWS:
                R=d.iloc[max(0,t-win):t,:].iloc[:,act].dropna(how="any").values
                if R.shape[0]>R.shape[1]+5: covs.append(LedoitWolf().fit(R).covariance_*252.0)
            if covs:
                S=np.mean(covs,axis=0); sig=np.sqrt(np.maximum(np.diag(S),0)); cap=name_cap_all[act]
                groups=[[list(act).index(i) for i in grp_idx_all[g] if i in set(act)] for g in GROUPS]
                gcaps=[grp_cap[g] for g in GROUPS]
                w_cmv[act]=constrained_minvar(S,sig,cap,groups,gcaps,iters=3)
        W={"VP":w_vp,"cmv":w_cmv,"cblend":0.25*w_vp+0.75*w_cmv}
        for k in keys:
            if prev[k] is not None: tt[k]+=np.abs(W[k]-prev[k]).sum()
            prev[k]=W[k]
        es=t+EXEC_LAG_D; ee=min(es+SIGNAL_REFRESH_EVERY_D,nt)
        for dd in range(es,ee):
            if dd<OOS: continue
            rv=d.iloc[dd].fillna(0).values
            for k in keys: rr[k].append(float(W[k]@rv))
    yrs=len(rr["VP"])/252.0
    out={k:{"stats":gstats(pd.Series(rr[k])),"turn":tt[k]/yrs} for k in keys}
    with open(f"results/cmv_off{o}.pkl","wb") as f: pickle.dump(out,f)
    print(f"window off={o} done ({len(rr['VP'])} days)",flush=True)

def aggregate():
    labels={"VP":"VP (baseline)","cmv":"Constrained min-var ens","cblend":"Constrained 75/25 blend"}
    R={}
    for o in OFF:
        with open(f"results/cmv_off{o}.pkl","rb") as f: R[o]=pickle.load(f)
    print(f"{'variant':<26s} {'grossSh':>7s} {'recent':>7s} {'turn/yr':>7s}")
    for k in ["VP","cmv","cblend"]:
        gm=np.mean([R[o][k]['stats'][2] for o in OFF]); rec=R[0][k]['stats'][2]; tm=np.mean([R[o][k]['turn'] for o in OFF])
        print(f"{labels[k]:<26s} {gm:7.3f} {rec:7.3f} {tm:7.1f}")
    print("\nper-window gross Sharpe (2014-16..2022-24 reversed; off=0 is recent):")
    for k in ["VP","cmv","cblend"]:
        print(f"  {labels[k]:<26s} " + " ".join(f"{R[o][k]['stats'][2]:.3f}" for o in OFF))

if __name__=="__main__":
    args=sys.argv[1:]
    w=[a for a in args if a.startswith("--off=")]
    if w: do_window(int(w[0].split("=")[1]))
    elif "--aggregate" in args: aggregate()
    else:
        for o in OFF: do_window(o)
        aggregate()