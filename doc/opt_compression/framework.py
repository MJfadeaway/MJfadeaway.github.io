"""Optimizer-as-compressor framework: how data moves parameters from init to solution.
Pure numpy. Run: python framework.py  -> results.json + figs
Components:
 (C1) Data-span: Delta = theta_t - theta_0 = -sum_i a_i(t) grad_i  (per-sample gradients span).
 (C2) Spectral mode dynamics: eigen-decomp of tangent kernel K=J J^T/n predicts residual decay.
 (C3) Compression measure: coding rate R(Delta)=1/2 logdet(I+ d/(eps^2 m) Delta Delta^T) of per-layer
      displacement + effective rank; tracks how many directions the optimizer 'spends'.
 (C4) Generalization split: test error vs fraction of displacement in top-k signal modes.
"""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def make_data(n, d, ntest, noise, rng):
    Wt = rng.standard_normal((d, 3)) / np.sqrt(d); v = rng.standard_normal(3)
    f = lambda X: np.tanh(X @ Wt) @ v
    X = rng.standard_normal((n, d)); Xt = rng.standard_normal((ntest, d))
    return X, f(X) + noise*rng.standard_normal(n), Xt, f(Xt)

class MLP:  # f(x)= a^T tanh(W x)/sqrt(m) * alpha  (centered at init: f0 subtracted)
    def __init__(s, d, m, alpha, rng):
        s.W = rng.standard_normal((m, d))/np.sqrt(d); s.a = rng.standard_normal(m); s.m=m; s.al=alpha
    def theta(s): return np.concatenate([s.W.ravel(), s.a])
    def set(s, th): s.W = th[:s.W.size].reshape(s.W.shape); s.a = th[s.W.size:]
    def f(s, X): return s.al*(np.tanh(X@s.W.T)@s.a)/np.sqrt(s.m)
    def jac(s, X):  # per-sample gradients, n x P
        H = np.tanh(X@s.W.T); D = 1-H**2; c = s.al/np.sqrt(s.m)
        JW = (D*s.a[None])[:, :, None]*X[:, None, :]*c
        return np.concatenate([JW.reshape(len(X), -1), H*c], 1)

def coding_rate(M, eps=0.1):
    m, n = M.shape; G = np.eye(min(m, n)) + (m/(eps**2*n))*(M@M.T if m<=n else M.T@M)
    return 0.5*np.linalg.slogdet(G)[1]
def eff_rank(s): p = s**2/(np.sum(s**2)+1e-30); return float(np.exp(-np.sum(p*np.log(p+1e-30))))

def run(alpha, seed=0, n=40, d=20, m=200, lr=0.5, T=3000, noise=0.1, log=(0,10,30,100,300,1000,3000)):
    rng = np.random.default_rng(seed); X,y,Xt,yt = make_data(n,d,2000,noise,rng)
    net = MLP(d,m,alpha,rng); f0 = net.f(X); f0t = net.f(Xt); th0 = net.theta()
    J0 = net.jac(X); K0 = J0@J0.T/n; ev, U = np.linalg.eigh(K0); ev, U = ev[::-1], U[:, ::-1]
    r0 = y; lin = U.T@r0; out = dict(alpha=alpha, ev=ev.tolist(), t=[], train=[], test=[], span_res=[],
        pred_err=[], cr=[], erank=[], dist=[], sig_frac=[])
    # lr scaled by alpha^2 so function-space speed is comparable across scales
    eta = lr/alpha**2
    for t in range(T+1):
        res = y - (net.f(X)-f0); J = net.jac(X)   # centered predictor
        if t in log:
            D = net.theta()-th0
            # C1: fraction of Delta outside span of per-sample gradients along path (use J0 and J_t rows)
            B = np.linalg.qr(np.vstack([J0,J]).T)[0]; sp = np.linalg.norm(D-B@(B.T@D))/(np.linalg.norm(D)+1e-12)
            # C2: linearized-dynamics prediction of residual
            rp = U@((1-eta*ev)**t*lin)   # discrete linearized dynamics, K0 already contains alpha^2
            pe = np.linalg.norm(rp-res)/np.linalg.norm(y)
            W0 = th0[:net.W.size].reshape(net.W.shape); dW = net.W-W0
            sv = np.linalg.svd(dW, compute_uv=False)
            k = 5
            # C4: share of function change living in top-k kernel modes
            fchg = (net.f(X)-f0); sig = np.linalg.norm(U[:, :k].T@fchg)**2/(np.linalg.norm(fchg)**2+1e-12)
            out['t'].append(t); out['train'].append(float(np.mean(res**2)))
            out['test'].append(float(np.mean((yt-(net.f(Xt)-f0t))**2)))
            out['span_res'].append(float(sp)); out['pred_err'].append(float(pe))
            out['cr'].append(float(coding_rate(dW))); out['erank'].append(eff_rank(sv))
            out['dist'].append(float(np.linalg.norm(D))); out['sig_frac'].append(float(sig))
        g = J.T@res/n; th = net.theta()+eta*g; net.set(th)
    return out

if __name__ == "__main__":
    R = {a: run(a) for a in (0.3, 1.0, 3.0)}   # small alpha -> rich/feature learning ... large -> lazy (NTK)
    # note: alpha scaling in jac uses al; lazy when alpha large
    json.dump({str(a): v for a, v in R.items()}, open("results.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, o in R.items():
        ax[0].loglog(np.array(o['t'])+1, o['train'], 'o-', label=f'alpha={a}')
        ax[1].semilogx(np.array(o['t'])+1, o['pred_err'], 'o-')
        ax[2].semilogx(np.array(o['t'])+1, o['erank'], 'o-')
        ax[3].semilogx(np.array(o['t'])+1, o['test'], 'o-')
    for i, t in enumerate(["train MSE", "linear-mode prediction err /|y| (C2)", "eff. rank of dW (C3)", "test MSE (C4)"]): ax[i].set_title(t); ax[i].set_xlabel("step+1")
    ax[0].legend(); plt.tight_layout(); plt.savefig("fig_main.png", dpi=110)
    for a, o in R.items():
        print(f"alpha={a}")
        for k in ('t','train','test','span_res','pred_err','cr','erank','dist','sig_frac'): print(" ", k, np.round(o[k], 4).tolist())
