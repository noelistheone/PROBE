#!/usr/bin/env python
"""Numerical verification of Proposition 2 (degree-dependent optimal regularization).
Model: user embedding estimated from d_u noisy observations, regularizer pulls toward anchor.
Objective: (1/d_u) sum_i ||y_i - e||^2 + mu ||e - a||^2  =>  e_hat = (ybar + mu*a)/(1+mu)
Risk:      R(mu) = (D*sigma^2/d_u + mu^2 b^2)/(1+mu)^2   =>   mu* = D*sigma^2/(d_u b^2)  ~ 1/d_u
"""
import numpy as np, json
rng=np.random.default_rng(0); D=64
def emp_opt(d,sig=1.,b=1.,trials=1500):
    e=np.zeros(D); a=e.copy(); a[0]=b; best=(None,1e18)
    for mu in np.concatenate([[0],np.logspace(-3,2,60)]):
        err=np.mean([((((e+rng.normal(0,sig/np.sqrt(d),D))+mu*a)/(1+mu)-e)**2).sum() for _ in range(trials)])
        if err<best[1]: best=(mu,err)
    return best[0]
out=[]
for d in [4,16,64,256]:
    out.append({'degree':d,'theory_mu_star':D/(d*1.0),'empirical_mu_star':float(emp_opt(d))})
    print(out[-1])
json.dump(out,open('results/wm/theory_prop2_verification.json','w'),indent=2)
print('saved results/wm/theory_prop2_verification.json')
