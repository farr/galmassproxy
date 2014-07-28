#!/usr/bin/env python

import emcee
import numpy as np
import os.path as op
import plotutils.runner as pr
import posterior as pos

# Observational error sizes from Richard's paper; should be applied to
# sorted (smallest-to-largest) masses
dms = np.array([ 0.07948718,  0.13670886,  0.14867424,  0.3384058 ,  0.21818182,
                 0.25257732,  0.14150944,  0.1641791 ,  0.23916084,  0.19162011,
                 0.1168    ,  0.13932584,  0.13492063,  0.10654008,  0.14329897,
                 0.11134021,  0.12027398,  0.185     ,  0.05628476,  0.169375  ,
                 0.25421687,  0.2027027 ])
dxs = np.array([ 0.22858724,  0.29014214,  0.37205699,  0.45214178,  0.16864295,
                 0.44537327,  0.21145289,  0.28158675,  0.20106134,  0.34335918,
                 0.31692716,  0.17382953,  0.15172787,  0.24297976,  0.29024132,
                 0.22649368,  0.32205503,  0.2647911 ,  0.21821011,  0.3049213 ,
                 0.34063273,  0.4068833 ])

def draw_data(alpha, b, sigma, return_mtrue = False, return_ptrue = False, dms=dms, dxs=dxs):
    N = 22

    ms = np.random.uniform(low=np.log(1e13), high=np.log(1e15), size=N)
    ms = np.sort(ms) # Small to large, to match error terms

    xs = 1/alpha*ms - b/alpha + np.random.normal(loc=0, scale=sigma*np.log(10.0), size=N)

    data = ms+np.random.normal(loc=0, scale=dms), xs+np.random.normal(loc=0, scale=dxs), \
           dms, dxs

    rval = data

    if return_ptrue:
        rval = (xs, ) + rval
    if return_mtrue:
        rval = (ms, ) + rval

    return rval

def process_thin_flatchain(alpha, b, sigma, logpost, fc):
    alphas = []
    alpha_var = []
    for p in fc:
        alphas.append(logpost.alpha(p))
    alpha_bias = alpha - np.mean(alphas)
    alpha_var = np.var(alphas)
    
    ms, mobs, pobs, dmobs, dpobs = draw_data(alpha, b, sigma, return_mtrue=True)
    mbias = []
    mvar = []
    for m, p, dp in zip(ms, pobs, dpobs):
        post_ms = []
        post_vs = []
        for par in fc:
            (mu, _), (s2, _) = logpost.mass_proxy_estimate(par, p, dp)
            post_ms.append(mu)
            post_vs.append(s2)
        mbias.append(m - np.mean(post_ms))
        mvar.append(np.mean(post_vs))
    mbias = np.mean(mbias)
    mvar = np.mean(mvar)

    return alpha_bias, np.sqrt(alpha_var), mbias, np.sqrt(mvar)

bias_header = '# alpha_bias alpha_sigma m_bias m_sigma\n'

if __name__ == '__main__':
    alpha = 0.8
    b = 5.0
    sig = 0.3
    output = 'bias.dat'

    if not op.isfile(output):
        with open(output, 'w') as out:
            out.write(bias_header)

    while True:
        data = draw_data(alpha, b, sig)
        logpost = pos.Posterior(*data)
        sampler = emcee.EnsembleSampler(100, 5, logpost)
        runner = pr.EnsembleSamplerRunner(sampler, logpost.pguess() + 1e-3*np.random.randn(100,5))

        for i in range(10):
            runner.run_mcmc(1000)

            try:
                if runner.thin_flatchain.shape[0] > 2000:
                    break
            except:
                pass
        try:
            fc = runner.thin_flatchain[-1000:,:]
        except:
            print 'Reached 10K iterations without convergence!'

        row = process_thin_flatchain(alpha, b, sig, logpost, fc)
        with open(output, 'a') as out:
            out.write('{0:g}\t{1:g}\t{2:g}\t{3:g}\n'.format(*row))
