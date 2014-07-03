#!/usr/bin/env python

import emcee
import numpy as np
import plotutils.runner as pr
import posterior as pos

def draw_data(alpha, b, sigma, return_mtrue = False):
    N = 22
    dms = np.random.lognormal(mean=np.log(0.5), sigma=0.25, size=N)
    dxs = np.random.lognormal(mean=np.log(0.5), sigma=0.25, size=N)

    ms = np.random.uniform(low=np.log(1e13), high=np.log(1e15), size=N)

    xs = 1/alpha*ms - b/alpha + np.random.normal(loc=0, scale=sigma*np.log(10.0), size=N)

    data = ms+np.random.normal(loc=0, scale=dms), xs+np.random.normal(loc=0, scale=dxs), \
           dms, dxs

    if return_mtrue:
        return (ms, ) + data
    else:
        return data

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
            mu, s2 = logpost.mass_estimate_mean_variance(par, p, dp)
            post_ms.append(mu)
            post_vs.append(s2)
        mbias.append(m - np.mean(post_ms))
        mvar.append(np.mean(post_vs))
    mbias = np.mean(mbias)
    mvar = np.mean(mvar)

    return alpha_bias, np.sqrt(alpha_var), mbias, np.sqrt(mvar)

if __name__ == '__main__':
    alpha = 0.8
    b = -5.0
    sig = 0.3
    output = 'bias.dat'

    while True:
        data = draw_data(alpha, b, sig)
        logpost = pos.Posterior(*data)
        sampler = emcee.EnsembleSampler(100, 5, logpost)
        runner = pr.EnsembleSamplerRunner(sampler, logpost.pguess() + 1e-3*np.random.randn(100,5))

        for i in range(10):
            runner.run_mcmc(1000)

            try:
                if runner.thin_flatchain.shape[0] > 1000:
                    break
            except:
                pass
        try:
            fc = runner.thin_flatchain
        except:
            print 'Reached 10K iterations without convergence!'

        row = process_thin_flatchain(alpha, b, sig, logpost, fc)
        with open(output, 'a') as out:
            out.write('{0:g}\t{1:g}\t{2:g}\t{3:g}\n'.format(*row))
