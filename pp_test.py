#!/usr/bin/env python

import emcee
import numpy as np
import posterior as pos
import plotutils.plotutils as pu
import plotutils.runner as pr

def draw_p0():
    p0 = np.zeros(5)

    p0[:2] = np.random.uniform(low=np.log(1e13), high=np.log(1e15), size=2)
    p0[2] = np.random.uniform(low=3.5, high=5.5)
    p0[3] = np.random.uniform(low=0.1, high=0.4)
    p0[4] = np.random.uniform(low=np.pi/8.0, high=3.0*np.pi/8.0)

    return p0

def draw_data(p0, N=100, dms=None, dxs=None):
    if dms is None or dxs is None:
        dms = np.random.lognormal(mean=np.log(0.5), sigma=0.25, size=N)
        dxs = np.random.lognormal(mean=np.log(0.5), sigma=0.25, size=N)

    logpost = pos.Posterior(np.zeros(N), np.zeros(N), dms, dxs)

    return logpost.draw_data(p0)

def one_run_ps(p0, data):
    logpost = pos.Posterior(*data)
    sampler = emcee.EnsembleSampler(100, 5, logpost)
    runner = pr.EnsembleSamplerRunner(sampler, logpost.pguess() + 1e-3*np.random.randn(100,5))

    ndone = 0
    while True:
        runner.run_mcmc(1000)
        ndone += 1000
        if ndone > 10000:
            raise ValueError('could not converge')

        try:
            if runner.thin_flatchain.shape[0] > 2000:
                break
        except:
            pass

    fc = runner.thin_flatchain[-1000:,:]
    ps = np.array([float(np.sum(fc[:,i] < p0[i]))/fc.shape[0] for i in range(5)])

    return ps
    
if __name__ == '__main__':
    output = 'ps.dat'
    
    while True:
        p0 = draw_p0()
        data = draw_data(p0)
        try:
            ps = one_run_ps(p0, data)
        except ValueError:
            continue

        with open(output, 'a') as out:
            out.write('\t'.join(['{0:g}'.format(p) for p in ps]) + '\n')

    
