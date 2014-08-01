#!/usr/bin/env python

import argparse
import emcee
import numpy as np
import os.path as op
import plots
import plotutils.runner as pr
import posterior as pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', metavar='FILE', required=True, help='observational data')
    parser.add_argument('--outdir', metavar='DIR', default='.', help='output directory (default %(default)s)')

    args = parser.parse_args()

    # Data file should have 
    #
    # Mobs dM/M Xobs dX/X
    # 
    # format
    data = np.loadtxt(args.data)
    
    logpost = pos.Posterior(np.log(data[:,0]), np.log(data[:,2]), data[:,1], data[:,3])
    sampler = emcee.EnsembleSampler(128, logpost.nparams, logpost)
    runner = pr.EnsembleSamplerRunner(sampler, logpost.pguess() + 1e-3*np.random.randn(128, logpost.nparams))
    runner.run_to_neff(16, savedir=args.outdir)

    plots.plot_fit(logpost, runner.thin_chain, outdir=args.outdir)
