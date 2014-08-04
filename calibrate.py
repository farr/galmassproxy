#!/usr/bin/env python

r"""calibrate.py

Run this program to produce a calibrated generative model for the
mass-proxy relationship in a data set.  It expects a data file
containing a header line with at least the entries 'mass', 'proxy',
'dm', and 'dp', followed by one line for each measurement of the
quantities and observational relative uncertainties:

mass proxy dm  dp
m1   p1    dm1 dp1
...

additional labelled columns are ignored.

Running the program produces output in a directory that will allow
subsequent estimates of masses from proxy observations using the
calibrated model.  In that directory are pickled versions of the
generative posterior and MCMC samples, information to enable
continuation of the fit if desired, and a PDF plot of the fit.

"""

import argparse
import bz2
import emcee
import numpy as np
import os.path as op
import pickle
import plots
import plotutils.runner as pr
import posterior as pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', metavar='FILE', required=True, help='observational data')
    parser.add_argument('--outdir', metavar='DIR', default='.', help='output directory (default %(default)s)')

    args = parser.parse_args()

    data = np.genfromtxt(args.data, names=True)

    logpost = pos.Posterior(np.log(data['mass']), np.log(data['proxy']), data['dm'], data['dp'])
    sampler = emcee.EnsembleSampler(128, logpost.nparams, logpost)
    runner = pr.EnsembleSamplerRunner(sampler, logpost.pguess() + 1e-3*np.random.randn(128, logpost.nparams))
    runner.run_to_neff(32, savedir=args.outdir)
    
    with bz2.BZ2File(op.join(args.outdir, 'logpost.pkl.bz2'), 'w') as out:
        pickle.dump(logpost, out)

    plots.plot_fit(logpost, runner.thin_chain, outdir=args.outdir)
