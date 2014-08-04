#!/usr/bin/env python

r"""estimate.py

Use to estimate masses based on observed proxy values (and associated
errors) from a pre-calibrated generative model for the mass-proxy
relationship.  The estimates will be returned as samples (fair draws)
from the model's posterior on the mass given the proxy observation.
This program expects the proxy data in a file with at least 'proxy'
and 'dp' column headers, followed by observed proxy values and
relative errors in those columns:

proxy dp
p1 dp1
...

The output will have one row for each proxy measurement, with one
column for each draw from the mass posterior for that system:

m1_draw m1_draw ...
m2_draw m2_draw ...
...

"""

import argparse
import bz2
import numpy as np
import os.path as op
import pickle
import posterior as pos
import plotutils.runner as pr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--caldir', metavar='DIR', required=True, help='directory with calibration data')
    parser.add_argument('--proxyfile', metavar='FILE', required=True, help='proxy observations')
    parser.add_argument('--output', metavar='FILE', default='masses.dat.bz2', help='mass posterior draws')

    args = parser.parse_args()

    runner = pr.load_runner(args.caldir)
    with bz2.BZ2File(op.join(args.caldir, 'logpost.pkl.bz2'), 'r') as inp:
        logpost = pickle.load(inp)

    flatchain = runner.thin_flatchain[-2048:,:]

    data = np.genfromtxt(args.proxyfile, names=True)

    ms = []
    for log_p, dp in zip(np.log(data['proxy']), data['dp']):
        mdraws = []
        for p in flatchain:
            ((log_m, log_p_est), (var_log_m, var_log_p)) = \
                logpost.mass_proxy_estimate(p, log_p, dp)
            mdraws.append(np.exp(np.random.normal(loc=log_m, scale=np.sqrt(var_log_m))))
        ms.append(mdraws)
    ms = np.array(ms)

    fname = args.output
    fbase, fext = op.splitext(fname)
    if not (fext == '.bz2'):
        fname = fname + '.bz2'

    with bz2.BZ2File(fname, 'w') as out:
        np.savetxt(out, ms)
