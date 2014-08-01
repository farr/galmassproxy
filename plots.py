import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import os.path as op
import plotutils.plotutils as pu
import scipy.stats as ss
import triangle as tri

def setup():
    width = 240.0/72.27
    params = {'backend': 'pdf',
              'axes.labelsize': 10,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [width,width]}
    mpl.rcParams.update(params)

def axes():
    pp.axes([0.16, 0.16, 0.95-0.16, 0.95-0.16])

def plot_corner(logpost, chain):
    tri.corner(chain.reshape((-1, chain.shape[2])), labels=logpost.pnames, quantiles=[0.05, 0.95])

def logerrorbar(xs, ys, xerr=None, yerr=None, *args, **kwargs):
    if xerr is not None:
        dxs = [np.exp(xs)-np.exp(xs-xerr), np.exp(xs+xerr)-np.exp(xs)]
    else:
        dxs = xerr

    if yerr is not None:
        dys = [np.exp(ys)-np.exp(ys-yerr), np.exp(ys+yerr)-np.exp(ys)]
    else:
        dys = yerr

    pp.errorbar(np.exp(xs), np.exp(ys), xerr=dxs, yerr=dys, *args, **kwargs)
    pp.xscale('log')
    pp.yscale('log')

def plot_mass_estimates(logpost, chain, obs_proxies, obs_dproxies, mtrues, outdir=None):
    if outdir is not None:
        setup()
        axes()

    flatchain = chain.reshape((-1, chain.shape[2]))

    mests = []
    mvars = []
    for pobs, dp, m in zip(obs_proxies, obs_dproxies, mtrues):
        ms = []
        vs = []
        for p in flatchain:
            (mest, _), (vest, _) = logpost.mass_proxy_estimate(p, pobs, dp)
            ms.append(mest-m)
            vs.append(vest)
        mests.append(np.array(ms))
        mvars.append(np.array(vs))
    mests = np.array(mests)
    mvars = np.array(mvars)

    xs = np.linspace(-4, 4, 500)
    ys = []
    for mest, mvar in zip(mests, mvars):
        y = 0
        for m, v in zip(mest, mvar):
            y += ss.norm.pdf(xs, loc=m, scale=np.sqrt(v))
        y /= mest.shape[0]
        ys.append(y)
    ys = np.array(ys)

    for y in ys:
        pp.plot(np.exp(xs), y)
    pp.xscale('log')
    pp.xlabel(r'$M_{500}/M_{500}^{true}$')
    pp.ylabel(r'$p\left(\ln \, M_{500}/M_{500}^{true}\right)$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'pmest.pdf'))
        with open(op.join(outdir, 'pmest.dat'), 'w') as out:
            out.write('# ln(M500/M500true)\tp1\tp2\t...\n')
            np.savetxt(out, np.column_stack((xs, ys.T)))

def plot_mass_corrections(logpost, chain, obs_proxies, obs_dproxies, proxies, masses, alpha, b, outdir=None):
    if outdir is not None:
        setup()
        axes()

    flatchain = chain.reshape((-1, chain.shape[2]))

    # Plot the "truth" line
    pp.plot(np.exp(np.sort(proxies)), np.exp(alpha*np.sort(proxies) + b), '-k')

    # Compute the mass estimate and variance
    pred_ms = []
    pred_dms = []
    for pobs, dp in zip(obs_proxies, obs_dproxies):
        ms = []
        vs = []
        for p in flatchain:
            (m, _), (dm, _) = logpost.mass_proxy_estimate(p, pobs, dp)
            ms.append(m)
            vs.append(dm)
        pred_ms.append(np.mean(ms))
        pred_dms.append(np.sqrt(np.mean(vs)))
    pred_ms = np.array(pred_ms)
    pred_dms = np.array(pred_dms)

    # Plot truth
    pp.plot(np.exp(proxies), np.exp(masses), '*k')

    # Plot the observations and predictions
    logerrorbar(obs_proxies, pred_ms, xerr=obs_dproxies, yerr=pred_dms, color='k', fmt='.')

    pp.xscale('log')
    pp.yscale('log')
    pp.xlabel(r'$X$')
    pp.ylabel(r'$M_{500}$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'mcorr.pdf'))
        with open(op.join(outdir, 'mcorr.dat'), 'w') as out:
            out.write('# ln(X)\tln(M)\tln(X_obs)\tdX_obs/X_obs\tln(M_pred)\tdM_pred/M_pred\n')
            np.savetxt(out, np.column_stack((proxies, masses, obs_proxies, obs_dproxies, pred_ms, pred_dms)))

def plot_fit(logpost, chain, outdir=None):
    if outdir is not None:
        setup()
        axes()

    flatchain = chain.reshape((-1, chain.shape[2]))

    xs = np.sort(logpost.obs_proxies)

    mean_ys = 0
    for p in flatchain:
        mean_ys += logpost.fit_line(p, xs)
    mean_ys /= flatchain.shape[0]

    ys = []
    for i in range(10):
        p = flatchain[np.random.randint(flatchain.shape[0]), :]
        ys.append(logpost.fit_line(p, xs))
    ys = np.array(ys)

    logerrorbar(logpost.obs_proxies, logpost.obs_masses, xerr=logpost.obs_dproxies, yerr=logpost.obs_dmasses, color='k', fmt='.')
    pp.plot(np.exp(xs), np.exp(mean_ys), '-k')
    for y in ys:
        pp.plot(np.exp(xs), np.exp(y), '-k', alpha=0.1)

    if outdir is not None:
        pp.savefig(op.join(outdir, 'fit.pdf'))
