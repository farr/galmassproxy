import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import plotutils.plotutils as pu
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

def plot_est_masses(logpost, chain, obs_proxies, obs_dproxies, obs_masses, obs_dmasses, ptrue, mtrue, alpha, b):
    pp.axes([0.16, 0.16, 0.95-0.16, 0.95-0.16])

    masses = np.zeros(obs_proxies.shape[0])
    dmasses = np.zeros(obs_dproxies.shape[0])
    proxies = np.zeros(obs_proxies.shape[0])
    dproxies = np.zeros(obs_dproxies.shape[0])

    flatchain = chain.reshape((-1, chain.shape[2]))

    for par in flatchain:
        for i, (p, dp) in enumerate(zip(obs_proxies, obs_dproxies)):
            mu, v = logpost.mass_proxy_estimate(par, p, dp)
            masses[i] += mu[0]
            proxies[i] += mu[1]
            dmasses[i] += v[0]
            dproxies[i] += v[1]

    masses /= flatchain.shape[0]
    dmasses /= flatchain.shape[0]
    proxies /= flatchain.shape[0]
    dproxies /= flatchain.shape[0]

    dmasses = np.sqrt(dmasses)
    dproxies = np.sqrt(dproxies)

    logerrorbar(obs_proxies, obs_masses, xerr=obs_dproxies, yerr=obs_dmasses, color='k', fmt='.')
    logerrorbar(proxies, masses, xerr=dproxies, yerr=dmasses, color='b', fmt='.')

    xs = np.linspace(min(np.min(proxies), np.min(obs_proxies)) - 0.3,
                     max(np.max(proxies), np.max(obs_proxies)) + 0.3,
                     100)
    ys = alpha*xs + b

    pp.plot(np.exp(xs), np.exp(ys), '-k')
    pp.plot(np.exp(ptrue), np.exp(masses), '*k')

    pp.xscale('log', nonposx='clip')
    pp.yscale('log', nonposy='clip')

    pp.xlabel(r'$X$')
    pp.ylabel(r'$M$')

def plot_fit(logpost, chain):
    pp.axes([0.16, 0.16, 0.95-0.16, 0.95-0.16])

    flatchain = chain.reshape((-1, 5))

    logerrorbar(logpost.obs_proxies, logpost.obs_masses, xerr=logpost.obs_dproxies, yerr=logpost.obs_dmasses, color='k', fmt='.')
    
    alphas = []
    mus = []
    for p in flatchain:
        alphas.append(logpost.alpha(p))
        mus.append(p[0:2])
    alphas = np.array(alphas)
    mus = np.array(mus)

    mean_alpha = np.mean(alphas)
    mean_b = np.mean(mus[:,0] - alphas*mus[:,1])

    xs = np.linspace(np.min(logpost.obs_proxies) - np.max(logpost.obs_dproxies), \
                     np.max(logpost.obs_proxies) + np.max(logpost.obs_dproxies), \
                     100)
    ys = mean_alpha*xs + mean_b

    pp.plot(np.exp(xs), np.exp(ys), '-k')

    for i in range(10):
        index = np.random.randint(alphas.shape[0])
        al = alphas[index]
        b = mus[index,0] - al*mus[index,1]
        ys = al*xs + b
        pp.plot(np.exp(xs), np.exp(ys), '-k', alpha=0.1)

    pp.xlabel(r'$X$')
    pp.ylabel(r'$M$')
