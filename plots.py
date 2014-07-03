import numpy as np
import matplotlib.pyplot as pp
import plotutils.plotutils as pu
import triangle as tri

def _plot_data(logpost, *args, **kwargs):
    pp.errorbar(np.exp(logpost.obs_proxies), np.exp(logpost.obs_masses), *args,
                fmt='.', color='k',
                xerr=(-np.exp(logpost.obs_proxies)*(np.expm1(-logpost.obs_dproxies)),
                      np.exp(logpost.obs_proxies)*np.expm1(logpost.obs_dproxies)),
                yerr=(-np.exp(logpost.obs_masses)*(np.expm1(-logpost.obs_dmasses)),
                      np.exp(logpost.obs_masses)*np.expm1(logpost.obs_dmasses)),
                **kwargs)
    pp.xscale('log')
    pp.yscale('log')

def plot_fit_data(logpost, flatchain):
    proxy_spread = np.std(logpost.obs_proxies)
    proxies = np.linspace(np.min(logpost.obs_proxies)-0.25*proxy_spread, np.max(logpost.obs_proxies) + 0.25*proxy_spread, 100)

    mean_masses = []
    for px in proxies:
        ms = []
        for p in flatchain:
            ms.append(logpost.mass_estimate_mean_variance(p, px, 0.0)[0])
        mean_masses.append(np.mean(ms))
    mean_masses = np.array(mean_masses)
    
    pp.figure()

    _plot_data(logpost)

    pp.plot(np.exp(proxies), np.exp(mean_masses), '-b')

    for i in range(10):
        ind = np.random.randint(flatchain.shape[0])
        p = flatchain[ind,:]

        ms = []
        for px in proxies:
            ms.append(logpost.mass_estimate_mean_variance(p, px, 0.0)[0])
        ms = np.array(ms)

        pp.plot(np.exp(proxies), np.exp(ms), '-b', alpha=0.25)

    pp.ylabel(r'$M$')
    pp.xlabel(r'$M_\mathrm{proxy}$')

def plot_corner(logpost, chain):
    tri.corner(chain.reshape((-1, chain.shape[2])), labels=logpost.pnames, quantiles=[0.05, 0.95])
