import numpy as np
import matplotlib.pyplot as pp
import triangle as tri

def plot_lnprobability(lnprob):
    pp.figure()
    pp.plot(np.mean(lnprob, axis=0))

    pp.ylabel(r'$\ln \pi$')
    pp.xlabel(r'Iteration')

def plot_chain(chain, mean=True):
    n = chain.shape[2]
    m = int(np.ceil(np.sqrt(n)))
    
    pp.figure()
    
    for k in range(n):
        pp.subplot(m,m,k+1)

        if mean:
            pp.plot(np.mean(chain[:,:,k], axis=0))
        else:
            pp.plot(chain[:,:,k].T)

def _plot_data(logpost, *args, **kwargs):
    pp.errorbar(np.exp(logpost.obs_masses), np.exp(logpost.obs_proxies), *args,
                fmt='.', color='k',
                xerr=np.exp(logpost.obs_masses)*logpost.obs_dmasses,
                yerr=np.exp(logpost.obs_proxies)*logpost.obs_dproxies, **kwargs)
    pp.xscale('log')
    pp.yscale('log')

def plot_fit_data(logpost, chain):
    mean_alpha = np.mean(chain[:,:,0])
    mean_beta = np.mean(chain[:,:,1])

    flatchain = chain.reshape((-1, chain.shape[2]))

    mass_spread = np.std(logpost.obs_masses)
    log_ms = np.linspace(np.min(logpost.obs_masses)-0.25*mass_spread, np.max(logpost.obs_masses) + 0.25*mass_spread, 1000)
    log_mean_proxy = mean_alpha * log_ms + mean_beta

    pp.figure()

    _plot_data(logpost)

    pp.plot(np.exp(log_ms), np.exp(log_mean_proxy), '-b')

    for i in range(10):
        ind = np.random.randint(flatchain.shape[0])
        p = flatchain[ind,:]

        pp.plot(np.exp(log_ms), np.exp(p[0]*log_ms + p[1]), '-b', alpha=0.25)

    pp.xlabel(r'$M$')
    pp.ylabel(r'$M_\mathrm{proxy}$')

def plot_corner(logpost, chain):
    tri.corner(chain.reshape((-1, chain.shape[2])), labels=logpost.pnames, quantiles=[0.05, 0.95])

def plot_posterior_masses(logpost, chain):
    flatchain = chain.reshape((-1, chain.shape[2]))

    ms = np.zeros(logpost.obs_masses.shape[0])
    ps = np.zeros(logpost.obs_proxies.shape[0])
    dms = np.zeros(logpost.obs_masses.shape[0])
    dps = np.zeros(logpost.obs_proxies.shape[0])

    for p in flatchain:
        mest = logpost.mass_estimate(p, logpost.obs_masses, logpost.obs_proxies, logpost.obs_dmasses, logpost.obs_dproxies)
        dmest = logpost.mass_variance_estimate(p, logpost.obs_dmasses, logpost.obs_dproxies)

        ms += mest
        ps += p[0]*mest + p[1]

        dms += dmest
        dps += p[0]*p[0]*dmest

    ms /= flatchain.shape[0]
    ps /= flatchain.shape[0]
    dms /= flatchain.shape[0]
    dps /= flatchain.shape[0]

    dms = np.sqrt(dms)
    dps = np.sqrt(dps)

    pp.figure()

    _plot_data(logpost, label='Data')

    pp.errorbar(np.exp(ms), np.exp(ps), fmt='.', color='b',
                xerr=np.exp(ms)*dms, yerr=np.exp(ps)*dps,
                label='Posterior')

    pp.xlabel(r'$M$')
    pp.ylabel(r'$M_\mathrm{proxy}$')

    pp.legend(loc='lpper left')
