import numpy as np
import scipy.stats as ss

class Posterior(object):
    def __init__(self, obs_masses, obs_proxes, obs_dmasses, obs_dproxies):
        self._obs_masses = obs_masses
        self._obs_proxies = obs_proxes
        self._obs_dmasses = obs_dmasses
        self._obs_dproxies = obs_dproxies

    @property
    def obs_masses(self):
        return self._obs_masses

    @property
    def obs_proxies(self):
        return self._obs_proxies

    @property
    def obs_dmasses(self):
        return self._obs_dmasses

    @property
    def obs_dproxies(self):
        return self._obs_dproxies

    @property
    def dtype(self):
        return np.dtype([('alpha', np.float),
                         ('beta', np.float),
                         ('sigma', np.float)])

    @property
    def pnames(self):
        return [r'$\alpha', r'$\beta$', r'$\sigma$']

    def to_params(self, p):
        return p.view(self.dtype).squeeze()

    def log_prior(self, p):
        p = self.to_params(p)

        if p['sigma'] <= 0.0:
            return np.NINF

        denom = np.sum(p['sigma']*p['sigma'] + self.obs_dproxies*self.obs_dproxies)

        return np.log(p['sigma']) - np.log(denom) - np.log(np.abs(p['alpha']))

    def log_likelihood(self, p):
        p = self.to_params(p)
    
        proxy_mean = p['alpha'] * self.obs_masses + p['beta']
        proxy_std = np.sqrt(p['alpha']*p['alpha']*self.obs_dmasses*self.obs_dmasses + \
                            p['sigma']*p['sigma'] + \
                            self.obs_dproxies*self.obs_dproxies)

        return np.sum(ss.norm.logpdf(self.obs_proxies, loc=proxy_mean, scale=proxy_std))

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return lp
        else:
            return lp + self.log_likelihood(p)

    def mass_estimate(self, p, mobs, pobs, dmobs, dpobs):
        p = self.to_params(p)
        a = p['alpha']
        b = p['beta']
        sig = p['sigma']

        return (mobs*(sig*sig + dpobs*dpobs) + dmobs*dmobs*a*(pobs - b))/(a*a*dmobs*dmobs + sig*sig + dpobs*dpobs)

    def mass_variance_estimate(self, p, dmobs, dpobs):
        p = self.to_params(p)
        a = p['alpha']
        b = p['beta']
        sig = p['sigma']

        return dmobs*dmobs*(sig*sig + dpobs*dpobs) / (a*a*dmobs*dmobs + sig*sig + dpobs*dpobs)
