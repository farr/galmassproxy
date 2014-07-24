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
        return np.dtype([('mu', np.float, 2),
                         ('sigmas', np.float, 2),
                         ('theta', np.float)])

    @property
    def nparams(self):
        return 5

    @property
    def pnames(self):
        return [r'$\mu_M$', r'$\mu_x$', r'$\sigma_1$', r'$\sigma_2$', r'$\theta$']

    def to_params(self, p):
        return p.view(self.dtype).squeeze()

    def log_prior(self, p):
        p = self.to_params(p)

        if np.any(p['sigmas'] <= 0):
            return np.NINF

        if p['theta'] < 0 or p['theta'] >= np.pi/2.0:
            return np.NINF

        return 0.0

    def _mm_cov_matrix(self, p):
        p = self.to_params(p)

        t = p['theta']
        ct = np.cos(t)
        st = np.sin(t)

        R = np.array([[ct, -st], [st, ct]])

        cc = np.diag(p['sigmas']*p['sigmas'])

        return np.dot(R, np.dot(cc, R.T))

    def log_likelihood(self, p):
        p = self.to_params(p)
    
        mmc = self._mm_cov_matrix(p)

        dm2 = self.obs_dmasses*self.obs_dmasses
        dx2 = self.obs_dproxies*self.obs_dproxies

        denoms = mmc[0,0]*mmc[1,1] + mmc[0,0]*dx2 - mmc[0,1]*mmc[0,1] + mmc[1,1]*dm2 + dm2*dx2
        ci00 = (mmc[1,1]+dx2)/denoms
        ci01 = -mmc[0,1]/denoms
        ci11 = (mmc[0,0]+dm2)/denoms

        idets = ci00*ci11 - ci01*ci01

        mus = np.column_stack((self.obs_masses, self.obs_proxies)) - p['mu']

        return np.sum(0.5*(np.log(idets) - np.log(2.0*np.pi)) - 0.5*(ci00*mus[:,0]*mus[:,0] + 2.0*ci01*mus[:,0]*mus[:,1] + ci11*mus[:,1]*mus[:,1]))

    def pguess(self):
        p0 = self.to_params(np.zeros(self.nparams))

        pts = np.column_stack((self.obs_masses, self.obs_proxies))

        mu = np.mean(pts, axis=0)
        p0['mu'] = mu

        cov = np.cov(pts, rowvar=0)

        evals, evecs = np.linalg.eig(cov)

        theta1 = np.arctan(evecs[0,1]/evecs[0,0])
        theta2 = np.arctan(evecs[1,1]/evecs[1,0])

        if theta1 > 0:
            p0['sigmas'][0] = evals[0]
            p0['sigmas'][1] = evals[1]
            p0['theta'] = theta1
        else:
            p0['sigmas'][0] = evals[1]
            p0['sigmas'][1] = evals[0]
            p0['theta'] = theta2

        return p0.reshape((1,)).view(np.float).squeeze()

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return lp
        else:
            return lp + self.log_likelihood(p)

    def draw_data(self, p0, dms=None, dxs=None):
        if dms is None or dxs is None:
            dms = self.obs_dmasses
            dxs = self.obs_dproxies

        p0 = self.to_params(p0)

        mu = p0['mu']
        cov = self._mm_cov_matrix(p0)

        msxs = np.random.multivariate_normal(mu, cov, size=dms.shape[0])
        deltas = np.random.normal(loc=0, scale=np.column_stack((dms, dxs)))

        return msxs[:,0]+deltas[:,0], msxs[:,1]+deltas[:,1], dms, dxs

    def alpha(self, p):
        p = self.to_params(p)

        cm = self._mm_cov_matrix(p)

        evals, evecs = np.linalg.eigh(cm)

        imax = np.argmax(evals)
        v = evecs[:,imax]

        # m = alpha*x + b
        return v[0]/v[1]

    def mass_proxy_estimate(self, p, pobs, dpobs):
        p = self.to_params(p)
        cm = self._mm_cov_matrix(p)

        sp = dpobs*dpobs

        mt = p['mu'][0] + (pobs-p['mu'][1])*cm[1,0]/(sp+cm[1,1])
        xt = (p['mu'][1]*sp + pobs*cm[1,1])/(sp + cm[1,1])

        sm = cm[0,0] - cm[1,0]*cm[1,0]/(cm[1,1] + sp)
        sx = sp*cm[1,1]/(cm[1,1] + sp)

        return np.array([mt, xt]), np.array([sm, sx])
