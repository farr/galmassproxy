Galaxy Mass Proxy Fitting
=========================

This code implements a "regression" on galaxy mass from observations
of mass and a mass proxy.  Using the regression posterior, subsequent
observations of the proxy can be converted to a posterior estimate of
the associated mass and its uncertainty.

Preliminaries
-------------

To use the code, you will need to install 

* [emcee](http://dan.iel.fm/emcee/current/), Dan Foreman-Mackey's
  excellent MCMC sampler.  The sampler is used to draw from the
  posterior of the regression model.

* [plotutils](https://github.com/farr/plotutils), a collection of
  routines to support managing emcee runs and the plotting of results.

Both of these libraries require [numpy](http://www.numpy.org/), and
plotutils requires [scipy](http://www.scipy.org/) and
[matplotlib](http://matplotlib.org/), too.

Basic Mode of Operation
-----------------------

First, one calibrates a mass-proxy relationship using 

       calibrate.py --data observational-data.dat --outdir output-directory

The file of observational data should be a text (or gzipped/bzipped
text) file with columns.  The first line is a header line, and should
contain at least the headers <code>mass</code>, <code>proxy</code>,
<code>dm</code>, and <code>dp</code> giving the measured mass and
proxy values, together with the (relative) observational uncertainties
in these quantities.  Various output files will be placed in the
indicated directory that will permit subsequent estimation of masses.
You may want to examine the "fit" displayed in <code>fit.pdf</code>,
which shows the data, the average "regression" line, and 10 draws from
the regression posterior.

Alternately, the package comes with several pre-calibrated mass-proxy
relationships, as described in
[arXiv:XXXX.XXXX](http://arxiv.org/abs/XXXX.XXXX); these can be found
in subdirectories of the <code>data</code> directory.

Once one has a directory of calbration results, observations of the
proxy only can be turned into mass estimates using 

      estimate.py --caldir calibration-directory --proxyfile proxy-data.dat --output mass-output.dat.bz2

The proxy data should be a text (or gzipped/bzipped text) file with
columns.  The first line is a header line, and should contain at least
the colums <code>proxy</code> and <code>dp</code> giving the observed
proxy values and (relative) uncertainties.  The results will be placed
in the output file (which will always be compressed with bzip2); each
line of that file contains draws from the posterior on the true mass
for each corresponding proxy measurement.  

The Model
---------

The model used for the regression is described in detail in
[arXiv:XXXX.XXXX](http://arxiv.org/abs/XXXX.XXXX), and implemented in
the extensively-commented <code>posterior.py</code>.  The model
assumes that the true masses and proxies are drawn from a correlated,
multivariate log-normal distribution; additionally, observed values
are assumed to distribute log-normally about the true values with
width equal to the quoted observational uncertainty.  The model fits
for the mean and covariance matrix of the log-normal, and then
combines the posterior on these parameters with subsequent
observations of proxy values to draw posterior estimates of masses.

The code in <code>plots.py</code> permits to generate plots
illustrating various aspects of the fit.

The code in <code>paper_test.py</code> implements the synthetic data
set and tests described in
[arXiv:XXXX.XXXX](http://arxiv.org/abs/XXXX.XXXX).

License
-------

This code is released under the [MIT
License](http://opensource.org/licenses/MIT).  See the file LICENSE
for details.