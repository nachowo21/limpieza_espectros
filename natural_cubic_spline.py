import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt

def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    """
    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The output data
    minval: float 
        Minimum of interval containing the knots.
    maxval: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p


class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self


class NaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.
    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.
    max: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError: # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl


def continuum(x,y,low_rej=1.8,high_re=0.0,niter=10,order=3,plots=False):
	# Rejection limits below and above the fit in units of the residual sigma. 
	# The number of knots can be used to control the amount of smoothness.
	#If low_reject and/or high_reject are greater than zero the sigma of the residuals between the fitted points and the fitted function is computed and those points whose residuals are less than -low_reject * sigma and greater than high_reject * sigma are excluded from the fit.
	#The function is then refit without the rejected points. This rejection procedure may be iterated a number of times given by the parameter niterate. This is how the continuum is determined. 
	wave=x
	flux=y
	w_int=wave
	f_int=flux
	w_excl=[]
	f_excl=[]
	for w in range(niter):
		model = get_natural_cubic_spline_model(w_int, f_int, minval=min(w_int), maxval=max(w_int), n_knots=order)
		y_est = model.predict(w_int)
		res=f_int-y_est
		med=np.median(res)
		sig=np.std(res)
		if (high_re>0) & (low_rej>0):
			i_out= (res>=high_re*sig) | (res<=-low_rej*sig)
		elif (high_re==0) & (low_rej>0):
			i_out=  (res<=-low_rej*sig)
		elif (high_re>0) & (low_rej==0):
			i_out=  res>=high_re*sig
		# Store rejected pixels
		w_excl.append(w_int[i_out])
		f_excl.append(f_int[i_out])
		# Redefine arrays for next model fitting
		w_int=w_int[~i_out]
		f_int=f_int[~i_out]
	w_excl=np.concatenate(w_excl)
	f_excl=np.concatenate(f_excl)
	
	if plots:
		print("Excluded: ",len(w_excl))
		plt.suptitle("Initial")
		plt.plot(wave, flux, color="black")
		plt.plot(wave,model.predict(wave),ls="--",color="red")
		plt.scatter(w_excl, f_excl, color="red",marker="x",s=20)
		plt.show()

		plt.suptitle("Normalized")
		plt.plot(wave,flux/model.predict(wave),color="black")
		plt.show()
	
	normalized_flux = flux/model.predict(wave)
	y_fit = model.predict(wave)
	return normalized_flux,y_fit,w_excl, f_excl

