import numpy as np
from casadi import *
from numpy.polynomial.hermite_e import *
import functools
import operator
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

def _nth_slice(i, ndim):
    sl = [np.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _vander_nd(vander_fs, points, degrees):
    
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}")
    if n_dims != len(degrees):
        raise ValueError(
            f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    # convert to the same shape and type
    # points = tuple(np.array(tuple(points), copy=False) + 0.0)

    # produce the vandermonde matrix for each dimension, placing the last
    # axis of each in an independent trailing axis of the output
    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    # we checked this wasn't empty already, so no `initial` needed
    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    """
    Like `_vander_nd`, but flattens the last ``len(degrees)`` axes into a single axis
    Used to implement the public ``<type>vander<n>d`` functions.

    I modified it to take the "upperleft" of the Vandermonde matrix.
    That is, for degree d, I include only polynomial terms whose coefficients sum to d.
    I exclude the terms whose coefficients are between d+1 and 2*d.
    """
    v = _vander_nd(vander_fs, points, degrees)
    v = v.reshape(v.shape[:-len(degrees)] + (-1,))
    if len(v.shape)>2.5:
        v = v.reshape(v.shape[0],v.shape[2])
    upperleft = np.zeros(((degrees[0]+1),(degrees[0]+1)),dtype=bool)
    for i in range(degrees[0]+1):
        upperleft[i,0:((degrees[0]+1)-i)]=True
    return v[0:v.shape[0],np.arange((degrees[0]+1)**2).reshape((degrees[0]+1),(degrees[0]+1))[upperleft]]
    

def herme2d_fit(x, y, deg, rcond=None, full=False, w=None, lambda_tikhonov = 0.01, max_iter = 2000, fitmethod = 'Tikhonov'):
    """
    Helper function used to implement the ``<type>fit`` functions.
    Parameters
    ----------
    vander_f : function(array_like, int) -> ndarray
        The 1d vander function, such as ``polyvander``
    c1, c2 :
        See the ``<type>fit`` functions for more detail
    """
    # x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    # if x.ndim != 1:
        # raise TypeError("expected 1D vector for x")
    # if x.size == 0:
    #     raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    # if len(x) != len(y):
        # raise TypeError("expected x and y to have same length")

    # if deg.ndim == 0:
    #     lmax = deg
    #     order = lmax + 1
    #     van = vander_f(x, lmax)

    # else:
    #     deg = np.sort(deg)
    #     lmax = deg[-1]
    #     order = len(deg)
    #     van = vander_f(x, lmax)[:, deg]
    van = _vander_nd_flat((hermevander,hermevander),x,[deg,deg])
    if fitmethod == 'LAD':
        lad_fit = QuantReg(y, van, q = 0.5, max_iter = max_iter).fit()
        sol = lad_fit.params
    if fitmethod == 'ElasticNet':
        e_n = ElasticNet(alpha = 0.001, l1_ratio = 0.01, max_iter = max_iter, warm_start = True, fit_intercept = False).fit(van, y)
        sol = e_n.coef_
    if fitmethod == 'Tikhonov':
        e_n = Ridge(alpha = lambda_tikhonov, max_iter = max_iter, fit_intercept = False).fit(van, y)
        sol = e_n.coef_

    # order = van.shape[1] #(deg+1)*(deg+2)/2


    # # set up the least squares matrices in transposed form
    # lhs = van.T
    # print(f"{lhs = }")
    # rhs = y.T
    # if w is not None:
    #     w = np.asarray(w) + 0.0
    #     if w.ndim != 1:
    #         raise TypeError("expected 1D vector for w")
    #     if len(x) != len(w):
    #         raise TypeError("expected x and w to have same length")
    #     # apply weights. Don't use inplace operations as they
    #     # can cause problems with NA.
    #     lhs = lhs * w
    #     rhs = rhs * w

    # # set rcond
    # # if rcond is None:
    # #     rcond = len(y)*np.finfo(y.dtype).eps

    # # Determine the norms of the design matrix columns.
    # if issubclass(lhs.dtype.type, np.complexfloating):
    #     scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    # else:
    #     scl = np.sqrt(np.square(lhs).sum(1))
    # scl[scl == 0] = 1
    # lhsT = lhs.T
    # if len(lhsT.shape)>2.5:
    #     lhsT = lhsT.reshape(lhsT.shape[0],lhsT.shape[2])
    #     scl = np.sqrt(np.sum(np.square(lhs),axis=2)).T
    #     lhsT_over_scl = lhsT #/scl
    # else:
    #     lhsT_over_scl = lhsT #/scl

    # # Solve the least squares problem.
    # # c, resids, rank, s = np.linalg.lstsq(lhsT_over_scl, rhs.T, rcond)
    # # c = (c.T/scl).T
    # beta = SX.sym('beta',lhsT_over_scl.shape[1],1)
    # if fitmethod == 'LAD':
    #     objective_fit = sum1(fabs(lhsT_over_scl@beta - rhs.T))
    # if fitmethod == 'Tikhonov':
    #     objective_fit = sum1((lhsT_over_scl@beta - rhs.T)**2) + lambda_tikhonov * sum1(beta**2)
    # nlp_fit = {'x':beta, 'f':objective_fit}
    # x_0_fit = DM.ones(beta.size1())
    # x_0_fit = DM([10.7,0.13,-0.06,-3.96,0.2,0.28])
    # solvr = nlpsol('solvr', 'ipopt', nlp_fit, {'ipopt.print_level':5,'ipopt.tol':1e-12,'ipopt.acceptable_tol':1e-12})
    # sol = solvr(x0=x_0_fit)['x']
    # my_resids = lhsT_over_scl@sol - rhs.T
    # alt_resids = lhsT_over_scl@c - rhs.T
    # my_sol = (sol/scl)
    # c = my_sol
    return sol #/scl # I don't really understand this
    # why divide by scl twice?


    # # Expand c to include non-fitted coefficients which are set to zero
    # if deg.ndim > 0:
    #     if c.ndim == 2:
    #         cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
    #     else:
    #         cc = np.zeros(lmax+1, dtype=c.dtype)
    #     cc[deg] = c
    #     c = cc

    # # warn on rank reduction
    # if rank != order and not full:
    #     msg = "The fit may be poorly conditioned"
    #     warnings.warn(msg, RankWarning, stacklevel=2)

    # if full:
    #     return c, [resids, rank, s, rcond]
    # else:
    #     return c


def hermevander_casadiSYM(x, deg):
    
    ideg = operator.index(deg)
    dims = (ideg + 1,) + x.shape
    v = SX.zeros(dims[:2])
    v[0,0:dims[1]] = 1
    if ideg > 0:
        v[1,0:dims[1]] = x.T
        for i in range(2, ideg + 1):
            v[i,0:dims[1]] = (v[i-1,0:dims[1]]*x.T - v[i-2,0:dims[1]]*(i - 1))
    return v.T

def vander_nd_flat_SYM(vander_fs, points, degrees):
    
    n_dims = len(vander_fs)
    v = SX.zeros(points[0].shape[0],(degrees[0]+1)**2)
    for i in range(points[0].shape[0]):
        v[i,0:(degrees[0]+1)**2] = reshape((vander_fs[0](points[0][i],degrees[0]).T@vander_fs[1](points[1][i],degrees[1])).T,1,(degrees[0]+1)**2)
    upperleft = np.zeros(((degrees[0]+1),(degrees[0]+1)),dtype=bool)
    for i in range(degrees[0]+1):
        upperleft[i,0:((degrees[0]+1)-i)]=True
    return v[0:v.shape[0],np.arange((degrees[0]+1)**2).reshape((degrees[0]+1),(degrees[0]+1))[upperleft]]

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
