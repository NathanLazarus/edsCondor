from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import pandas as pd
import numpy as np
# from numba import jit
from multiprocessing import Pool, freeze_support, cpu_count
# from statsmodels.tsa.arima_process import arma_generate_sample as AR_gen
from casadi import *
from scipy.special import h_roots
from polyfit_helper_funcs import *
from polyfit_helper_funcs import _vander_nd_flat
import sys

sim_num = str(17) #sys.argv[1]
T = 40000
burnin = 50
sigma_rho = 0.017
lambda_zeta = 0.92
delta = 0.08
alpha = 0.32
g = 1.014

beta = 0.98
eta = 2
P = 1
L_max = np.inf
L_min = 1e-6
C_min = 1e-6
K_min = 1e-9
degree = 6
# damp = 0.02
iter_count = 0
# k_coefs = np.array([3.60014515, 0.02656186, 1.91000997, -0.27549642, 5.0906649, -0.29205494, -0.21088586, -2.11306702, -0.01295193, 0.252112])

k_coefs = np.array([-2.52223679e-02,  3.14325706e-02,  5.01033193e-02, -4.21653511e-02,
                    -1.37994882e-01, -1.44272604e-02, -1.19342985e-02, -6.03401537e-02,
                     1.00500937e-01,  1.16229754e-01, -1.39680351e-01, -2.56880900e-01,
                    -1.89408316e-01, -1.05148594e-01,  2.54676148e-01,  1.59784452e-01,
                    -2.99204643e-01, -7.61316149e-03, -8.17626449e-02,  3.98571631e-01,
                    -9.45282186e-02,  2.46187662e-01,  1.10811578e-01, -8.35159959e-02,
                    -1.86926213e-01, -8.80258593e-02,  8.52046144e-02, -1.71162032e-04])
k_coefs_old = k_coefs


n_quadrature_nodes = 5
points, weights = h_roots(n_quadrature_nodes)
weights = weights/np.sqrt(np.pi)


# @jit(cache = True, nopython = True)
def integrationnodes(znow, lambda_zeta, nodes):
    # return np.outer(znow ** lambda_zeta, nodes ** sigma_rho).flatten(order = 'F')
    return np.outer(znow ** lambda_zeta, nodes ** sigma_rho).T.flatten()

# @jit(cache = True)
def k_function(k, z, k_coefs):
    k_poly = _vander_nd_flat(
        (hermevander,hermevander),
        [k.reshape((-1,1)),z.reshape((-1,1))],
        [degree,degree]
        ) @ k_coefs
    output_length = k_poly.shape[0]
    return fmax(k_poly,np.zeros(output_length)+K_min)


def condition1(consumption, labor, capital, zeta, capitalplus):
    return ((1/g)*(zeta*capital**alpha*labor**(1-alpha) + (1-delta) * capital - consumption) - capitalplus)


def condition3(consumption, labor, capital, zeta):
    return ((1/P)*(1-alpha)*zeta*capital**alpha*labor**(-alpha)
        - consumption*labor**eta) #length = T

def solve_for_endogenous_vars(know,kplus,zeta):
    n_periods = zeta.shape[0]
    consumption = SX.sym('consumption',n_periods,1)
    labor = SX.sym('labor',n_periods,1)

    objective = 1
    x_0 = DM.ones(vertcat(consumption,labor).shape[0])

    lower_bound_C = vertcat(DM.zeros(n_periods) + C_min)    # lower bound on the consumption -> not binding anyway
    upper_bound_C = vertcat(DM.zeros(n_periods) + np.inf)
    lower_bound_L = vertcat(DM.zeros(n_periods) + L_min)
    upper_bound_L = vertcat(DM.zeros(n_periods) + L_max) # upper bound on labor also doesn't bind

    nonlin_con = vertcat(condition1(consumption, labor, know, zeta, kplus),condition3(consumption, labor, know, zeta))
    nlp = {'x':vertcat(consumption, labor), 'f':objective, 'g':nonlin_con}
    solver = nlpsol('solver', 'ipopt', nlp,{'ipopt.print_level':0})
    with suppress_stdout_stderr():
        solution = solver(x0=x_0,lbx=vertcat(lower_bound_C,lower_bound_L),ubx=vertcat(upper_bound_C,upper_bound_L),lbg=-1e-6,ubg=1e-6)
    c, l = vertsplit(solution['x'],consumption.shape[0])
    return c, l


# @jit(cache = True, nopython = True)
def sqdist(x, y):
    return ((x - y) ** 2).sum(-1)

def eds(data_as_principal_components, epsilon):
    data = data_as_principal_components[:-1,:] #last row doesn't have a kplus value recorded
    candidate_points = np.hstack([
        np.arange(data.shape[0]).reshape(data.shape[0],1),
        data])
    inEDS = np.empty(candidate_points.shape)
    counter = 0
    while candidate_points.size > 0:
        addingtoEDS = candidate_points[0]
        inEDS[counter] = addingtoEDS
        candidate_points = candidate_points[
            sqdist(candidate_points[:,1:], addingtoEDS[1:]) > epsilon ** 2
        ]
        counter += 1
    else:
        inEDS = inEDS[0:counter]

    return inEDS


def eds_fixed_epsilon(
    data, epsilon = 0.05
):
    standardize = StandardScaler()
    normalized = standardize.fit_transform(data)
    pca = PCA(n_components = data.shape[1])
    as_principal_components = pca.fit_transform(normalized)
    indices, entries = np.hsplit(eds(as_principal_components, epsilon),np.array([1]))

    return indices

# if __name__ == "__main__":
#     with Pool(cpu_count()) as p:
#         pointstouse = p.map(
#             eds_fixed_epsilon,
#             (
#                 state_data
#             ),
#         )
#     print(pointstouse)

#     np.savetxt("eds_points.csv", np.vstack(pointstouse))

# solve for steady state
cstar = SX.sym('cstar')
lstar = SX.sym('lstar')
kstar = SX.sym('kstar')

obj = 1


def steadystateconstraint(cstar,lstar,kstar):
    c1 = cstar - (kstar**alpha*lstar**(1-alpha) + (1-delta)*kstar - g*kstar)
    c2 = lstar - (((1-alpha)/P)*kstar**alpha*(1/cstar))**(1/(eta+alpha))
    c3 = kstar - ((g/beta - (1 - delta))*(P/alpha))**(1/(alpha-1))*lstar
    return vertcat(c1,c2,c3)


starconstraint = steadystateconstraint(cstar,lstar,kstar)
star_x_0 = DM.ones(3)

star_nlp = {'x':vertcat(cstar,lstar,kstar), 'f':obj, 'g':starconstraint}
star_solver = nlpsol('star_solver', 'ipopt', star_nlp,{'ipopt.print_level':0})
with suppress_stdout_stderr():
    star_solution = star_solver(x0=star_x_0,lbg=-1e-14,ubg=1e-14)
ssc, ssl, ssk = vertsplit(star_solution['x'])
print(ssk, ssc, ssl)

k = np.zeros(T+1)

np.random.seed(int(sim_num)) #iter_count)
rho = np.random.randn(T)*sigma_rho
toZeta = [1]
for i in range(T):
    toZeta.append(toZeta[i]**lambda_zeta*np.exp(rho[i]))
zeta = np.array(toZeta[1:T+1])

k[0] = ssk
for i in range(T):
    k[i+1] = k_function(k[i],zeta[i],k_coefs)

state_data = np.vstack([
    k[:T],
    zeta]).T
state_data = state_data[burnin:,:]

indices = eds_fixed_epsilon(state_data[:,0:2],0.05)
indices = np.squeeze(indices.astype('int'))
n_eds_points = indices.shape[0]
know, znow = np.hsplit(state_data[indices],2)


zplus = integrationnodes(znow, lambda_zeta, np.exp(np.sqrt(2)*points))
k_plus_not_repeated = state_data[indices+1,0]
kplus = np.repeat(k_plus_not_repeated,n_quadrature_nodes)
cnow_not_repeated, lnow = solve_for_endogenous_vars(know, k_plus_not_repeated, znow)

cnow = np.repeat(cnow_not_repeated,n_quadrature_nodes)
kplusplus = k_function(kplus,zplus,k_coefs)

cplus, lplus = solve_for_endogenous_vars(kplus,kplusplus,zplus)

one_for_fixed_point_iteration = (((beta/g)*((1/P)* zplus * alpha*kplus**(alpha-1)* lplus**(1-alpha) + 1 - delta) )) #*(1/cplus))*cnow)

kplus_for_fixed_point_iteration = np.array(one_for_fixed_point_iteration*kplus).reshape((n_eds_points,n_quadrature_nodes),order = 'F') @ weights
howbout_this_one = np.array(one_for_fixed_point_iteration).reshape((n_eds_points,n_quadrature_nodes),order = 'F') @ weights

np.save('data' + sim_num + '.npy', np.hstack([x.reshape(-1,1) for x in [know, znow, kplus_for_fixed_point_iteration, k_plus_not_repeated]]))
