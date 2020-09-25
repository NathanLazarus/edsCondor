import sys
import glob
import numpy as np
from polyfit_helper_funcs import herme2d_fit
from casadi import fabs

degree = 6
damp = 0.02

try:
    if sys.argv[1]=="initialize":
        initialize = True
    else:
        initialize = False
except:
    initialize = False

if initialize == True:
    k_coefs = np.array([-2.52223679e-02,  3.14325706e-02,  5.01033193e-02, -4.21653511e-02,
                        -1.37994882e-01, -1.44272604e-02, -1.19342985e-02, -6.03401537e-02,
                         1.00500937e-01,  1.16229754e-01, -1.39680351e-01, -2.56880900e-01,
                        -1.89408316e-01, -1.05148594e-01,  2.54676148e-01,  1.59784452e-01,
                        -2.99204643e-01, -7.61316149e-03, -8.17626449e-02,  3.98571631e-01,
                        -9.45282186e-02,  2.46187662e-01,  1.10811578e-01, -8.35159959e-02,
                        -1.86926213e-01, -8.80258593e-02,  8.52046144e-02, -1.71162032e-04])

else:
    data = np.vstack([np.load(x) for x in glob.glob('data*.npy')])
    know, znow, kplus_for_fixed_point_iteration, k_plus_not_repeated = np.hsplit(data, 4)
    k_coefs_new = herme2d_fit([know, znow], kplus_for_fixed_point_iteration, degree, lambda_tikhonov = 0.01, fitmethod = 'Tikhonov')
    k_coefs_old = np.load('k_coefs.npy')
    k_coefs = damp * k_coefs_new + (1-damp) * k_coefs_old
    
    
    print('err = {}'.format(np.mean(fabs(1-(kplus_for_fixed_point_iteration/k_plus_not_repeated)))))
    print(k_coefs)
    print(f'{data.shape[0] =}')

np.save('k_coefs.npy',k_coefs)