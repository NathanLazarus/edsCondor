import sys
import numpy as np

try:
	sim_num = sys.argv[1]
except IndexError:
	print('Needs Sim Number\n')
	raise

coefs = np.load('coefs.npy')
ffs = np.broadcast_to(np.arange(2).reshape(2,1),(2,4))
data = coefs.reshape(1,4)+np.broadcast_to(np.arange(2).reshape(2,1),(2,4))+np.array([0,int(sim_num),0,0])

np.save('data'+sim_num+'.npy',data)