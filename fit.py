import sys
import numpy as np


try:
	if sys.argv[1]=="initialize":
		initialize = True
	else:
		initialize = False
except:
	initialize = False


if initialize == True:
	coefs = np.array([0,0,0,1])

else:
	data0 = np.load('data0.npy')
	data1 = np.load('data1.npy')
	coefs = np.sum(data0+data1,0)

np.save('coefs.npy',coefs)

if np.max(coefs)>100:
	print(coefs)
	sys.exit(33)

# print(initialize)
# coefs.npy
