from __future__ import print_function
import numpy as np

class LWPR(object):
	"""docstring for LWPR"""
	def __init__(self, arg):
		super(LWPR, self).__init__()
		self.arg  = arg
		self.lamb = 0.5


	def get_weight(self, x, D, kc, diag_only=True, kernel_type='Gaussian'):
		"""
			x: input data
			D: distance metric
			kc: kernel center
			kernel_type: Gaussian or Bisquare
		"""
		diff = np.zeros_like(x)
		weights = np.zeros_like(x)

		for i in range(len(x)):
			diff[i] = x[i] - kc
			if kernel_type=='Gaussian':
				weights[i] = np.exp(-0.5*diff.T.dot(D).dot(diff))
			elif kernel_type=='BiSquare':	
				if np.exp(-0.5*diff.T.dot(D).dot(diff)) > 1:
					weights[i] = 0
				else:
					weights[i] = 1.0 -0.5*diff.T.dot(D).dot(diff)

		return weights		


	def update_means(self, rf, x, y, w):
		W, beta, x_nut, u_nut = [np.zeros(x.shape)] * 4

		for i in range(len(x_nut)):
			W[i+1]     = self.lamb * W[i] + w
			x_nut[i+1] = (self.lamb * W[i] * x_nut[i] + w * x)/W[i+1]
			beta[i+1]  = (self.lamb * W[i] * beta[i] + w * y)/W[i+1]

		return x_nut, beta

	def update_local_models(self, x, y, r):
		u      = np.zeros_like(x)
		res    = np.zeros_like(x)

		# initializations
		z      = x
		res[0] = y - beta[0]

		for i in range(r):
			u[i+1]    = self.lamb * u[i] + w * z * res[i]
			s         = z.T.dot(u[i+1])
			SS[i+1]   = self.lamb * SS[i] + w * (s**2)
			SR[i+1]   = self.lamb * SR[i] + w * (s**2) * res[i]
			SZ[i+1]   = self.lamb * SZ[i] + w * z * (s**2)
			beta[i+1] = SR[i+1]/SS[i+1]




