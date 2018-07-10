from __future__ import print_function

import numpy as np
import scipy

class Bundle(object):
	"""docstring for Bundle"""
	def __init__(self, dicko):
		super(Bundle, self).__init__()
		for key, val in dicko.iteritems():
			object.__setattr__(self, key, val)

	def __setattr__(self, key, value):
	    if not hasattr(self, key):
	        raise AttributeError("%r has no attribute %s" % (self, key))
	    object.__setattr__(self, key, value)

		
class LWPR(object):
	"""docstring for LWPR"""
	def __init__(self, x, y, lamb = 0.5, diag_only=False, 
					meta=True, meta_rate=0.1, penalty=1e-4,
					init_alpha=[1,1]):
		super(LWPR, self).__init__()

		self.x 			    = x
		self.y 			    = y
		self.n_in 		    = len(self.x)
		self.n_out 		    = len(self.y)
		self.lamb 		    = lamb
		self.diag_only	    = diag_only
		self.meta 		    = meta
		self.meta_rate 	    = meta_rate
		self.penalty 	    = penalty
		self.init_alpha     = init_alpha

		# convy variables
        self.n_data         = 0
        self.w_gen          = 0.1
        self.w_prune        = 0.9
        self.init_lambda    = 0.999
        self.final_lambda   = 0.9999
        self.tau_lambda     = 0.99999
        self.init_P         = 1
        self.n_pruned       = 0
        self.add_threshold  = 0.5

      	# other variables
		self.n_reg          = 2
      	self.init_D         = np.eye(self.n_in)*25
      	self.init_M         = scipy.linalg.cholesky(self.init_D)
      	self.init_alpha     = self.init_alpha  #np.ones(self.n_in)*
      	self.mean_x         = np.zeros((self.n_in,1))
      	self.var_x          = np.zeros((self.n_in,1))
      	self.rfs            = []
      	self.kernel         = 'Gaussian' # can also be 'BiSquare'
      	self.max_rfs        = 1000
      	self.allow_D_update = 1
      	self.conf_method    = 'std' # can also be 't-test', but requires statistics toolbox


      	# fix c
      	self.init_rf(c, self.y)
      	# fix predict, update, and change conditions

	def init_rf(self, c, y):
		rf = {
			'D': self.init_D,
			'M': self.init_M,
			'alpha': init_alpha,
			'b0': y,
			'n_in': self.n_in,
			'n_out': self.n_out,
			# if (n_in > 1)
			'n_reg': 2,
			# else
			#   n_reg = 1,
			# end
			'B': np.zeros((self.n_reg,self.n_out)), # the regression parameters
			'c': c,                         # the center of the '			'SXresYres'   : zeros(n_reg,n_in),         # needed to compute projections
			'ss2': np.ones((self.n_reg,1))/self.init_P, # variance per projection
			'SSYres': np.zeros((self.n_reg,self.n_out)),        # needed to compute linear model
			'SSXres': np.zeros((self.n_reg,self.n_in)),         # needed to compute input reduction
			'W': np.eye(self.n_reg,self.n_in),           # matrix of projections vectors
			'Wnorm': np.zeros((self.n_reg,1)),            # normalized projection vectors
			'U': np.eye(self.n_reg,self.n_in),           # reduction of input space
			'H': np.zeros((self.n_reg,self.n_out)),        # trace matrix
			'r': np.zeros((self.n_reg,1)),            # trace vector
			'h': np.zeros_like(self.init_alpha),     # a memory term for 2nd order gradients
			'b': math.log(self.init_alpha+1.e-10),      # a memory term for 2nd order gradients
			'sum_w': np.ones(self.n_reg,1)*1.e-10,      # the sum of weights
			'sum_e_cv2i': np.zeros(self.n_reg,1),            # weighted sum of cross.valid. err. per dim
			'sum_e_cv2': 0,                         # weighted sum of cross.valid. err. of final output
			'sum_e2': 0,                         # weighted sum of error (not CV)
			'trustworthy': 0,    # statistical confidence
			'n_data': np.ones(self.n_reg,1)*1.e-10,      # discounted amount of data in '			'trustworthy' : 0,                         # indicates statistical confidence
			'lamb': np.ones(self.n_reg,1)*self.init_lambda, # forgetting rate
			'mean_x': np.zeros((self.n_in,1)),             # the weighted mean of inputs
			'var_x': np.zeros((self.n_in,1)),             # the weighted variance of inputs
			'w': 0,                         # store the last computed weight
			's': np.zeros((self.n_reg,1)),            # store the projection of inputs
			'n_dofs': 0,                         # the local degrees of freedom
			'mean_x': None,
			'var_x': None,
			's': None
		}
		self.rf  = Bundle(rf)

	def comnpute_weight(self, x, D, kc, diag_only=True, kernel_type='Gaussian'):
		"""
			x: input data
			D: distance metric
			kc: kernel center
			kernel_type: Gaussian or Bisquare
		"""
		diff = np.zeros_like(x)
		weights = np.zeros_like(x)

		diff = x - kc
		if kernel_type=='Gaussian':
			weights = np.exp(-0.5*diff.T.dot(D).dot(diff))
		elif kernel_type=='BiSquare':	
			if np.exp(-0.5*diff.T.dot(D).dot(diff)) > 1:
				weights = 0
			else:
				weights = 1.0 -0.5*diff.T.dot(D).dot(diff)

		return weights	

	def update_means(self, x, y, w):

		self.rf.mean_x  = (self.rf.sum_w.dot(self.rf.mean_x).dot(self.rf.lamb) +
								 w.dot(x))/(self.rf.sum_w.dot(self.rf.lamb) + w)
		self.rf.var_x   = (self.rf.sum_w.dot(self.rf.var_x).dot(self.rf.lamb) +
								 w.dot(x-self.rf.mean_x)**2)/(self.rf.sum_w.dot(self.rf.lamb) + w)
		self.rf.b0      = (self.rf.sum_w.dot(self.rf.b0).dot(self.rf.lamb) + w.dot(y)) / 
								 (self.rf.sum_w.dot(self.rf.lamb) + w)

		xmz             = x - self.rf.mean_x
		ymz             = y - self.rf.b0

		return xmz, ymz

	def update_regression(self, x, y, w):
		# update linear regression parameters
		n_reg, n_in = self.rf.W
		n_out       = len(y)

		self.rf.s, xres  = self.compute_projection(x, rf.W, rf.U)

		# compute all residual errors and targets at all projection stages
		yres  = self.rf.B * (self.rf.s * np.ones(1,n_out))
		for i in range(1, n_reg):
		  yres[i,:] = yres[i,:] + yres[i-1,:]

		yres        = np.ones(n_reg,1).dot(y.T) - yres
		e_cv        = yres
		ytarget     = np.concatenate((y.T, yres[:n_reg-1,:]), axis=0)

		# update the projections
		lambda_slow       = 1 - (1-self.rf.lamb)/10;
		self.rf.SXresYres = self.rf.SXresYres * (lambda_slow.dot(np.ones(1,n_in))) + w * (sum(ytarget,2)*np.ones(1,n_in))*xres;
		self.rf.Wnorm     = np.sqrt(sum(self.rf.SXresYres ** 2, 2)) + 1.e-10
		self.rf.W         = np.divide(self.rf.SXresYres, (self.rf.Wnorm * np.ones(1,n_in)))


		# update sufficient statistics for regressions
		self.rf.ss2       = (self.rf.lamb * self.rf.ss2) + (self.rf.s ** 2) * w;
		self.rf.SSYres    = (self.rf.lamb * np.ones(1,n_out)) * self.rf.SSYres + w * ytarget * (self.rf.s*np.ones(1,n_out))
		self.rf.SSXres    = (self.rf.lamb * np.ones(1,n_in))  * self.rf.SSXres + w * (self.rf.s*ones(1,n_in)) * xres

		# update the regression and input reduction parameters
		self.rf.B = np.divide(self.rf.SSYres, (self.rf.ss2*ones(1,n_out)))
		self.rf.U = np.divide(self.rf.SSXres, (self.rf.ss2*ones(1,n_in)))

		# the new predicted output after updating
		self.rf.s, xres = self.compute_projection(x, self.rf.W, self.rf.U)
		yp = self.rf.B.T * self.rf.s
		e  = y  - yp
		yp = yp + self.rf.b0

		# is the RF trustworthy: a simple data count
		if (self.rf.n_data > n_in*2):
		  self.rf.trustworthy = 1

		return yp, e_cv, e

	def update_distance_metric(x,y,w,e_cv,e,xn):

		# update the distance metric

		# an indicator vector in how far individual projections are trustworthy
		# based on how much data the projection has been trained on
		derivative_ok = (self.rf.n_data > 0.1/(1.- self.rf.lamb)) and self.rf.trustworthy

		# useful pre-computations: they need to come before the updates
		s                    = self.rf.s
		e_cv2                = sum(e_cv ** 2, 2)
		e2                   = e.T.dot(e)
		self.rf.sum_e_cv2i   = self.rf.sum_e_cv2i *self.rf.lamb    + w*e_cv2
		self.rf.sum_e_cv2    = self.rf.sum_e_cv2.dot(self.rf.lamb) + w*e_cv2[-1]
		self.rf.sum_e2       = self.rf.sum_e2.dot(self.rf.lamb)    + w*e2
		self.rf.n_dofs       = self.rf.n_dofs.dot(self.rf.lamb)    + w**2*(s/self.rf.ss2).T * s
		e_cv                 = e_cv[-1,:].T
		e_cv2                = e_cv.T * e_cv
		h                    = w * sum(s** 2. / self.rf.ss2 * derivative_ok)
		W                    = self.rf.sum_w
		E                    = self.rf.sum_e_cv2
		transient_multiplier = (e2/(e_cv2+1.e-10))**4 # this is a numerical safety heuristic
		if transient_multiplier > 1: # when adding a projection, this can happen
		  transient_multiplier = 1

		n_out                = len(y);

		penalty   = self.penalty/len(x)*w/W # normalize penality w.r.t. number of inputs
		meta      = self.meta
		meta_rate = self.meta_rate
		kernel    = self.kernel
		diag_only = self.diag_only

		# is the update permissible?
		if (not derivative_ok ) or (not self.allow_D_update:
		  transient_multiplier = 0

		# the derivative dJ1/dw
		Ps    = s / rf.ss2 * derivative_ok;  # zero the terms with insufficient data support
		Pse   = Ps * e.T
		dJ1dw = -E/W**2 + 1/W*(e_cv2 - sum(sum((2*Pse)*rf.H)) - sum((2*Ps**2)*self.rf.r))

		# the derivatives dw/dM and dJ2/dM
		dwdM,dJ2dM,dwwdMdM,dJ2J2dMdM = dist_derivatives(w,self.rf, xn - self.rf.c, \ 
											diag_only, kernel, penalty,meta)

		# the final derivative becomes (note this is upper triangular)
		dJdM = np.divide(dwdM.dot(dJ1dw), (n_out + dJ2dM))

		# the second derivative if meta learning is required, and meta learning update
		if (meta):
		  # second derivatives
		  dJ1J1dwdw = -e_cv2/W**2 - 2/W*sum(sum((-Pse/W -2*Ps*(s.T*Pse))*rf.H)) + \
		  				2/W*e2*h/w - 1/W**2*(e_cv2-2*sum(sum(Pse*self.rf.H))) + 2*E/W**3

		  dJJdMdM = (dwwdMdM*dJ1dw + dwdM**2*dJ1J1dwdw)/n_out + dJ2J2dMdM;

		  # update the learning rates
		  aux = meta_rate * transient_multiplier * (dJdM * self.rf.h);

		  # limit the update rate
		  ind = np.where(abs(aux) > 0.1);
		  if (~isempty(ind)),
		    aux(ind) = 0.1*sign(aux(ind));
		  end
		  rf.b = rf.b - aux;

		  % prevent numerical overflow
		  ind = find(abs(rf.b) > 10);
		  if (~isempty(ind)),
		    rf.b(ind) = 10*sign(rf.b(ind));
		  end

		  rf.alpha = exp(rf.b);

		  aux = 1 - (rf.alpha.*dJJdMdM) * transient_multiplier ;
		  ind = find(aux < 0);
		  if (~isempty(ind)),
		    aux(ind) = 0;
		  end

		  rf.h = rf.h.*aux - (rf.alpha.*dJdM) * transient_multiplier;

		end

		% update the distance metric, use some caution for too large gradients
		maxM = max(max(abs(rf.M)));
		delta_M = rf.alpha.*dJdM*transient_multiplier;
		ind = find(delta_M > 0.1*maxM);
		if (~isempty(ind)),
		  rf.alpha(ind) = rf.alpha(ind)/2;
		  delta_M(ind) = 0;
		  disp(sprintf('Reduced learning rate'));
		end

		rf.M = rf.M - delta_M;
		rf.D = rf.M'*rf.M;

		% update sufficient statistics: note this must come after the updates and
		% is conditioned on that sufficient samples contributed to the derivative
		H = (rf.lambda*ones(1,n_out)).*rf.H + (w/(1-h))*s*e_cv'*transient_multiplier;
		r = rf.lambda.*rf.r + (w^2*e_cv2/(1-h))*(s.^2)*transient_multiplier;
		rf.H = (derivative_ok*ones(1,n_out)).*H + (1-(derivative_ok*ones(1,n_out))).*rf.H;
		rf.r = derivative_ok.*r + (1-derivative_ok).*rf.r;



	def update_local_models(self, x, y, r):
		u      = np.zeros_like(x)
		res    = np.zeros_like(x)

		# update means of input and output
		W, beta, x_nut, u_nut = [np.zeros(x.shape)] * 4

		for i in range(len(x_nut)):
			W[i+1]     = self.lamb * W[i] + w
			x_nut[i+1] = (self.lamb * W[i] * x_nut[i] + w * x)/W[i+1]
			beta[i+1]  = (self.lamb * W[i] * beta[i] + w * y)/W[i+1]

		# initializations
		z      = x
		res[0] = y - beta[0]
		r = len(x)
		# now update the local models
		for i in range(r):
			u[i+1]    = self.lamb * u[i] + w * z * res[i]
			s         = z.T.dot(u[i+1])
			SS[i+1]   = self.lamb * SS[i] + w * (s**2)
			SR[i+1]   = self.lamb * SR[i] + w * s * res[i]
			SZ[i+1]   = self.lamb * SZ[i] + w * z * s
			beta[i+1] = SR[i+1]/SS[i+1]
			p[i+1]    = SZ[i+1]/SS[i+1]
			z         = z - s * p[i+1]
			MSE[i +1] = self.lamb * MSE[i] + w * (res[i+1] ** 2)

	def predict(self):
		# initializations
		y = beta
		z = x - x_nut

		for i in range(k):
			s = u[i].T.dot(z)
			y = y + beta.dot(s[i])
			z = z - s * p[i]

	def lwpr(self, xnew, ynew, rf):
		"""
			xnew -- new x data point
			ynew -- new y data point
			rf   -- receptive fields
		"""
		T = np.eye(len(xnew))
		xres = xnew
		yres = ynew

		for k in range(len(rf)):
			weight = self.get_weight(xnew, D, kc)

			xt = xres.dot(T)
			u[k] = xt.T.dot(yres)

			s[i] = xres.dot(u[i])
			beta[i] = s[i].T.dot(yres) / (s[i].T.dot(s[i]))

			yres = yres - s[i].dot(beta[i])

			p[i] = xres.T.dot(s[i])/(s[i].T.dot(s[i]))
			xres = xres - s[i].dot(p[i].T)





