from __future__ import print_function

import time
import math
import scipy
import logging
import numpy as np
import scipy.linalg as LAS
logger = logging.getLogger(__name__)

class Bundle(object):
	"""docstring for Bundle"""
	def __init__(self, dicko):
		super(Bundle, self).__init__()
		self.dicko = dicko
		for key, val in dicko.items():
			object.__setattr__(self, key, val)

	def __setattr__(self, key, value):
		object.__setattr__(self, key, value)

	def __repr__(self):
		s = repr(self.__dict__.keys())
		return s

class LWPRID(object):
	def __init__(self, *args, **kwargs):
		super(LWPRID, self).__init__()

		if args:
			self.ID = dict(ID = args[0])
		else:
			self.ID = dict(ID = kwargs['ID'])

		Bundle(self.ID)



class LWPR(object):
	"""docstring for LWPR"""
	def __init__(self, action='Init', *args, **kwargs):
		super(LWPR, self).__init__()

		self.first_time = True
		self.initializations(action, *args, **kwargs)

	def initializations(self, action, *args, **kwargs):

		if action == 'Init':
			self.ID             	= LWPRID(*args, **kwargs)
			self.ID.ID              = args[0]
			self.ID.n_in 		    = args[1]
			self.ID.n_out 		    = args[2]   #len(self.ID.y)
			self.ID.diag_only	    = args[3]   #diag_only
			self.ID.meta 		    = args[4]   #meta
			self.ID.meta_rate 	    = args[5]   #meta_rate
			self.ID.penalty 	    = args[6]   #penalty
			self.ID.init_alpha      = args[7]   #init_alpha
			self.ID.norm            = args[8]   #norm
			self.ID.norm_out        = args[9]   #norm_out
			self.ID.name            = args[10]

			# convy variables
			self.ID.n_data         = 0
			self.ID.w_gen          = 0.1
			self.ID.w_prune        = 0.9
			self.ID.init_lambda    = 0.999
			self.ID.final_lambda   = 0.9999
			self.ID.tau_lambda     = 0.99999
			self.ID.init_P         = 1
			self.ID.n_pruned       = 0
			self.ID.add_threshold  = 0.5

			# other variables
			self.ID.n_reg          = 2
			self.ID.init_D         = np.eye(self.ID.n_in)*25
			self.ID.init_M         = LAS.cholesky(self.ID.init_D)
			self.ID.init_alpha     = np.ones((self.ID.n_in, self.ID.n_in))*self.ID.init_alpha
			self.ID.mean_x         = np.zeros((self.ID.n_in,1))
			self.ID.var_x          = np.zeros((self.ID.n_in,1))
			self.ID.rfs            = []
			self.ID.kernel         = 'Gaussian' # can also be 'BiSquare'
			self.ID.max_rfs        = 1000
			self.ID.allow_D_update = 1
			self.ID.conf_method    = 'std' # can also be 't-test', but requires statistics toolbox

		elif action == 'Change':
			#self.ID     = LWPRID(None, **kwargs)
			# print('changing')
			if 'init_D' in kwargs:
				self.ID.init_D = kwargs['init_D']
			elif 'init_alpha' in kwargs:
				self.ID.init_alpha = kwargs['init_alpha']
			elif 'w_gen' in kwargs:
				self.ID.w_gen = kwargs['w_gen']
			elif 'meta' in kwargs:
				self.ID.meta = kwargs['meta']
			elif 'meta_rate' in kwargs:
				self.ID.meta_rate = kwargs['meta_rate']
			elif 'init_lambda' in kwargs:
				self.ID.init_lambda = kwargs['init_lambda']
			elif 'final_lambda' in kwargs:
				self.ID.final_lambda = kwargs['final_lambda']
			elif 'tau_lambda' in kwargs:
				self.ID.tau_lambda = kwargs['tau_lambda']
			self.ID.init_M = scipy.linalg.cholesky(self.ID.init_D)

		elif action == 'Update':
			self.ID.ID   = args[0]
			x            = args[1]
			y            = args[2]

			if len(args) > 4:
				composite_control = 1
				e_t               = args[3]
				alpha             = args[4]
			else:
				composite_control = 0

			# update global mean and variance of the training data for information purposes
			self.ID.mean_x = (self.ID.mean_x * self.ID.n_data + x) / (self.ID.n_data + 1)
			self.ID.var_x  = (self.ID.var_x * self.ID.n_data) + ((x - self.ID.mean_x) ** 2)/(self.ID.n_data + 1)
			self.ID.n_data +=  1

			# normalize the input
			xn = np.divide(x, self.ID.norm)

			# normalize the output
			yn = np.divide(y, self.ID.norm_out)

			if self.first_time:
				self.ID.rfs.append(self.init_rf(self.ID,[],xn,yn))

			self.first_time = False

			"""
				check all RFs updating
				wv is a vector of 3 weights, ordered [w; sec_w; max_w]
				iv is the corresponding vector containing the RF indices
			"""
			wv        = np.zeros((3))
			iv        = np.zeros((3))
			yp        = np.zeros((1)) if y.ndim is 0 else np.zeros(y.shape)

			sum_w     = 0
			sum_n_reg = 0
			tms       = []

			for i in range(len(self.ID.rfs)):
				# compute the weight and keep the three larget weights sorted
				w  = self.compute_weight(self.ID.diag_only,self.ID.kernel, \
										 self.ID.rfs[i].c,self.ID.rfs[i].D,xn)#
				self.ID.rfs[i].w = w.squeeze().tolist()
				wv[0]            = w.squeeze().tolist()
				iv[0]            = i
				ind              = np.argsort(wv)
				wv               = sorted(wv)
				iv               = np.array([int(x) for x in iv[ind]])

				sum_n_reg     = sum_n_reg + len(self.ID.rfs[i].s)

				# only update if activation is high enough
				if (w > 0.001):
					rf = self.ID.rfs[i]
					# update weighted mean for xn and y, and create mean-zero
					rf,xmz,ymz = self.update_means(self.ID.rfs[i],xn,yn,w)

					# update the regression
					rf,yp_i,e_cv,e = self.update_regression(rf,xmz,ymz,w)

					# print('yp: {}, yp_i: {}, w: {}'.format(yp, yp_i, w))
					if (rf.trustworthy):
					  yp    = w.dot(yp_i.T) + yp
					  sum_w += w

					# update simple statistical variables
					rf.sum_w  = rf.sum_w   * rf.lamb + w
					rf.n_data = rf.n_data  * rf.lamb + 1
					rf.lamb   = self.ID.tau_lambda * rf.lamb + \
								self.ID.final_lambda*(1.-self.ID.tau_lambda)

					# update the distance metric
					rf, tm = self.update_distance_metric(self.ID.ID, rf, xmz, \
									ymz,w,e_cv,e,xn);
					tms.append(1)

					# check whether a projection needs to be added
					rf = self.check_add_projection(self.ID.ID, rf)

					# incorporate updates
					if rf is not None:
						self.ID.rfs[i] = rf
				else:
					self.ID.rfs[i].w = 0
					tms.append(0)

			mean_n_reg = sum_n_reg/(len(self.ID.rfs)+1.e-10);
			tms = np.array(tms)
			# if LWPR is used for control, incorporate the tracking error
			if (composite_control):
			  inds = np.where(tms > 0)
			  # print('inds: ', inds)
			  if inds[0]:
				  for j in range(len(inds[0])):
					  i = inds[0][j]
					  self.ID.rfs[i].B  = self.ID.rfs[i].B  + alpha * tms[j] / self.ID.rfs[i].ss2 * self.ID.rfs[i].w/sum_w * (xn-self.ID.rfs[i].c) * e_t;
					  self.ID.rfs[i].b0 = self.ID.rfs[i].b0 + alpha * tms[j] / self.ID.rfs[i].sum_w[1] * self.ID.rfs[i].w/sum_w  * e_t;

			# do we need to add a new RF?
			# print('len(self.ID.rfs): ', len(self.ID.rfs), ' | self.ID.max_rfs: ', self.ID.max_rfs)
			if ((wv[2] <= self.ID.w_gen) and (len(self.ID.rfs)<self.ID.max_rfs)):
				if (	(wv[2] > 0.1*self.ID.w_gen ) and
						(self.ID.rfs[int(iv[2])].trustworthy)
						):
					# print('adding new rf cond 1.1')
					self.ID.rfs.append(self.init_rf(self.ID,self.ID.rfs[iv[2]],xn,yn))
				else:
					if len(self.ID.rfs)==0:
						self.ID.rfs = self.init_rf(self.ID.ID, [], xn, yn)
						# print('adding just one rf cond 2.1')
					else:
						# print('adding new rf anyway;  cond 2.2')
						self.ID.rfs.append(self.init_rf(self.ID,[],xn,yn))

			# do we need to prune a RF? Prune the one with smaller D
			if ((wv[1] > self.ID.w_prune) and (wv[2] > self.ID.w_prune)):
				if (sum(sum(self.ID.rfs[iv[1]].D)) > sum(sum(self.ID.rfs[iv[2]].D))):
					del self.ID.rfs[iv[1]]
					logger.debug('{}: Pruned #RF={}'.format(self.ID.ID,iv[1]))
				else:
					del self.ID.rfs[iv[2]]
					logger.debug('{}: Pruned #RF={}'.format(self.ID.ID,iv[2]))
				self.ID.n_pruned += 1

			# the final prediction
			if (sum_w > 0):
				# yp    = yp.squeeze() * self.ID.norm_out/sum_w.squeeze()
				yp    = yp * self.ID.norm_out/sum_w

			self.output       = yp, wv[2], mean_n_reg

		elif action == 'Predict':
			self.ID.ID     = args[0]
			x      		   = args[1]
			cutoff 		   = args[2]
			compute_conf   = 1

			# normalize the inputs
			xn = x/self.ID.norm

			# maintain the maximal activation
			max_w    = 0
			yp       = np.zeros((self.ID.n_out))
			sum_w    = 0
			sum_conf = 0
			sum_yp2  = 0

			for i in range(len(self.ID.rfs)):
			  # compute the weight
			  w  = self.compute_weight(self.ID.diag_only,self.ID.kernel,self.ID.rfs[i].c,self.ID.rfs[i].D,xn)
			  self.ID.rfs[i].w = w.squeeze()
			  max_w = max(max_w,w)

			  # only predict if activation is high enough
			  if (w > cutoff and self.ID.rfs[i].trustworthy):
				  # the mean zero input
				  xmz = xn - self.ID.rfs[i].mean_x

				  # compute the projected inputs
				  s, xres = self.compute_projection(xmz,self.ID.rfs[i].W,self.ID.rfs[i].U);

				  # the prediction
				  aux     = self.ID.rfs[i].B.T.dot(s) + self.ID.rfs[i].b0
				  yp      = yp + aux * w
				  sum_yp2 = sum_yp2 + aux**2 * w
				  sum_w   = sum_w + w

				  # confidence intervals if needed
				  if (compute_conf):
					  dofs = self.ID.rfs[i].sum_w[0]-self.ID.rfs[i].n_dofs
					  if self.ID.conf_method == 'std':
						  # print('self.ID.rfs[i].sum_e2: {}\n dofs:{} s: {}, self.ID.rfs[i].ss2: {}, w: {}'
						  #   .format(self.ID.rfs[i].sum_e2, dofs, s, self.ID.rfs[i].ss2, w))
						  sum_conf = sum_conf + w*self.ID.rfs[i].sum_e2/dofs*(1+(s/self.ID.rfs[i].ss2).T.dot(s).dot(w))
					  elif self.ID.conf_method ==  't-test':
						  sum_conf = sum_conf + scipy.stats.t.ppf(0.975,dofs)**2*w*self.ID.rfs[i].sum_e2/dofs*(1+(s/self.rfs[i].ss2).T.dot(s).dot(w))

			# the final prediction
			if (sum_w > 0):
				yp  	= yp/sum_w
				aux 	= (sum_yp2/sum_w - yp**2).dot(sum_w)/(sum_w**2)
				conf 	= np.sqrt(sum_conf/sum_w**2 + aux)
				yp 	    = yp * self.ID.norm_out
				conf 	= conf * self.ID.norm_out

			self.output      = yp, max_w

			if compute_conf:
				self.output = yp, max_w, conf

		elif action == 'Structure':
			self.ID = LWPRID(*args)
			self.output = self.ID

		elif action == 'Clear':
			self.ID = LWPRID(*args)

			self.ID  = None

	def init_rf(self, ID, template_rf, c, y):
		if template_rf:
			rf = template_rf
		else:
			rf = {
				'D': self.ID.init_D,
				'M': self.ID.init_M,
				'alpha': self.ID.init_alpha,
				'b0': y
			}

		# if more than univariate input, start with two projections such that
		# we can compare the reduction of residual error between two projections
		n_in 	= 	self.ID.n_in;
		n_out 	= 	self.ID.n_out;
		if (n_in > 1):
		  n_reg = 2
		else:
		  n_reg = 1

		rf_temp = {
					'B': np.zeros((n_reg, self.ID.n_out)),      # the regression parameters
					'c': c,                                     # the center of the '			'SXresYres'   : zeros(n_reg,n_in),         # needed to compute projections
					'SXresYres':  np.zeros((n_reg, n_in)),      # needed to compute projections
					'ss2': np.ones((n_reg,1))/self.ID.init_P, 		# variance per projection
					'SSYres': np.zeros((n_reg, n_out)),     # needed to compute linear model
					'SSXres': np.zeros((n_reg,n_in)),           # needed to compute input reduction
					'W': np.eye(n_reg,n_in),                    # matrix of projections vectors
					'Wnorm': np.zeros((n_reg,1)),               # normalized projection vectors
					'U': np.eye(n_reg,n_in),                    # reduction of input space
					'H': np.zeros((n_reg, n_out)),          # trace matrix
					'r': np.zeros((n_reg,1)),                   # trace vector
					'sum_w': np.ones((n_reg))*1.e-10,         # the sum of weights
					'sum_e_cv2i': np.zeros((n_reg)),          # weighted sum of cross.valid. err. per dim
					'sum_e_cv2': 0,                             # weighted sum of cross.valid. err. of final output
					'sum_e2': 0,                                # weighted sum of error (not CV)
					'n_data': np.ones((n_reg,1))*1.e-10,        # discounted amount of data in '			'trustworthy' : 0,                         # indicates statistical confidence
					'trustworthy': 0,                           # statistical confidence
					'lamb': np.ones((n_reg,1))*self.ID.init_lambda,# forgetting rate
					'mean_x': np.zeros((n_in,1)),               # the weighted mean of inputs
					'var_x': np.zeros((n_in,1)),                # the weighted variance of inputs
					'w': 0,                                     # store the last computed weight
					's': np.zeros((n_reg,1)),                   # store the projection of inputs
					'n_dofs': 0,                                # the local degrees of freedom
				}
		if isinstance(rf, dict):
			rf['h'] = np.zeros_like(rf['alpha']),        # a memory term for 2nd order gradients
			rf['b'] = math.log(rf['alpha']+1.e-10),      # a memory term for 2nd order gradients

			rf.update(rf_temp)
			return Bundle(rf)

		elif isinstance(rf, Bundle):
			rf.h  = np.zeros_like(rf.alpha),        # a memory term for 2nd order gradients
			rf.b  = math.log(rf.alpha+1.e-10),      # a memory term for 2nd order gradients

			return rf



	def compute_weight(self, diag_only, kernel_type, kc, D, x):
		"""
			x: input data
			D: distance metric
			kc: kernel center
			kernel_type: Gaussian or Bisquare
		"""

		diff = x - kc

		if diag_only:
			d2 = diff.T.dot(diag(D)*diff)
		else:
			d2 = diff.T * D * diff

		if kernel_type=='Gaussian':
			weights = np.exp(-0.5*d2)
		elif kernel_type=='BiSquare':
			if np.exp(-0.5*d2) > 1:
				weights = 0
			else:
				weights = 1.0 -(0.5*d2)**2

		return weights

	def update_means(self, rf, x, y, w):
		# print('rf: ', rf.sum_w)
		# print('w: {}, x: {}, rf.sum_w: {}, rf.lamb: {}'.format(w, x, rf.sum_w, rf.lamb))
		rf.mean_x  = (rf.sum_w.dot(rf.mean_x).dot(rf.lamb[0]) +
								 w * x)/(rf.sum_w.dot(rf.lamb[0]) + w)
		rf.var_x   = (rf.sum_w.dot(rf.var_x).dot(rf.lamb[0]) +
								 w * (x - rf.mean_x)**2)/(rf.sum_w.dot(rf.lamb[0]) + w)
		# print('rf.mean_x: ', rf.mean_x, rf.var_x, rf.b0, y)
		rf.b0      = (rf.sum_w * rf.b0 * rf.lamb[0] + w * y) / (rf.sum_w.dot(rf.lamb[0]) + w)

		xmz             = x - rf.mean_x
		ymz             = y - rf.b0

		return rf, xmz, ymz

	def update_regression(self, rf, x, y, w):
		# update linear regression parameters
		n_reg, n_in = rf.W.shape
		n_out       = len(y)

		# print('y: {}, n_out: {}'.format(y, n_out))

		rf.s, xres  = self.compute_projection(x, rf.W, rf.U)

		# compute all residual errors and targets at all projection stages
		yres  = rf.B * rf.s.dot(np.ones((1, n_out)))


		for i in range(1, n_reg):
		  yres[i,:] = yres[i,:] + yres[i-1,:]

		yres        = np.ones((n_reg,1)).dot(y.T) - yres
		# print('yres: ', yres)
		e_cv        = yres
		ytarget     = np.c_[y, yres[0, slice(0, n_reg)]]

		# update the projections
		lambda_slow  = 1 - (1- rf.lamb)/10;
		# print('rf.SXresYres: {}, ytarget: {}, w: {}| xres: {}'
		# 			.format(rf.SXresYres, ytarget, w, xres))
		rf.SXresYres = rf.SXresYres * (lambda_slow * np.ones((1,n_in))) + \
						w * np.sum(ytarget,axis=1) * np.ones((1,n_in))*xres
		rf.Wnorm     = np.sqrt(sum(rf.SXresYres ** 2, 2)) + 1.e-10
		rf.W         = np.divide(rf.SXresYres, rf.Wnorm)

		# update sufficient statistics for regressions
		rf.ss2       = (rf.lamb * rf.ss2) + (rf.s ** 2).dot(w);
		# print(' rf.lamb {}, rf.SSYres: {}, w: {}, ytarget: {}, rf.s: {}, n_out: {}'
		# 		.format(rf.lamb.shape, rf.SSYres.shape, w.shape, ytarget.shape, \
		# 			   rf.s.shape, n_out))
		rf.SSYres    = rf.lamb * np.ones((1,n_out)) * rf.SSYres + \
						w * ytarget * (rf.s * np.ones((1,n_out)))
		# print('rf.lamb {}, rf.SSXres: {}, xres: {}'
		#      .format(rf.lamb.shape, rf.SSXres.shape, xres.shape))
		rf.SSXres    = rf.lamb * np.ones((1,n_in))  * rf.SSXres + w * rf.s*np.ones((1,n_in)) * xres

		# update the regression and input reduction parameters
		# print('rf.ss2:  ', rf.ss2.shape)
		rf.B = np.divide(rf.SSYres, rf.ss2.dot(np.ones((1,n_out))))
		rf.U = np.divide(rf.SSXres, rf.ss2.dot(np.ones((1,n_in))))

		# the new predicted output after updating
		rf.s, xres = self.compute_projection(x, rf.W, rf.U)
		yp = rf.B.T.dot(rf.s)
		e  = y  - yp
		yp = yp + rf.b0

		# is the RF trustworthy: a simple data count
		# print('rf.n_data: {} n_in: {}, rf.lamb: {}'.format(rf.n_data, n_in, rf.lamb))
		if (rf.n_data > n_in*2):
		  rf.trustworthy = 1

		return rf, yp, e_cv, e

	def update_distance_metric(self, ID, rf, x,y,w,e_cv,e,xn):
		# update the distance metric

		# an indicator vector in how far individual projections are trustworthy
		# based on how much data the projection has been trained on
		derivative_ok = (rf.n_data[0][0] > 0.1/(1.- rf.lamb[0][0])) and rf.trustworthy

		# useful pre-computations: they need to come before the updates
		s                    = rf.s
		e_cv2                = np.sum(e_cv ** 2, 1)
		e2                   = e.T.dot(e)
		rf.sum_e_cv2i        = rf.sum_e_cv2i * rf.lamb    + w * e_cv2
		rf.sum_e_cv2         = rf.sum_e_cv2 * rf.lamb[0][0] + w * (e_cv2[-1])
		rf.sum_e2            = rf.sum_e2*(rf.lamb[0][0])    + w * e2
		rf.n_dofs            = rf.n_dofs*(rf.lamb[0][0])    + w**2*(s/rf.ss2).T * s
		e_cv                 = e_cv[-1,:].T
		e_cv2                = e_cv.T.dot(e_cv)
		h                    = w * sum(s** 2. / rf.ss2 * derivative_ok)
		W                    = rf.sum_w[0]
		E                    = rf.sum_e_cv2
		transient_multiplier = (e2/(e_cv2+1.e-10))**4 # this is a numerical safety heuristic

		if transient_multiplier > 1: # when adding a projection, this can happen
			transient_multiplier = 1

		n_out                = len(y);

		penalty   = self.ID.penalty/self.ID.n_in*w/W # normalize penality w.r.t. number of inputs
		meta      = self.ID.meta
		meta_rate = self.ID.meta_rate
		kernel    = self.ID.kernel
		diag_only = self.ID.diag_only

		# is the update permissible?
		# print('derivative_ok: ', derivative_ok)
		if (not derivative_ok ) or (not self.ID.allow_D_update):
			transient_multiplier = 0

		# the derivative dJ1/dw
		Ps    = s / rf.ss2 * derivative_ok;  # zero the terms with insufficient data support
		Pse   = Ps.dot(e.T)
		dJ1dw = -E/W**2 + 1/W*(e_cv2 - sum(sum((2*Pse)*rf.H)) - sum((2*Ps**2)*rf.r))

		# the derivatives dw/dM and dJ2/dM
		dwdM,dJ2dM,dwwdMdM,dJ2J2dMdM = self.dist_derivatives(w,rf, xn - rf.c,
															diag_only, kernel, penalty,meta)

		# the final derivative becomes (note this is upper triangular)
		dJdM = np.divide(dwdM.dot(dJ1dw), n_out) + dJ2dM

		# the second derivative if meta learning is required, and meta learning update
		if (meta):
		  # second derivatives
		  # print(W, e2, h, w, W.dot(e2).dot(h), 2/(W.dot(e2).dot(h)+1e-5)/w, 1/W**2 * (e_cv2-2*sum(sum(Pse*rf.H))), 2*E/W**3, )
		  dJ1J1dwdw = -e_cv2/W**2 - 2/W*sum(sum((-Pse/W -2*Ps*(s.T*Pse))*rf.H)) + \
						2/(W.dot(e2).dot(h)+1e-5)/w - 1/W**2 * (e_cv2-2*sum(sum(Pse*rf.H))) + 2*E/W**3

		  dJJdMdM = (dwwdMdM*dJ1dw + (dwdM**2).dot(dJ1J1dwdw))/n_out + dJ2J2dMdM

		  # update the learning rates
		  aux = meta_rate * transient_multiplier * (dJdM * rf.h)

		  # print('aux: ', aux)
		  # limit the update rate
		  ind = np.where(abs(aux) > 0.1)
		  if ind:
			  aux[ind] = 0.1*np.sign(aux[ind])

		  rf.b = rf.b - aux;

		  # prevent numerical overflow
		  ind = np.where(abs(rf.b) > 10);
		  if ind:
			  rf.b[ind] = 10*np.sign(rf.b[ind])

		  rf.alpha = math.exp(rf.b)

		  aux = 1 - (rf.alpha*dJJdMdM) * transient_multiplier
		  ind = np.where(aux < 0)

		  if ind:
			  aux[ind] = 0

		  rf.h = rf.h * aux - (rf.alpha * dJdM) * transient_multiplier

		# update the distance metric, use some caution for too large gradients
		maxM = max(max(abs(rf.M)));
		delta_M = rf.alpha*dJdM.dot(transient_multiplier);
		ind = np.where(delta_M > 0.1*maxM);
		if ind:
			rf.alpha = rf.alpha/2;
			delta_M[ind] = 0;
			logger.debug('Reduced learning rate')

		rf.M = rf.M - delta_M;
		rf.D = rf.M.T*rf.M;

		# update sufficient statistics: note this must come after the updates and
		# is conditioned on that sufficient samples contributed to the derivative
		H         = (rf.lamb.dot(np.ones((1,n_out))))*rf.H + (w/(1-h))*s*e_cv.T*transient_multiplier
		r         = rf.lamb*rf.r + (w**2*e_cv2/(1-h))*(s**2)*transient_multiplier
		rf.H = derivative_ok * np.ones((1,n_out)) *H + (1-derivative_ok * np.ones((1,n_out))) * rf.H
		rf.r = derivative_ok*r + (1-derivative_ok) * (rf.r)

		return rf, transient_multiplier

	def dist_derivatives(self, w,rf,dx,diag_only,kernel,penalty,meta):
		# compute derivatives of distance metric: note that these will be upper
		# triangular matrices for efficiency

		n_in      = len([dx]);
		dwdM      = np.zeros((n_in, n_in));
		dJ2dM     = np.zeros((n_in, n_in));
		dJ2J2dMdM = np.zeros((n_in, n_in));
		dwwdMdM   = np.zeros((n_in, n_in));

		for n in range(n_in):
		  for m in range(n, n_in):
			  sum_aux    = 0
			  sum_aux1   = 0

			  # take the derivative of D with respect to nm_th element of M */

			  if (diag_only and n==m):
				  aux = 2*rf.M[n, n];
				  dwdM[n,n] = dx[n]**2 * aux
				  sum_aux = rf.D[n,n].dot(aux)
				  if (meta):
					  sum_aux1 += aux**2

			  elif not diag_only:
				  for i in range(n,n_in):
					  # aux corresponds to the in_th (= ni_th) element of dDdm_nm
					  # this is directly processed for dwdM and dJ2dM
					  if (i == m):
						  aux =  2 * rf.M[n,i]
						  dwdM[n,m] = dwdM[n,m] + dx * dx *aux
						  sum_aux += rf.D * aux

						  if (meta):
							  sum_aux1 = sum_aux1 + aux**2
					  else:
						  aux = rf.M[n,i]
						  # dwdM[n,m] = dwdM[n,m] + 2 * dx[i].dot(dx[m]).dot(aux)
						  dwdM[n,m] = dwdM[n,m] + 2 * dx*(dx)*(aux)
						  sum_aux = sum_aux + 2 * self.rf.D[i,m]*(aux)
					  if (meta):
						  sum_aux1 = sum_aux1 + 2*aux**2

			  if kernel == 'Gaussian':
				  dwdM[n,m]  = -0.5*w*dwdM[n,m]
			  elif kernel == 'BiSquare':
				  dwdM[n,m]  = -np.sqrt(w)*dwdM[n,m]

			  dJ2dM[n,m]  = 2.* penalty.dot(sum_aux)

			  if (meta):
				  # dJ2J2dMdM[n,m] = 2 * penalty*(2*rf.D[m,m] + sum_aux1)
				  dJ2J2dMdM[n,m] = 2 * penalty*(2*rf.D + sum_aux1)
				  dJ2J2dMdM[m,n] = dJ2J2dMdM[n,m]
				  if kernel   ==  'Gaussian':
					  # dwwdMdM[n,m]   = dwdM[n,m]**2/w - w*dx[m]**2;
					  dwwdMdM[n,m]   = dwdM[n,m]**2/w - w*dx**2;
				  elif kernel ==  'BiSquare':
					  dwwdMdM[n,m]   = dwdM[n,m] ** 2/w/2 - 2*np.sqrt(w)*dx[m] ** 2
				  dwwdMdM[m,n]   = dwwdMdM[n,m]

		return dwdM,dJ2dM,dwwdMdM,dJ2J2dMdM

	def compute_projection(self,x,W,U):
		# recursively compute the projected input

		n_reg,n_in = W.shape

		s = np.zeros((n_reg,1))

		xres = np.zeros((n_reg, 1))
		# print('x: ', x, ' xres: ', xres)

		for i in range(n_reg):
		  # print('s: ', s, ' W: ', W, ' W[i].dot(x) ', W[i])
		  xres[i,:] = x.T
		  # s[i]      = W[i,:].dot(x)
		  s[i]      = W[i][0] * x
		  x         = x - U[i,:].T.dot(s[i])

		return s, xres

	def check_add_projection(self, ID, rf):
		# checks whether a new projection needs to be added to the rf

		n_reg,n_in  = rf.W.shape
		n_reg,n_out = rf.B.shape

		if (n_reg >= n_in):
		  return

		"""
			here, the mean squared error of the current regression dimension
			is compared against the previous one. Only if there is a signficant
			improvement in MSE, another dimension gets added. Some additional
			heuristics had to be added to ensure that the MSE decision is
			based on sufficient data
		"""

		n_reg           -= 1

		mse_n_reg   	= rf.sum_e_cv2i[n_reg]   / rf.sum_w[n_reg] + 1.e-10
		mse_n_reg_1 	= rf.sum_e_cv2i[n_reg-1]   / rf.sum_w[n_reg-1] + 1.e-10

		if (
			(mse_n_reg/mse_n_reg_1 < self.ID.add_threshold) and
			(rf.n_data[n_reg]/rf.n_data[0] > 0.99) and
			(rf.n_data[n_reg] * (1.-rf.lamb[n_reg]) > 0.5)
			):

		  rf.B         = np.r_[rf.B,  np.zeros((1,n_out))]
		  rf.SXresYres = np.r_[rf.SXresYres,  np.zeros((1,n_in))]
		  rf.ss2       = np.r_[rf.ss2, self.ID.init_P]
		  rf.SSYres    = np.r_[rf.SSYres, np.zeros((1,n_out))]
		  rf.SSXres    = np.r_[rf.SSXres, np.zeros((1,n_in))]
		  rf.W         = np.r_[rf.W, np.zeros((1,n_in))]
		  rf.Wnorm     = np.r_[rf.Wnorm, 0]
		  rf.U         = np.r_[rf.U, np.zeros((1,n_in))]
		  rf.H         = np.r_[rf.H, np.zeros((1,n_out))]
		  rf.r         = np.r_[rf.r, 0]
		  rf.sum_w     = np.r_[rf.sum_w, 1.e-10]
		  rf.sum_e_cv2i= np.r_[rf.sum_e_cv2i, 0]
		  rf.n_data    = np.r_[rf.n_data, 0]
		  rf.lamb      = np.r_[rf.lamb, self.ID.init_lambda]
		  rf.s         = np.r_[rf.s, 0]

		return rf
