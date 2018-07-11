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
		# if not hasattr(self, key):
		#     raise AttributeError("%r has no attribute %s" % (self, key))
		object.__setattr__(self, key, value)


class LWPR(object):
	"""docstring for LWPR"""
	def __init__(self, action='Init', *args):
		# , lamb = 0.5, diag_only=False,
		# meta=True, meta_rate=0.1, penalty=1e-4,
		# init_alpha=[1,1], norm=None, norm_out = None
		super(LWPR, self).__init__()

		self.initializations(action, args)

	def initializations(self, action, args):

		if action == 'Init':
			self.ID             = args[0]
			self.n_in 		    = args[1]
			self.n_out 		    = args[2]   #len(self.y)
			self.diag_only	    = args[3]   #diag_only
			self.meta 		    = args[4]   #meta
			self.meta_rate 	    = args[5]   #meta_rate
			self.penalty 	    = args[6]   #penalty
			self.init_alpha     = args[7]   #init_alpha
			self.norm           = args[8]   #norm
			self.norm_out       = args[9]   #norm_out
			#self.lamb 		    = args[10]   #lamb
			self.name           = args[10]
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
			self.init_alpha     = np.ones(self.n_in)*self.init_alpha
			self.mean_x         = np.zeros((self.n_in,1))
			self.var_x          = np.zeros((self.n_in,1))
			self.rfs            = []
			self.kernel         = 'Gaussian' # can also be 'BiSquare'
			self.max_rfs        = 1000
			self.allow_D_update = 1
			self.conf_method    = 'std' # can also be 't-test', but requires statistics toolbox

		elif action == 'Change':
			self.ID     = args[0]
			self.init_M = scipy.linalg.cholesky(self.init_D)

		elif action == 'Update':
			self.ID = args[0]
			self.x  = args[1]
			self.y  = args[2]

			if len(args) > 4:
				self.composite_control = 1
				self.e_t               = args[3]
				self.alpha             = args[4]
			else:
				self.composite_control = 0

			# update global mean and variance of the training data for information purposes
			self.mean_x = self.mean_x.dot(self.n_data + x) / (self.n_data + 1)
			self.var_x  = self.var_x.dot(self.n_data) + ((x - self.mean_x) ** 2)/(self.n_data + 1)
			self.n_data = self.n_data + 1

			# normalize the input
			xn = np.divide(x, self.norm)

			# normalize the output
			yn = np.divide(y, self.norm_out)

			"""
				check all RFs updating
				wv is a vector of 3 weights, ordered [w; sec_w; max_w]
				iv is the corresponding vector containing the RF indices
			"""
			wv        = np.zeros((3,1))
			iv        = np.zeros((3,1))
			yp        = np.zeros(y.shape)
			sum_w     = 0
			sum_n_reg = 0
			tms       = np.zeros(len(self.rfs))

			for i in range(len(self.rfs)):
				# compute the weight and keep the three larget weights sorted
				w  = self.compute_weight(self.diag_only,self.kernel,self.rfs[i].c,self.rfs[i].D,xn);
				self.rfs[i].w = w
				wv[0]         = w
				iv[0]         = i
				ind           = np.argsort(wv)
				iv            = iv[ind]

				# keep track of the average number of projections
				sum_n_reg     = sum_n_reg + len(self.rfs[i].s)

				# only update if activation is high enough
				if (w > 0.001):

					rf = self.rfs[i]

					# update weighted mean for xn and y, and create mean-zero
					# variables
					rf,xmz,ymz = self.update_means(self.rfs[i],xn,yn,w)

					# update the regression
					rf,yp_i,e_cv,e = self.update_regression(rf,xmz,ymz,w)
					if (rf.trustworthy):
					  yp    = w.dot(yp_i) + yp
					  sum_w = sum_w + w

					# update simple statistical variables
					rf.sum_w  = rf.sum_w   * rf.lamb + w
					rf.n_data = rf.n_data  * rf.lamb + 1
					rf.lamb   = self.tau_lambda * rf.lamb + self.final_lambda*(1.-self.tau_lambda)

					# update the distance metric
					rf, tm = self.update_distance_metric(ID, rf, xmz,ymz,w,e_cv,e,xn);
					tms[i] = 1

					# check whether a projection needs to be added
					rf = self.check_add_projection(ID, rf)

					# incorporate updates
					self.rfs[i] = rf

				else:
					self.rfs[i].w = 0

			mean_n_reg = sum_n_reg/(len(self.rfs)+1.e-10);

			# if LWPR is used for control, incorporate the tracking error
			if (composite_control):
			  inds = np.where(tms > 0)
			  if not inds:
				  for j in range(len(inds[0])):
					  i = inds[j]
					  self.rfs[i].B  = self.rfs[i].B  + alpha * tms[j] / self.rfs[i].ss2 * self.rfs[i].w/sum_w * (xn-self.rfs[i].c) * e_t;
					  self.rfs[i].b0 = self.rfs[i].b0 + alpha * tms[j] / self.rfs[i].sum_w(1) * self.rfs[i].w/sum_w  * e_t;

			# do we need to add a new RF?
			if (wv[2] <= self.w_gen and len(self.rfs)<self.max_rfs):
				if (wv[2] > 0.1*self.w_gen and self.rfs[iv[2]].trustworthy):
					self.rfs[len(self.rfs)+1] = self.init_rf(ID,self.rfs[iv[2]],xn,yn);
				else:
					if (len(self.rfs)==0):
					    self.rfs = self.init_rf(ID,[],xn,y);
					else:
					    self.rfs[len(self.rfs)+1] = self.init_rf(ID,[],xn,yn)

			# do we need to prune a RF? Prune the one with smaller D
			if (wv[range(1,2)] > self.w_prune):
				if (sum(sum(self.rfs[iv[1]].D)) > sum(sum(self.rfs[iv[2]].D))):
					self.rfs[iv[1]] = []
					print('%d: Pruned #RF=%d'.format(ID,iv[2]))
				else:
					self.rfs[iv[2]] = []
					print('%d: Pruned #RF=%d'.format(ID,iv[2]))
				self.n_pruned += 1

			# the final prediction
			if (sum_w > 0):
				yp *= self.norm_out/sum_w

			self.output       = [yp, wv[2], mean_n_reg]

		elif action == 'Predict':
			ID     = args[0]
			x      = args[1]
			cutoff = args[2]

			if nargout == 3:
			  compute_conf = 1
			else:
			  compute_conf = 0

			# normalize the inputs
			xn = x/self.norm

			# maintain the maximal activation
			max_w    = 0
			yp       = np.zeros((self.n_out,1))
			sum_w    = 0
			sum_conf = 0
			sum_yp2  = 0

			for i in range(len(self.rfs)):

			  # compute the weight
			  w  = self.compute_weight(self.diag_only,self.kernel,self.rfs[i].c,self.rfs[i].D,xn)
			  self.rfs[i].w = w
			  max_w = max(np.c_[max_w,w])

			  # only predict if activation is high enough
			  if (w > cutoff and self.rfs[i].trustworthy):
				  # the mean zero input
				  xmz = xn - self.rfs[i].mean_x

				  # compute the projected inputs
				  s = self.compute_projection(xmz,self.rfs[i].W,self.rfs[i].U);

				  # the prediction
				  aux     = self.rfs[i].B.T.dot(s) + self.rfs[i].b0;
				  yp      = yp + aux * w;
				  sum_yp2 = sum_yp2 + aux**2 * w;
				  sum_w   = sum_w + w;

				  # confidence intervals if needed
				  if (compute_conf):
					  dofs = self.rfs[i].sum_w[0]-self.rfs[i].n_dofs
					  if self.conf_method == 'std':
						  sum_conf = sum_conf + w*self.rfs[i].sum_e2/dofs*(1+(s/self.rfs[i].ss2).T.dot(s).dot(w))
					  elif self.conf_method ==  't-test':
						  sum_conf = sum_conf + tinv(0.975,dofs)^2*w*self.rfs[i].sum_e2/dofs*(1+(s/self.rfs[i].ss2).T.dot(s).dot(w))

			# the final prediction
			if (sum_w > 0):
				yp  	= yp/sum_w
				aux 	= (sum_yp2/sum_w - yp**2).dot(sum_w)/(sum_w**2)
				conf 	= np.sqrt(sum_conf/sum_w**2 + aux)
				yp 	    = yp * self.norm_out
				conf 	= conf * self.norm_out

			self.

			output      = [yp, max_w]

			if compute_conf:
				self.output += [conf]

		elif action == 'Structure':
			ID = args[0]
			self.output = LWPR(ID)

		elif action == 'Clear':
			ID = args[0]

			self.ID  = []

	def init_rf(self, ID, template_rf, c, y):
		if template_rf:
			rf = template_rf
		else:
			rf = {
			'D': self.init_D,
			'M': self.init_M,
			'alpha': init_alpha,
			'b0': y
			}

		# if more than univariate input, start with two projections such that
		# we can compare the reduction of residual error between two projections
		n_in 	= 	self.n_in;
		n_out 	= 	self.n_out;
		if (n_in > 1):
		  n_reg = 2
		else:
		  n_reg = 1

		rf_temp = {
			'B': np.zeros((self.n_reg, self.n_out)), # the regression parameters
			'c': c,                         # the center of the '			'SXresYres'   : zeros(n_reg,n_in),         # needed to compute projections
			'SXresYres':  np.zeros((self.n_reg, self.n_in)),         # needed to compute projections
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
			'mean_x': 0,
			'var_x': 0,
			's': None
		}
		rf = rf.update(rf_temp)

		return Bundle(rf)


	def compute_weight(self, x, D, kc, diag_only=True, kernel_type='Gaussian'):
		"""
			x: input data
			D: distance metric
			kc: kernel center
			kernel_type: Gaussian or Bisquare
		"""
		diff = np.zeros_like(x)
		weights = np.zeros_like(x)

		diff = x - kc

		if self.diag_only:
			d2 = diff.T.dot(diag(D))*diff
		else:
			d2 = diff.T.dot(D).dot(diff)

		if kernel_type=='Gaussian':
			weights = np.exp(-0.5*d2)
		elif kernel_type=='BiSquare':
			if np.exp(-0.5*d2) > 1:
				weights = 0
			else:
				weights = 1.0 -(0.5*d2)**2

		return weights

	def update_means(self, rf, x, y, w):

		rf.mean_x  = (rf.sum_w.dot(rf.mean_x).dot(rf.lamb) +
								 w.dot(x))/(rf.sum_w.dot(rf.lamb) + w)
		rf.var_x   = (rf.sum_w.dot(rf.var_x).dot(rf.lamb) +
								 w.dot(x-rf.mean_x)**2)/(rf.sum_w.dot(rf.lamb) + w)
		rf.b0      = (rf.sum_w.dot(rf.b0).dot(rf.lamb) + w.dot(y)) / (rf.sum_w.dot(rf.lamb) + w)

		xmz             = x - rf.mean_x
		ymz             = y - rf.b0

		return rf, xmz, ymz

	def update_regression(self, x, y, w):
		# update linear regression parameters
		n_reg, n_in = self.rf.W
		n_out       = len(y)

		self.rf.s, xres  = self.compute_projection(x, rf.W, rf.U)

		# compute all residual errors and targets at all projection stages
		yres  = self.rf.B * (self.rf.s.dot(np.ones(1,n_out)))
		for i in range(1, n_reg):
		  yres[i,:] = yres[i,:] + yres[i-1,:]

		yres        = np.ones((n_reg,1)).dot(y.T) - yres
		e_cv        = yres
		ytarget     = np.r_[y.T, yres[:n_reg-1,:]]

		# update the projections
		lambda_slow       = 1 - (1- rf.lamb)/10;
		rf.SXresYres = rf.SXresYres * (lambda_slow.dot(np.ones(1,n_in))) + w.dot(sum(ytarget,2).dot(np.ones(1,n_in)))*xres;
		rf.Wnorm     = np.sqrt(sum(rf.SXresYres ** 2, 2)) + 1.e-10
		rf.W         = np.divide(rf.SXresYres, (rf.Wnorm.dot(np.ones(1,n_in))))

		# update sufficient statistics for regressions
		rf.ss2       = (rf.lamb * rf.ss2) + (rf.s ** 2).dot(w);
		rf.SSYres    = (rf.lamb.dot(np.ones(1,n_out))) * rf.SSYres + w.dot(ytarget) * (rf.s.dot(np.ones(1,n_out)))
		rf.SSXres    = (rf.lamb.dot(np.ones(1,n_in)))  * rf.SSXres + w.dot(rf.s*ones(1,n_in)) * xres

		# update the regression and input reduction parameters
		rf.B = np.divide(rf.SSYres, (rf.ss2.dot(np.ones(1,n_out))))
		rf.U = np.divide(rf.SSXres, (rf.ss2.dot(np.ones(1,n_in))))

		# the new predicted output after updating
		rf.s, xres = self.compute_projection(x, rf.W, rf.U)
		yp = rf.B.T.dot(rf.s)
		e  = y  - yp
		yp = yp + rf.b0

		# is the RF trustworthy: a simple data count
		if (rf.n_data > n_in*2):
		  rf.trustworthy = 1

		return rf, yp, e_cv, e

	def update_distance_metric(self, ID, rf, x,y,w,e_cv,e,xn):
		# update the distance metric

		# an indicator vector in how far individual projections are trustworthy
		# based on how much data the projection has been trained on
		derivative_ok = (rf.n_data > 0.1/(1.- rf.lamb)) and rf.trustworthy

		# useful pre-computations: they need to come before the updates
		s                    = rf.s
		e_cv2                = sum(e_cv ** 2, 2)
		e2                   = e.T.dot(e)
		rf.sum_e_cv2i        = rf.sum_e_cv2i * rf.lamb    + w.dot(e_cv2)
		rf.sum_e_cv2         = rf.sum_e_cv2.dot(rf.lamb[0]) + w.dot(e_cv2[-1])
		rf.sum_e2            = rf.sum_e2.dot(rf.lamb[0])    + w.dot(e2)
		rf.n_dofs            = rf.n_dofs.dot(rf.lamb[0])    + w**2*(s/rf.ss2).T * s
		e_cv                 = e_cv[-1,:].T
		e_cv2                = e_cv.T.dot(e_cv)
		h                    = w.dot(sum(s** 2. / rf.ss2 * derivative_ok))
		W                    = rf.sum_w[0]
		E                    = rf.sum_e_cv2
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
		if (not derivative_ok[0] ) or (not self.allow_D_update):
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
		  dJ1J1dwdw = -e_cv2/W**2 - 2/W*sum(sum((-Pse/W -2*Ps*(s.T*Pse))*rf.H)) + \
						2/W.dot(e2).dot(h)/w - 1/W**2 * (e_cv2-2*sum(sum(Pse*rf.H))) + 2*E/W**3

		  dJJdMdM = (dwwdMdM*dJ1dw + (dwdM**2).dot(dJ1J1dwdw))/n_out + dJ2J2dMdM

		  # update the learning rates
		  aux = meta_rate * transient_multiplier * (dJdM * rf.h)

		  # limit the update rate
		  ind = np.where(abs(aux) > 0.1);
		  if ind:
			  aux[ind] = 0.1*sign(aux[ind])

		  rf.b = rf.b - aux;

		  # prevent numerical overflow
		  ind = np.where(abs(rf.b) > 10);
		  if ind:
			  rf.b[ind] = 10*sign(rf.b[ind])

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
		    rf.alpha[ind] = rf.alpha[ind]/2;
		    delta_M[ind] = 0;
		    logger.debug('Reduced learning rate')

		rf.M = rf.M - delta_M;
		rf.D = rf.M.T*rf.M;

		# update sufficient statistics: note this must come after the updates and
		# is conditioned on that sufficient samples contributed to the derivative
		H         = (rf.lamb.dot(np.ones(1,n_out)))*rf.H + (w/(1-h))*s*e_cv.T*transient_multiplier
		r         = rf.lamb*rf.r + (w**2*e_cv2/(1-h))*(s**2)*transient_multiplier
		rf.H = (derivative_ok.dot(ones(1,n_out)))*H + (1-derivative_ok.dot(ones(1,n_out))) * rf.H
		rf.r = derivative_ok*r + (1-derivative_ok) * (rf.r)

		return transient_multiplier

	def dist_derivatives(self, w,rf,dx,diag_only,kernel,penalty,meta):
		# compute derivatives of distance metric: note that these will be upper
		# triangular matrices for efficiency

		n_in      = len(dx);
		dwdM      = np.zeros((n_in));
		dJ2dM     = np.zeros((n_in));
		dJ2J2dMdM = np.zeros((n_in));
		dwwdMdM   = np.zeros((n_in));

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
					  sum_aux1 = sum_aux1 + aux**2

			  elif not diag_only:
				  for i in range(n,n_in):
					  # aux corresponds to the in_th (= ni_th) element of dDdm_nm
					  # this is directly processed for dwdM and dJ2dM
					  if (i == m):
						  aux = 2*self.rf.M[n,i]
						  dwdM[n,m] = dwdM[n,m] + dx[i].dot(dx[m]).dot(aux);
						  sum_aux 	= sum_aux   + self.rf.D[i,m].dot(aux);
						  if (meta):
							  sum_aux1 = sum_aux1 + aux**2
					  else:
					     aux = self.rf.M[n,i]
					     dwdM[n,m] = dwdM[n,m] + 2 * dx[i].dot(dx[m]).dot(aux)
					     sum_aux = sum_aux + 2 * self.rf.D[i,m].dot(aux)
					  if (meta):
						   sum_aux1 = sum_aux1 + 2*aux**2

			  if kernel == 'Gaussian':
			  	  dwdM[n,m]  = -0.5*w*dwdM[n,m]
			  elif kernel == 'BiSquare':
			  	  dwdM[n,m]  = -np.sqrt(w)*dwdM[n,m]

			  dJ2dM[n,m]  = 2.* penalty.dot(sum_aux)

			  if (meta):
			      dJ2J2dMdM[n,m] = 2 * penalty*(2*self.rf.D[m,m] + sum_aux1)
			      dJ2J2dMdM[m,n] = dJ2J2dMdM[n,m]
			      if kernel   ==  'Gaussian':
				      dwwdMdM[n,m]   = dwdM[n,m]**2/w - w*dx[m]**2;
			      elif kernel ==  'BiSquare':
				      dwwdMdM[n,m]   = dwdM[n,m] ** 2/w/2 - 2*np.sqrt(w)*dx[m] ** 2
			      dwwdMdM[m,n]   = dwwdMdM[n,m]

		return dwdM,dJ2dM,dwwdMdM,dJ2J2dMdM

	def compute_projection(self,x,W,U):
		# recursively compute the projected input

		n_reg,n_in = W.shape

		s = np.zeros((n_reg,1))

		for i in range(n_reg):
		  xres[i,:] = x.T
		  s[i]      = W[i,:].dot(x);
		  x         = x - U[i,:].T.dot(s[i]);

		return s, xres

	def check_add_projection(self, rf):
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

		mse_n_reg   = rf.sum_e_cv2i[n_reg]   / rf.sum_w[n_reg] + 1.e-10;
		mse_n_reg_1 = rf.sum_e_cv2i[n_reg-1] / rf.sum_w[n_reg-1] + 1.e-10;

		if (mse_n_reg/mse_n_reg_1 < self.add_threshold and
			rf.n_data[n_reg]/rf.n_data[1] > 0.99 and
			rf.n_data[n_reg].dot(1.-rf.lamb[n_reg]) > 0.5):

		  rf.B         = np.r_[rf.B,  np.zeros((1,n_out))]
		  rf.SXresYres = np.r_[rf.SXresYres,  np.zeros((1,n_in))]
		  rf.ss2       = np.r_[rf.ss2, self.init_P]
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
		  rf.lamb      = np.r_[rf.lamb, self.init_lambda]
		  rf.s         = np.r_[rf.s, 0]

		return rf
