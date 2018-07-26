from __future__ import print_function

import os
from os.path import expanduser, join
import numpy as np
from lwpr import LWPR
import scipy.linalg as LAS

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D


def test_lwpr_nD(s,R):

  #test for LWPR on n-dimensional data set with 2D embedded function
  d = 5;
  n = 500;

  # a random training set using the CROSS function
  X = (np.random.rand(n,2)-.5)*2;
  Y = max(math.exp(-X[:,0]**2 * 10),math.exp(-X[:,1]**2 * 50),\
            1.25*math.exp(-(X[:,0]**2+X[:,1]**2)*5)).T;
  Y = Y.T + np.random.randn(n,1)*0.1;

  # rotate input data into d-dimensional space
  if not R:
    R = np.random.randn(d)
    R = R.T*R;
    R = LAS.orth(R)

  Xorg = X
  X = np.c_[X, np.zeros((n,d-1))].dot(R)

  # a systematic test set on a grid
  Xt = []
  for i in range(-1, 1, 0.05):
    for j in range(-1, 1, 0.05):
      Xt = np.r_[Xt, np.c_[i j]]
  Yt = max((math.exp(-Xt[:,0]**2 * 10),\
            math.exp(-Xt[:,1]**2 * 50),\
            1.25*math.exp(-(Xt[:,0]**2+Xt[:,1]**2)*5)).T);
  Yt = Yt.T;

  # rotate the test data
  Xtorg = Xt;
  Xt = np.c_[Xt, np.zeros(len(Xt), d-2)]*R;

  # initialize LWPR, use only diagonal distance metric
  ID = 1
  if not s:
      print('initializing lwpr parameters')
      lwpr_obj = LWPR('Init',ID,d,1,1,0,0,1e-7,50,ones(d,1),[1],'lwpr_test')
  else:
      lwpr_obj = LWPR('Init',ID,s)

  # set some parameters
  kernel = 'Gaussian';
  # kernel = 'BiSquare'; % note: the BiSquare kernel requires different values for
  #                              an initial distance metric, as in the next line
  # lwpr('Change',ID,'init_D',eye(d)*7);

  lwpr_obj.initializations('Change',[], ID, init_D=np.eye(d)*25)
  lwpr_obj.initializations('Change',[], ID, init_alpha=np.ones((d))*100)     # this is a safe learning rate
  lwpr_obj.initializations('Change',[], ID, w_gen=0.2)                  # more overlap gives smoother surfaces
  lwpr_obj.initializations('Change',[], ID, meta=1)                     # meta learning can be faster, but numerical more dangerous
  lwpr_obj.initializations('Change',[], ID, meta_rate=100)
  lwpr_obj.initializations('Change',[], ID, init_lambda=0.995)
  lwpr_obj.initializations('Change',[], ID, final_lambda=0.9999)
  lwpr_obj.initializations('Change',[], ID, tau_lambda=0.9999)


  # train the model
  for j in range(20):
    inds = np.random.permutation(n)
    mse = 0
    for i in range(n):
      yp,w,np = lwpr_obj.initializations(('Update',ID,X[inds[i],:].T,Y[inds[i],:].T)
      mse = mse + (Y[inds[i],:]-yp)**

    nMSE = mse/n/var(Y,1)
    print('#Data={} #rfs={} nMSE={} #proj={} (TrainingSet)'
      .format(lwpr_obj.ID.n_data,len(lwpr_obj.ID.rfs),nMSE,np));

  # create predictions for the test data
  Yp = np.zeros(Yt.shape);
  for i in range(len(Xt)):
    lwpr_obj = lwpr_obj.initializations(action='Predict',ID,Xt[i,:].T,0.001)
    yp, w, _=lwpr_obj.output
    Yp[i,0] = yp

  ep   = Yt-Yp
  mse  = np.mean(ep**2)
  nmse = mse/np.var(Y)
  print('#Data={} #rfs=%d nMSE={} #proj={} (TestSet)'
    .format(lwpr_obj.ID.n_data,len(lwpr_obj.ID.rfs),nmse,np))

  # get the data structure
  s  = LWPR('Structure',ID)

  plt.close('all')
  f  = plt.figure(figsize=(8, 8))
  ax = f.gca()

#   % plot the raw noisy data
#   subplot(2,2,1);
#   plot3(Xorg(:,1),Xorg(:,2),Y,'*');
#   title('Noisy data samples');
#
#   % plot the fitted surface
#   axis([-1 1 -1 1 -.5 1.5]);
#   subplot(2,2,2);
#   [x,y,z]=makesurf([Xtorg,Yp],sqrt(length(Xtorg)));
#   surfl(x,y,z);
#   axis([-1 1 -1 1 -.5 1.5]);
#   title(sprintf('The fitted function: nMSE=%5.3f',nmse));
#
#   % plot the true surface
#   subplot(2,2,3);
#   [x,y,z]=makesurf([Xtorg,Yt],sqrt(length(Xtorg)));
#   surfl(x,y,z);
#   axis([-1 1 -1 1 -.5 1.5]);
#   title('The true function');
#
#   % plot the local models
#   subplot(2,2,4);
#   for i=1:length(lwprs(ID).rfs),
#     D = R'*lwprs(ID).rfs(i).D*R;
#     c = R*lwprs(ID).rfs(i).c;
#     draw_ellipse(D(1:2,1:2),c(1:2),0.1,kernel);
#     hold on;
#   end
#   plot(Xorg(:,1),Xorg(:,2),'.');
#   hold off;
#   axis('equal');
#   title('Projected input space view of RFs');
#
# % --------------------------------------------------------------------------------
# function [X,Y,Z]=makesurf(data,nx)
# % [X,Y,Z]=makesurf(data,nx) converts the 3D data file data into
# % three matices as need by surf(). nx tells how long the row of the
# % output matrices are
#
#   [m,n]=size(data);
#
#   n=0;
#   for i=1:nx:m,
#     n = n+1;
#     X(:,n) = data(i:i+nx-1,1);
#     Y(:,n) = data(i:i+nx-1,2);
#     Z(:,n) = data(i:i+nx-1,3);
#   end;
#
#
# % --------------------------------------------------------------------------------
# function []=draw_ellipse(M,C,w,kernel)
# % function draw ellipse draws the ellipse corresponding to the
# % eigenvalues of M at the location c.
#
#   [V,E] = eig(M);
#
#   E = E;
#   d1 = E(1,1);
#   d2 = E(2,2);
#
#   steps = 50;
#   switch kernel
#    case 'Gaussian'
#     start = sqrt(-2*log(w)/d1);
#    case 'BiSquare'
#     start = sqrt(2*(1-sqrt(w))/d1);
#   end
#
#
#   for i=0:steps,
#     Xp(i+1,1) = -start + i*(2*start)/steps;
#     switch kernel
#      case 'Gaussian'
#       arg = (-2*log(w)-Xp(i+1,1)^2*d1)/d2;
#      case 'BiSquare'
#       arg = (2*(1-sqrt(w))-Xp(i+1,1)^2*d1)/d2;
#     end
#     if (arg < 0),
#       arg = 0;
#     end; % should be numerical error
#     Yp(i+1,1) = sqrt(arg);
#   end;
#
#   for i=1:steps+1;
#     Xp(steps+1+i,1) = Xp(steps+1-i+1,1);
#     Yp(steps+1+i,1) = -Yp(steps+1-i+1,1);
#   end;
#
#   % transform the rf
#
#   M = [Xp,Yp]*V(1:2,1:2)';
#
#   Xp = M(:,1) + C(1);
#   Yp = M(:,2) + C(2);
#
#   plot(C(1),C(2),'ro',Xp,Yp,'c');
#
#   return s, R
