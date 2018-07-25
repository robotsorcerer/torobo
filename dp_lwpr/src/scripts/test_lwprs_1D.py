from __future__ import print_function

import os
from os.path import expanduser, join
import numpy as np
from lwpr import LWPR

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D

def test_lwpr_1D(s):
    data_path = join(expanduser('~'), 'Documents/StatisticalLearning/lwpr')
    # load training and testing data
    train = np.loadtxt(data_path + '/' + 'train.data')
    inds  = np.argsort(train[:,0])
    train = train[inds,:]
    X     = train[:,0]
    Y     = train[:,1]
    n     = len(X)

    print('n: ', n)
    test  = np.loadtxt(data_path + '/' + 'test.data')
    inds  = np.argsort(test[:,0])
    test  = test[inds,:]
    Xt    = test[:,0]
    Yt    = test[:,1]

    norm_in = (max(X) - min(X))/2
    norm_out = (max(Y) - min(Y))/2

    # initialize LWPR
    ID = 1
    if not s:
        print('initializing lwpr parameters')
        lwpr_obj = LWPR('Init',ID,1,1,0,0,0,1e-8,50,norm_in,norm_out,'lwpr_test')
    else:
        lwpr_obj = LWPR('Init',ID,s)

    # set some parameters
    kernel = 'Gaussian';
    # kernel = 'BiSquare'; % note: the BiSquare kernel requires different values for
    #                              an initial distance metric, as in the next line
    # LWPR('Change',ID,'init_D',[7 0.01; 0.01 7]);

    lwpr_obj.initializations('Change', [], ID=ID,  init_D=30);
    lwpr_obj.initializations('Change', [], ID=ID,  init_alpha=100);
    lwpr_obj.initializations('Change', [], ID=ID,  w_gen=0.5);             # more overlap gives smoother surfaces
    lwpr_obj.initializations('Change', [], ID=ID,  meta=1);                # meta learning can be faster, but numerical more dangerous
    lwpr_obj.initializations('Change', [], ID=ID,  meta_rate=100);
    lwpr_obj.initializations('Change', [], ID=ID,  init_lambda=0.995);
    lwpr_obj.initializations('Change', [], ID=ID,  final_lambda=0.9999);
    lwpr_obj.initializations('Change', [], ID=ID,  tau_lambda=0.99999);

    # train the model
    for j in range(100):
        inds = np.random.permutation(n)
        mse = 0
        for i in range(n):
            lwpr_obj.initializations('Update', ID, X[inds[i]], Y[inds[i]])
            yp, w, _  = lwpr_obj.output
            # print('yp: {}, w: {}'.format(yp, w))
            mse       = mse + (Y[inds[i]] - yp)**2;
        nMSE = mse/n/(np.var(Y)/len(Y));
        print('j: {}, #Data={} #rfs={} nMSE={} {}'
                .format(j, lwpr_obj.ID.n_data, len(lwpr_obj.ID.rfs), nMSE, '(TrainingSet)'))

    # create predictions for the test data
    Yp   = np.zeros(Yt.shape)
    Conf = np.zeros(Yt.shape)
    for i in range(len(Xt)):
        lwpr_obj = lwpr_obj.initializations(action='Predict',args=[ID,Xt[i].T,0.001])
        yp, w, conf = lwpr_obj.output

    Yp[0,i] = yp
    Conf[i,0] = conf

    ep   = Yt-Yp
    mse  = np.mean(ep**2)
    nmse = mse/(np.var(Y)/len(Y))
    print('#Data=%d #rfs=%d nMSE=%5.3f (TestSet)'.format(lwpr_obj.n_data,len(lwpr_obj.rfs),nmse))

    # get the data structure
    s = LWPR('Structure',ID);

    plt.close('all')
    f = plt.figure(figsize=(8, 8))
    ax = f.gca()

    # plot the raw noisy data
    ax.plot(Xt,Yt,'k',X[:,1],Y,'*',Xt,Yp,'r',Xt,Yp+Conf,'r:',Xt,Yp-Conf,'r:')
    ax.set_title('Noisy data samples and fitted function: nMSE={5:.3f}'.format(nmse))

    # plot the local models
    xmin  =  min(Xt)
    xmax  =  max(Xt)
    dx    =  (xmax - xmin)/200
    Xk    =  range(xmin, xmax, dx)
    for i in range(len(self.rfs)):
        N = 1/norm_in
        Yk=math.exp(-0.5*N*self.rfs[i].D*N*(Xk-self.rfs[i].c * norm_in) ** 2)
        ax.plot(Xk,Yk,'k')
    ax.set_title('RF Kernels')

    #save results.mat Xt Yp Conf -mat
    #
    # % --------------------------------------------------------------------------------
    # function [X,Y,Z]=makesurf(data,nx)
    # % [X,Y,Z]=makesurf(data,nx) converts the 3D data file data into
    # % three matices as need by surf(). nx tells how long the row of the
    # % output matrices are
    #
    # [m,n]=size(data);
    #
    # n=0;
    # for i=1:nx:m,
    # n = n+1;
    # X(:,n) = data(i:i+nx-1,1);
    # Y(:,n) = data(i:i+nx-1,2);
    # Z(:,n) = data(i:i+nx-1,3);
    # end;
    #
    #
    # % --------------------------------------------------------------------------------
    # function []=draw_ellipse(M,C,w,kernel)
    # % function draw ellipse draws the ellipse corresponding to the
    # % eigenvalues of M at the location c.
    #
    # [V,E] = eig(M);
    #
    # E = E;
    # d1 = E(1,1);
    # d2 = E(2,2);
    #
    # steps = 50;
    # switch kernel
    # case 'Gaussian'
    # start = sqrt(-2*log(w)/d1);
    # case 'BiSquare'
    # start = sqrt(2*(1-sqrt(w))/d1);
    # end
    #
    #
    # for i=0:steps,
    # Xp(i+1,1) = -start + i*(2*start)/steps;
    # switch kernel
    #  case 'Gaussian'
    #   arg = (-2*log(w)-Xp(i+1,1)^2*d1)/d2;
    #  case 'BiSquare'
    #   arg = (2*(1-sqrt(w))-Xp(i+1,1)^2*d1)/d2;
    # end
    # if (arg < 0),
    #   arg = 0;
    # end; % should be numerical error
    # Yp(i+1,1) = sqrt(arg);
    # end;
    #
    # for i=1:steps+1;
    # Xp(steps+1+i,1) = Xp(steps+1-i+1,1);
    # Yp(steps+1+i,1) = -Yp(steps+1-i+1,1);
    # end;
    #
    # % tranform the rf
    #
    # M = [Xp,Yp]*V(1:2,1:2)';
    #
    # Xp = M(:,1) + C(1);
    # Yp = M(:,2) + C(2);
    #
    # plot(C(1),C(2),'ro',Xp,Yp,'c');
    #
    # return s

if __name__ == '__main__':
    test_lwpr_1D([])
