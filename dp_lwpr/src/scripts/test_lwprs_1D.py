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
    for j in range(10):
        inds = np.random.permutation(n)
        mse = 0
        for i in range(n):
            lwpr_obj.initializations('Update', ID, X[inds[i]], Y[inds[i]])
            yp, w, _  = lwpr_obj.output
            mse       = mse + (Y[inds[i]] - yp)**2;
        nMSE = mse/n/(np.var(Y)/len(Y));
        print('j: {} #Data={} #rfs={} nMSE={} {}'
                .format(j, lwpr_obj.ID.n_data, \
                 len(lwpr_obj.ID.rfs), nMSE, \
                 '(TrainingSet)'))

    # create predictions for the test data
    Yp   = np.zeros(Yt.shape)
    Conf = np.zeros(Yt.shape)
    for i in range(len(Xt)):
        lwpr_obj.initializations('Predict', ID, Xt[i].T, 0.001)
        yp, w, conf = lwpr_obj.output

        print('Yp, {} yp: {} w: {}, conf: {}'.format(Yp.shape, yp, w, conf))
        Yp[i] = yp
        Conf[i] = conf

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
    ax.plot(Xt,Yt,'k',X[:,1],Y,'*',Xt,Yp,'r',\
            Xt,Yp+Conf,'r:',Xt,Yp-Conf,'r:')
    ax.set_title('Noisy data samples and fitted function: nMSE={5:.3f}'.format(nmse))
    plt.show()

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
    plt.show()

    #save results.mat Xt Yp Conf -mat
    #
    # % --------------------------------------------------------------------------------
    def makesurf(data,nx):
        # [X,Y,Z]=makesurf(data,nx) converts the 3D data file data into
        # three matices as need by surf(). nx tells how long the row of the
        # output matrices are
        m,n=data.shape

        n=0
        for i in range(m, nx): #1:nx:m,
            n += 1
            X[:,n] = data[i:i+nx-1,0]
            Y[:,n] = data[i:i+nx-1,1]
            Z[:,n] = data[i:i+nx-1,2]
        return X, Y, Z

    # --------------------------------------------------------------------------------
    def draw_ellipse(M,C,w,kernel):
        # function draw ellipse draws the ellipse corresponding to the
        # eigenvalues of M at the location c.

        E, V = np.linalg.eig(M)

        E = E;
        d1 = E[0,0];
        d2 = E[1,1];

        steps = 50;
        if kernel == 'Gaussian':
            start = np.sqrt(-2*math.log(w)/d1)
        elif kernel == 'BiSquare':
            start = np.sqrt(2*(1-np.sqrt(w))/d1)

        for i in range(steps):
            Xp[i+1,1] = -start + i*(2*start)/steps
        if kernel == 'Gaussian':
          arg = (-2*math.log(w)-Xp[i+1,1]**2*d1)/d2
        elif kernel == 'BiSquare':
          arg = (2*(1-np.sqrt(w))-Xp[i+1,1]**2*d1)/d2
        if (arg < 0):
          arg = 0
        Yp[i+1,1] = np.sqrt(arg)

        for i in range(steps+1):
            Xp[steps+1+i,1] = Xp(steps+1-i+1,1)
            Yp[steps+1+i,1] = -Yp(steps+1-i+1,1)

        # tranform the rf
        M = np.c_[Xp,Yp]*V[0:1,0:1].T;

        Xp = M[:,0] + C[0]
        Yp = M[:,1] + C[1]


        f = plt.figure(figsize=(8, 8))
        ax = f.gca()
        ax.plot(C[0],C[1],'ro',Xp,Yp,'c')

if __name__ == '__main__':
    test_lwpr_1D([])
