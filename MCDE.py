# Main class for Markov Chain Density Estimator (MCDE)
# Based on paper: arXiv:XXXX
#
# Copyright (c) 2020: Andrea De Simone, Alessandro Morandini.

from __future__ import division
import numpy as np
import math
import scipy
import os
import mcint
from sklearn.neighbors import  KernelDensity

import random
from mpmath import hyp1f2


class MCDensityEstimator(object):
    """
    Density estimator using Markov Chains (optimized by using connection with KDE)
    """

    def __init__( self,
                  weight_func='gaussian',
                  bw=1.0,
                  interpolation_method='linear',
                  metric='euclidean',
                  normalize=True,
                  mov_bias=1.,
                  vol_ext=0,
                  kde_tol=1e-4
                  ):

        self.weight_func = weight_func
        self.bw = bw
        self.interpolation_method=interpolation_method
        self.metric=metric
        self.vol_ext=vol_ext
        self.kde_tol=kde_tol
        self.normalize=normalize
        self.mov_bias=mov_bias



        assert self.bw > 0.0, "Invalid argument. bw must be positive."
        assert self.vol_ext >= 0.0, "Invalid argument. vol_ext must be non negative."
        assert self.kde_tol > 0.0, "Invalid argument. kde_tol must be positive."


    def get_params(self, deep=True):
        return {"weight_func": self.weight_func,
                "bw": self.bw, "interpolation_method": self.interpolation_method, "metric": self.metric,
                "kde_tol": self.kde_tol, "vol_ext": self.vol_ext, "normalize": self.normalize,
                "mov_bias":self.mov_bias}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # calculate the translation term K(0)/Nh^D
    def trasl(self):
        
        if self.weight_func == 'gaussian':
            fac=(1/(2*math.pi))**(self.D/2.)
            return(fac/(self.N*self.bw**self.D))
        
        elif self.weight_func == 'exponential':
            fac=scipy.special.gamma(self.D+1.)*math.pi**(self.D/2.)/scipy.special.gamma(self.D/2.+1)
            return(1/(fac*self.N*self.bw**self.D))

        elif self.weight_func == 'epanechnikov':
            fac=(2./(self.D+2.))*math.pi**(self.D/2.)/scipy.special.gamma(self.D/2.+1)
            return(1/(fac*self.N*self.bw**self.D))
        
        elif self.weight_func == 'tophat':
            fac=math.pi**(self.D/2.)/scipy.special.gamma(self.D/2+1)
            return(1/(fac*self.N*self.bw**self.D))
        
        elif self.weight_func == 'linear':
            fac=(1./(self.D+1.))*math.pi**(self.D/2.)/scipy.special.gamma(self.D/2.+1.)
            return(1/(fac*self.N*self.bw**self.D))
        
        # does not work for even D
        elif self.weight_func == 'cosine':
            print('The cosine Kernel might have issues for even number of features\n')
            fac=np.float(hyp1f2(0.5+(self.D-1.)/2.,0.5,1.5+(self.D-1.)/2.,-math.pi**2/16.))*math.pi**(self.D/2.)/scipy.special.gamma(self.D/2.+1.)
            return(1/(fac*self.N*self.bw**self.D))
        
    
    # KDE is implemented through scikit-learn
    def kdefit(self):
        self.kde=KernelDensity(kernel=self.weight_func, bandwidth=self.bw, rtol=self.kde_tol).fit(self.x)

    # prob_kde returns the PDF estimated with KDE    
    def prob_kde( self, y):
        return(np.exp(self.kde.score_samples(np.array([y]).reshape(-1,self.D))))
    
    # function to be integrated in the case the KDE-extension is used
    def int_kde(self, y):
        return(self.prob_kde(y)-self.btrasl)*np.heaviside(self.prob_kde(y)-self.btrasl,0)

    def fit( self , X):
        #array must always be at least 2D to compute distance matrices
        self.x = np.asarray(X)
        if self.x.flatten().shape[0]/self.x.shape[0] == 1:
            self.x = self.x.reshape(self.x.shape[0],1)

        # Number of points in the sample
        self.N = self.x.shape[0]
        # Dimensionality of points
        self.D = self.x.shape[1]

        self.kdefit()
        self.btrasl=self.mov_bias*self.trasl()
        
        self.pi=self.prob_kde(self.x)-self.btrasl
        
        # normalize is an handle that can turn off normalization
        if self.normalize:
            self.estimate_pdf()
    

    def estimate_pdf( self ):
        # Un-normalized estimated probability density
        self.pdf=self.pi

        # Normalization through KDE-extension                  
        if self.interpolation_method=='kde':         
            self.integral, _ = mcint.integrate(self.int_kde,
                                                    self.sampler(),
                                                    measure=self.volume(),
                                                    n=200000 )



        else:
        # Normalization through an interpolation 
            if self.D==1:
                if self.interpolation_method=='linear':
                    self.interp = scipy.interpolate.interp1d(
                                                    self.x[:,0],
                                                    self.pdf,
                                                    fill_value=0.,
                                                    kind=self.interpolation_method )
                    ind=np.argsort(self.x,axis=0)[:,0]

                    self.integral=np.trapz(x=self.x[ind].flatten(),y=self.pdf[ind])
                
                elif self.interpolation_method=='nearest':
                    self.interp = scipy.interpolate.interp1d(
                                                    self.x[:,0],
                                                    self.pdf,
                                                    fill_value=0.,
                                                    kind=self.interpolation_method )
                    self.integral, _ = mcint.integrate( self.interp,
                                                    self.sampler(),
                                                    measure=self.volume(),
                                                    n=200000 )
                
                else:
                    print('Invalid norm func!\n\n')




            elif self.D>=2:

                if self.interpolation_method=='linear':
                    self.interp = scipy.interpolate.LinearNDInterpolator(
                                            self.x, self.pdf, fill_value=0.0 )
                elif self.interpolation_method=='nearest':
                    self.interp = scipy.interpolate.NearestNDInterpolator(
                                                            self.x, self.pdf )
                else:
                    print('Invalid norm func!\n\n')

                self.integral, _ = mcint.integrate( self.interp,
                                                    self.sampler(),
                                                    measure=self.volume(),
                                                    n=200000 )


        # Normalized estimated probability density
        self.pdf = self.pdf / self.integral


        # combine estimated prob with data into Nx(D+1) array [x_i, p_hat(x_i)]
        self.pX = np.append(self.x, self.pdf.reshape((self.N,1)), axis=1)



    # sampler and volume defined for the integration
    def sampler( self ):
        while True:
            gen_list = list()
            for i in range(self.D):
                r = random.uniform( self.x.T[i].min()-self.vol_ext*self.bw, self.x.T[i].max()+self.vol_ext*self.bw)
                gen_list.append(r)
            yield (gen_list)


    def volume( self ):
        vol = 1.0
        for i in range(self.D):
            vol = vol * ( self.x.T[i].max() + 2*self.vol_ext*self.bw - self.x.T[i].min() )
        return( vol )


    # the estimator can be used to evaluate the probability in new points
    # through evaluate_pdf
    # the interpolation can either be linear/nearest or extended with KDE
    def evaluate_pdf( self, query ):
        if self.interpolation_method=='kde':         
            py = (self.prob_kde(query)-self.trasl())*np.heaviside(self.prob_kde(query)-self.trasl(),0)/ self.integral
        else:
            py = self.interp.__call__(query)/self.integral
        return(py)    
