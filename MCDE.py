# Main class for Markov Chain Density Estimator (MCDE)
# Based on paper: arXiv:XXXX
#
# Copyright (c) 2020: Andrea De Simone, Alessandro Morandini.

import numpy as np
import scipy
import os
import mcint
from sklearn.preprocessing import normalize
import random



class MCDensityEstimator(object):
    """
    Density estimator using Markov Chains.
    """

    def __init__( self,
                  weight_func='exp2',
                  beta=1.0,
                  max_iter=10000,
                  rtol = 1e-12,
                  interpolation_method='nearest',
                  metric='euclidean'
                  ):
        """
        Initialize instance of the estimator class.

        X: input data set
        weight_func: weighting function. Possible choices:
                    'exp': exp(-beta*d)
                    'exp2': exp(-beta*d^2)
        beta: distance-weighting parameter of the exponential.
        max_iter: max number of iterations
        rtol: relative tolerance to find principal eigenvalue = 1.0
        interpolation_method: interpolation used to normalize the PDF and predict values (nearest or linear)
        metric: distances are calculated based on this distance measure
                    (see scipy.spatial.distance.pdist for the available metrics)
        """

        self.weight_func = weight_func
        self.beta = beta
        self.max_iter = max_iter
        self.rtol = rtol
        self.interpolation_method=interpolation_method
        self.metric=metric

        #assert self.prec >=1  , "Invalid argument. prec must be >=1."
        assert self.beta > 0.0, "Invalid argument. beta must be positive."
        assert self.max_iter >= 1, "Invalid argument. max_iter must be >= 1."
        assert self.rtol > 0.0 , "Invalid argumento. rtol must be positive."

    def get_params(self, deep=True):
        return {"weight_func": self.weight_func,
                "beta": self.beta, "max_iter":self.max_iter,
                "rtol": self.rtol,"interpolation_method": self.interpolation_method, "metric": self.metric}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit( self , X):
        """
        Compute the NxN matrices:
        Distance matrix (self.d_matrix),
        Weight matrix (self.W_matrix),
        Transition probability matrix (self.Q_matrix)
        """

        #array must always be at least 2D to compute distance matrices
        self.x = np.asarray(X)
        if self.x.flatten().shape[0]/self.x.shape[0] == 1:
            self.x = self.x.reshape(self.x.shape[0],1)

        # Number of points in the sample
        self.N = self.x.shape[0]
        # Dimensionality of points
        self.D = self.x.shape[1]

        #---
        #print("Computing matrices ...", end=' ')

        # Compute Distance matrix
        self.d_matrix = scipy.spatial.distance.squareform(
                            scipy.spatial.distance.pdist(
                                                    self.x,
                                                    metric=self.metric) )

        # Compute Weight matrix
        if self.weight_func == 'exp':
            self.W_matrix = np.exp(-self.d_matrix*self.beta)

        elif self.weight_func == 'exp2':
            self.W_matrix = np.exp(-np.square(self.d_matrix)*self.beta)


        else:
            print("invalid weight function")

        # fill diagonal with 0
        np.fill_diagonal(self.W_matrix, 0.0)


        # Compute Transition probability matrix
        # (Each row is normalized to 1)
        self.Q_matrix = normalize(self.W_matrix, axis=1, norm='l1')

        # Find stationary distribution (with power iteration method)
        evec_found = False
        Qt = np.transpose(self.Q_matrix)

        # start with simple guess (normalized to sum to 1)
        new_vec = np.mean(self.W_matrix,axis=0)
        new_vec = new_vec / np.sum(new_vec)

        # Loop over tolerances (each time it is increased by a factor of 10)
        while self.rtol <= 1e-5:

            # Loop of matrix*vector products
            #  until |Principal Eigenvalue - 1| < self.rtol
            for i in range(self.max_iter):

                last_vec = np.copy(new_vec)

                new_vec =  np.dot(Qt, last_vec)
                new_vec = new_vec / np.sum(new_vec)
                eigval = np.dot(last_vec, new_vec)/np.dot(last_vec, last_vec)

                # stop when new vector is close to previous one
                if np.isclose(eigval, 1.0, atol=self.rtol, rtol=0):
                    evec_found = True
                    break  # exit for loop

            if evec_found:
                self.pi = new_vec
                self.estimate_pdf()
                break  # exit while loop

            else:
                self.rtol = self.rtol * 10
                if self.rtol <= 1e-3:
                    print("\n  Run again with tolerance "
                          + "increased to {:.0e}".format(self.rtol))

        if not evec_found:
            print("\nStationary distribution not "
                       +"found after {:d} iterations".format(self.max_iter))
            print("Increase max_iter.")
            temp = os.system('say -v Samantha "run failed" ')



    def estimate_pdf( self ):
        """
        Estimated pdf at each data point, based on stationary
        distribution distances
        """

        # Un-normalized estimated probability density
        self.pdf = self.pi

        # Normalization through an interpoaltion or to a specified number
        if self.D==1:
            if self.interpolation_method=='linear':
                self.interp = scipy.interpolate.interp1d(
                                                self.x[:,0],
                                                self.pdf,
                                                fill_value='extrapolate',
                                                kind=self.interpolation_method )
            elif self.interpolation_method=='nearest':
                self.interp = scipy.interpolate.interp1d(
                                                self.x[:,0],
                                                self.pdf,
                                                fill_value='extrapolate',
                                                kind=self.interpolation_method )
            else:
                print('Invalid norm func!\n\n')

            self.integral, _ = mcint.integrate( self.interp,
                                                self.sampler(),
                                                measure=self.volume(),
                                                n=100000 )


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
                                                n=100000 )


        # Normalized estimated probability density
        self.pdf = self.pdf / self.integral

        # combine estimated prob with data into Nx(D+1) array [x_i, p_hat(x_i)]
        self.pX = np.append(self.x, self.pdf.reshape((self.N,1)), axis=1)




    # sampler and volume defined for the integration
    def sampler( self ):
        while True:
            gen_list = list()
            for i in range(self.D):
                r = random.uniform( self.x.T[i].min(), self.x.T[i].max() )
                gen_list.append(r)
            yield (gen_list)


    def volume( self ):
        vol = 1.0
        for i in range(self.D):
            vol = vol * ( self.x.T[i].max() - self.x.T[i].min() )
        return( vol )


    # the estimator can be used to evaluate the probability in new points
    # through evaluate_pdf
    def evaluate_pdf( self, query ):
        py = self.interp.__call__(query)/self.integral
        return(py)
