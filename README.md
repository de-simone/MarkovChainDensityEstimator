# Markov Chain Density Estimator (MCDE)

MCDE is a novel way to estimate the PDF of a distribution based on the properties of Markov Chains. More details can be found on [20xx.xxxx](https://google.com/)  **insert url here**

All data science tasks that have a density based approach can be tackled using this estimator. We have showed that anomaly detection based on MCDE works in a very satisfying way. We have also showed that this density estimator is less affected by the "curse of dimnesionality" with respect to other common choices, such as Kernel Density Estimation.

### Installation

MCDE works in both python 2 and python 3. It requires: numpy, scipy, mcint and sklearn.

In order to import the main class just copy the file [MCDE.py](./MCDE.py) **link to file in github** to your working directory and add on top of your python file:

```
from MCDE import MCDensityEstimator
```

### Parameters

 + *weight_func*: function weighting the distances, already implemented are the exponential of the distance ('exp') and the exponentaial of the squared distance; ('exp2'),
 +  *beta*: the main parameter of the estimator, analogous to the bandwidth of the Kernel Density Estimator. beta is fixed by performing an optimization step;
 +  *max_iter*: cut on the maximum number of repetitions in the loop looking for an eigenvector, ncessary to avoid infinite loops;
 +  *rtol*: sets the precision in the search for the eigenvector, if the required precision cannot be obtained within max_iter steps, rtol will be increased;
 +  *interpolation_method*: in order to normalize the PDF and estimate the PDF in points not belonging to the original dataset it is necessary to intepolate. As of now the Nearest Neighbor interpolator ('nearest') and the linear interpolator ('linear') are implemented. The linear interpoaltor is more accurate, but the computation is exceedingly long for a sample with more than 6 features.
 +  *metric*: the distances are calculated according to some metric, by default we use the euclidean metric. See [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) for more options.


### Usage

An example of possible usage is in the Jupyter notebook [example.ipynb](./example.ipynb) **link to file in github**. 
There you can find how to estimate the PDF in a 1D and a 2D example, including the optimization phase necessary to choose the parameter of our estimator. A qualitative comparison with the true PDF is also shown.

### License

This project is licensed under the terms of the MIT license.
