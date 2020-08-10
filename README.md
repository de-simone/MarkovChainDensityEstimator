# Markov Chain Density Estimator (MCDE)

MCDE is a novel way to estimate the PDF of a distribution based on the properties of Markov Chains. This estimator can be seen as a modified version of Kernel Density Estimation (KDE) as well. More details can be found on [20xx.xxxx](https://arxiv.org/)

All data science tasks that have a density based approach can be tackled using this estimator. We have showed that local anomaly detection based on MCDE works in a very satisfying way. We have also showed that this density estimator can perform better than KDE for a large enough sample.

### Installation

MCDE works in both python 2 and python 3. It requires: numpy, scipy, mcint and sklearn.

In order to import the main class just copy the file [MCDE.py](./MCDE.py) to your working directory and add on top of your python file:

```
from MCDE import MCDensityEstimator
```

### Parameters

 +  weight_func: function weighting the distances, corresponds to the kernels of KDE. These are implemtented via sklearn, so we have 'gaussian', 'exponential', 'epanehcnikov', 'tophat', 'linear', 'cosine';
 +  bw: the main parameter of the estimator, analogous to the bandwidth of the Kernel Density Estimator. bw is fixed by performing an optimization;
 +  interpolation_method: in order to normalize the PDF and estimate the PDF in points not belonging to the original dataset, it is necessary to intepolate. As of now the Nearest Neighbor interpolator ('nearest'), the linear interpolator ('linear') and an extension of KDE ('kde') to points not belonging to the sample are implemented. The linear interpoaltor is more accurate, but the computation is exceedingly long for a sample with more than 6 features;
 +  metric: the distances are calculated according to some metric, by default we use the euclidean metric. See [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) for more options;
 +  normalize: binary handle that allows to turn on/off normalization of the PDF;
 +  mov_bias: value of the movement bias, set by default to 1;
 +  vol_ext: the normalization is performed on the smallest n-box containing all sample points. This needs to be extended for a KDE interpolator by a factor 1-3. Final performance is not heavily affected by this;
 +  kde_tol: Internal parameter of sklearn, sets the precision in the KDE calculation, increasing it might spped up the calculations, at the price of inferior precision. Final performance is not heavily affected by this.



### Usage

An example of possible usage is in the Jupyter notebook [example.ipynb](./example.ipynb). 
Here we have showed how to estimate the PDF in a 1D and a 2D example. For the 1D case we also presented a self-consistent optimization step. A qualitative comparison with the true PDF is also shown.

### License

This project is licensed under the terms of the MIT license.
