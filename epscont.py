#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Sat Jan 15 17:01:02 2022

# @author: pgerstoft
# """
import numpy as np

def crandn(*shape): # complex gaussian random noise
    return np.random.randn(*shape)+ 1j*np.random.randn(*shape)

def epscont(shape,sigma,epsilon=0,lambdavar=1,return_mask=False):
#def epscont(shape,sigma,epsilon=0,mu=0,lambdavar=1,return_mask=False):
    """Generate an array of n x m random deviates from an epsilon-contaminated 
     complex Gaussian distribution:
         f(x)= (1-epsilon)*CN(mu,sigma^2) + epsilon*CN(mu,(lambdavar sigma)^2)
     The variance is 
        varx = (1-epsilon)*sigma^2 + epsilon * (lambdavar*sigma)^2

     INPUT:
         shape of the array, tuple of integers 
         mu  = complex number ( mean of random variable x)
         lambdavar = multiplier for epsilon contaminated variance (positive real) 
         sigma = variance of true normal  (positive real) 

     USAGE: 
         n = 500; epsilon = 0.2; mu = 1+1i; lambdavar=2; sigma=1;
         x = epscont(n,epsilon,mu,lambdavar,sigma);
           =_d (1-eps)*n_1 + eps*n_2 
         var(x)
         varx = (1-epsilon)*sigma^2 + epsilon * (lambdavar*sigma)^2
    """

    is2 = 1/np.sqrt(2) 

    # complex gaussian noise, same mean applied
    x = sigma*is2*crandn(*shape) #+ mu

    # contaminate at binmask
    bmask   = np.random.binomial(1, epsilon, size=shape)
    cnt     = np.sum(bmask.ravel())
    x[np.where(bmask)] = lambdavar*sigma*is2*crandn(cnt) #+ mu

    if return_mask:
        return x,bmask
    else:
        return x

if __name__ == "__main__":
    # test pars
    pars = dict(sigma=1.,epsilon=0.05,lambdavar=1)

    # both functions have the same size
    n0 = epscont((100,5),**pars)
    T = np.zeros((100,5))
    n1 = epscont(T.shape,**pars)
    assert n0.size == n1.size, "n0 and n1 do not have the same size"

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    noise,mask = epscont((20,20),1,epsilon = 0,return_mask = True)
    plt.subplot(231)
    plt.imshow(noise.real)
    plt.colorbar()
    plt.ylabel('eps=0, \nsigma = 1')
    plt.title('real')
    plt.subplot(232)
    plt.imshow(noise.imag)
    plt.title('imag')
    plt.colorbar()
    plt.subplot(233)
    plt.imshow(mask,vmin=0,vmax=1,cmap='binary')
    plt.title('mask')
    plt.colorbar()

    noise,mask = epscont((20,20),sigma=1, epsilon = 0.05,  lambdavar = 100, return_mask = True)
    plt.subplot(234)
    plt.imshow(noise.real)
    plt.colorbar()
    plt.ylabel('eps=0.05, \nsigma = 1, lambda = 100')
    plt.subplot(235)
    plt.imshow(noise.imag)
    plt.colorbar()
    plt.subplot(236)
    plt.imshow(mask,vmin=0,vmax=1,cmap='binary')
    plt.title('mask')
    plt.colorbar()
    plt.draw()
    plt.savefig('Epscont.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()