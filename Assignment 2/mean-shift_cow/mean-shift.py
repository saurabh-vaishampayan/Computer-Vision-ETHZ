import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

BATCHSIZE = 512

def distance(x, X):
    x_ = torch.transpose(x.unsqueeze(1),0,1)
    dist = torch.cdist(x_,X)
    return dist
    
def distance_batch(x, X):
    dist = torch.cdist(x,X)
    return dist
    
def gaussian(dist, bandwidth):
    wt = torch.exp(-dist**2/(2*(bandwidth**2)))
    return wt

def update_point(weight, X):
    
    nr = torch.matmul(weight, X)
    
    nr = torch.transpose(nr,0,1)
    dr = torch.sum(weight,axis=1)
    
    updated_pt = (torch.transpose((nr/dr),0,1)).squeeze(1)
    return updated_pt
    
def update_point_batch(weight, X):
    
    nr = torch.matmul(weight, X)
    
    nr = torch.transpose(nr,0,1)
    dr = torch.sum(weight,axis=1)
    
    updated_pt = torch.transpose((nr/dr),0,1)
    return updated_pt
    
def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    i = 0
    data_loader = torch.utils.data.DataLoader(X,batch_size=BATCHSIZE)
    
    for x in data_loader:
        dist = distance_batch(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i*BATCHSIZE:i*BATCHSIZE+x.size()[0]] = update_point_batch(weight, X)
        i+=1
        
    return X_
    
def meanshift(X):
    X = X.clone()
    for _ in range(20):
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

torch.set_grad_enabled(False)

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
#X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)