# -*- coding: utf-8 -*-
# author: krocki
import numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from gradcheck import *

HN = 100 # the code layer
M = 784 # input/reconstruction size
S = 8 # batch size
lr = 1e-3 # learning rate
max_iters=100000
save_png = False

do_gradcheck = False
datatype = np.float64 if do_gradcheck else np.float32

def save_arr(fname, arr, rec):
  arr = arr.reshape(28, 28, S)
  arr = arr.transpose(2, 0, 1).reshape(S * 28, 28)
  rec = rec.reshape(28, 28, S)
  rec = rec.transpose(2, 0, 1).reshape(S * 28, 28)
  full = np.zeros((S*28, 28*2))
  full[:, :28] = arr
  full[:, 28:] = rec
  img = plt.imshow((full[:,:]), cmap='viridis', interpolation='nearest')
  plt.xticks([]),plt.yticks([])
  plt.savefig("{:}".format(fname))

def sigmoid(hs):
  return 1.0/(1.0 + np.exp(-hs))

def dsigmoid(h, dh):
  return dh * h * (1.0 - h)

def forward(xs, model):
  states = {}; states['xs'] = xs
  hs = np.dot(model['Wxh'].T, xs)
  hs = sigmoid(hs) ; states['hs'] = hs
  ys = np.dot(model['Why'].T, hs)
  ps = sigmoid(ys) ; states['ys'] = ys
  states['ps'] = ps
  return states

def lossfun(states, ts):
  # MSE
  ce = (ts - states['ps']) ** 2
  return np.sum(ce)

def backward(states, model, ts):
  grads = {}
  dy = 2.0 * (states['ps'] - ts)
  dy = dsigmoid(states['ps'], dy)
  grads['Why'] = np.dot(states['hs'], dy.T)
  grads['hs'] = np.dot(model['Why'], dy)
  grads['hs'] = dsigmoid(states['hs'], grads['hs'])
  grads['Wxh'] = np.dot(states['xs'], grads['hs'].T)

  return grads

def apply_grads(model, grads, lr):
  for t in model:
    model[t] -= grads[t] * lr
  return model

if __name__ == "__main__":

  f = gzip.open('./data/mnist.pkl.gz', 'rb')
  data, _, _ = pickle.load(f, encoding='latin1')
  f.close()

  model = {}

  Wxh = np.random.randn(M, HN).astype(datatype) * 0.01
  Why = np.random.randn(HN, M).astype(datatype) * 0.01

  model['Wxh']=Wxh
  model['Why']=Why

  i=0;
  smooth_loss = None

  while i<max_iters:
    rands = np.random.randint(0, \
      high=data[0].shape[0], size=S)

    xs = data[0][rands].T
    ts = xs
    states = forward(xs, model)
    loss = lossfun(states, ts)
    smooth_loss = loss * 0.999 + smooth_loss * 0.001 if smooth_loss else loss
    grads = backward(states, model, ts)

    if (i%1000==0 and i>0):
      if save_png: save_arr('x_rec_{}.png'.format(i), xs, states['ps'])
      print('iter {:6}, loss = {:5.2f}'.format(i, smooth_loss), end='')
      if do_gradcheck:
        err=checkgrads(xs, model, forward, lossfun, ts, grads)
        print(', gradcheck err {:2.9f}'.format(err), end='')
        if err>1e-7: print(' !!!')
        else: print(' OK')
      else: print('')

    model = apply_grads(model, grads, lr)
    i=i+1
