# -*- coding: utf-8 -*-
# author: krocki
import numpy as np
import gzip, pickle
from gradcheck import *

HN = 100
M = 784
Y = 10
S = 8
max_iters = 100000
lr = 1e-3 # learning rate

do_gradcheck = True
datatype = np.float64 if do_gradcheck else np.float32

def sigmoid(hs):
  return 1.0/(1.0 + np.exp(-hs))

def dsigmoid(h, dh):
  return dh * h * (1.0 - h)

def softmax(ys):
  m0 = np.max(ys, axis=0)
  ps = np.exp(ys-m0)
  sums = np.sum(ps, axis=0)
  ps = ps / sums
  return ps

def forward(xs, model):
  hs = np.dot(model['Wxh'].T, xs)
  hs = sigmoid(hs)
  ys = np.dot(model['Why'].T, hs)
  ps = softmax(ys)
  states = {}
  states['xs'] = xs
  states['hs'] = hs
  states['ys'] = ys
  states['ps'] = ps
  return states

def lossfun(states, ts):
  ce = -np.log(states['ps'][ts>0])
  return np.sum(ce)

def backward(states, model, ts):
  grads = {}
  dy = states['ps'] - ts
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

  e = np.eye(Y, Y)

  model = {}

  Wxh = np.random.randn(M, HN).astype(datatype) * 0.01
  Why = np.random.randn(HN, Y).astype(datatype) * 0.01

  model['Wxh']=Wxh
  model['Why']=Why

  i=0;
  smooth_loss = None

  while i<max_iters:
    rands = np.random.randint(0, \
      high=data[0].shape[0], size=S)
    ts = e[:, data[1][rands]]

    xs = data[0][rands].T
    states = forward(xs, model)
    loss = lossfun(states, ts)
    smooth_loss = loss * 0.999 + smooth_loss * 0.001 if smooth_loss else loss
    grads = backward(states, model, ts)
    if (i%1000==0 and i>0):
      print('iter {:6}, loss = {:5.2f}'.format(i, smooth_loss), end='')
      if do_gradcheck:
        err=checkgrads(xs, model, forward, lossfun, ts, grads)
        print(', gradcheck err {:2.9f}'.format(err), end='')
        if err>1e-7: print(' !!!')
        else: print(' OK')
      else: print('')

    model = apply_grads(model, grads, lr)
    i=i+1
