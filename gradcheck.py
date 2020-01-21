# -*- coding: utf-8 -*-
# author: krocki
import numpy as np
import gzip, pickle

HE = 64
HD = 32
M = 784
Y = 10
S = 8

datatype = np.float64

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

def samplegrad(xs, model, ts):

  samples = 100
  delta = 1e-5

  grads = {}

  for layer, params in enumerate(model):
    p = model[params]
    grads[params] = {}
    p_shape=p.size
    R = np.random.randint(0, p_shape, size=samples)
    for r in R:
      p_old = p.flat[r]
      p.flat[r] = p_old + delta
      y0 = lossfun(forward(xs, model), ts)
      p.flat[r] = p_old - delta
      y1 = lossfun(forward(xs, model), ts)
      gn = (y0 - y1) / (2 * delta)
      p.flat[r] = p_old
      grads[params][r] = gn

  return grads

def checkgrads(xs, model, ts, analytical):
  numerical = samplegrad(xs, model, ts)
  errors = {}

  for i,k in enumerate(numerical):
    n = numerical[k]
    a = analytical[k]
    e = np.zeros_like(a)
    nmin, nmax = None, None
    amin, amax = None, None
    for j,m in enumerate(n):
      e.flat[m] = np.fabs(n[m] - a.flat[m])
      if amin: amin = min(amin, a.flat[m])
      else: amin = a.flat[m]
      if amax: amax = max(amax, a.flat[m])
      else: amax = a.flat[m]
      if nmin: nmin = min(nmin, n[m])
      else: nmin = n[m]
      if nmax: nmax = max(nmax, n[m])
      else: nmax = n[m]
    errors[k] = e
    print('{}: samples {}\n\trange n [{}, {}] \
    \n\trange a [{}, {}]\n\tmax err = {}'.format(k, len(n), \
    nmin, nmax, amin, amax, np.max(errors[k])))

if __name__ == "__main__":

  f = gzip.open('./data/mnist.pkl.gz', 'rb')
  data, _, _ = pickle.load(f, encoding='latin1')
  f.close()

  rands = np.random.randint(0, \
    high=data[0].shape[0], size=S)

  xs = data[0][rands].T
  e = np.eye(Y, Y)

  ts = e[:, data[1][rands]]
  model = {}

  Wxh = np.random.randn(M, HE).astype(datatype) * 0.01
  Why = np.random.randn(HE, Y).astype(datatype) * 0.01
  #Wyh = np.random.randn(Y, HD).astype(datatype) * 0.01
  #Whx = np.random.randn(HD, M).astype(datatype) * 0.01

  model['Wxh']=Wxh
  model['Why']=Why
  #model['Wyh']=Wyh
  #model['Whx']=Whx

  states = forward(xs, model)
  loss = lossfun(states, ts)
  analytical = backward(states, model, ts)
  checkgrads(xs, model, ts, analytical)
