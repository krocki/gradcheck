import numpy as np
def samplegrad(xs, model, forward, lossfun, ts):

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

def checkgrads(xs, model, forward, lossfun, ts, analytical):
  numerical = samplegrad(xs, model, forward, lossfun, ts)
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
    return np.max(errors[k])


