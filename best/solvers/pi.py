import numpy as np
import time
import copy

from best import DTYPE, DTYPE_ACTION, DTYPE_OUTPUT
from best.models.pomdp import *

def bellman_policy(network, W, P_ux):
  '''calculate dense V function via one Bellman step
     V(x) = E[ W(x') | x, P_ux(x)]'''
  Q_ux = network.bellman(W)
  V_xx = np.tensordot(P_ux, Q_ux, axes=(range(len(network.M)), range(len(network.M))))  # get rid of m
  for n in range(len(network.N)):
    V_xx = np.diagonal(V_xx, axis1=0, axis2=len(network.N)-n)
  return V_xx

def just_one_one(row):
  idx = np.nonzero(row)
  ret = np.zeros(row.shape)
  ret[idx[0][0]] = 1
  return ret

def PI(network, accept, prec=1e-5, verbose=False):
  '''solve problem 
      max   P( x(t) \in accept )
     using policy iteration'''

  it = 0
  start = time.time()
  np.set_printoptions(precision=2, floatmode='fixed')
  V_x = np.zeros(network.N)

  dumb = False

  while True:

    print("iteration", it, ": ", V_x)

    # policy improvement
    Q_ux = network.bellman(V_x)
    V_x_max = np.amax(Q_ux, keepdims=1, axis=tuple(range(len(network.M))))

    if not dumb:
      # uniform distribution over all close to argmax
      P_ux = np.greater_equal(Q_ux, V_x_max - prec, dtype=DTYPE)
      P_ux = P_ux / np.sum(P_ux, keepdims=1, axis=tuple(range(len(network.M))))
    else:
      # all maximizing actions
      P_ux = (Q_ux == V_x_max).reshape(prod(network.M), prod(network.N))
      # pick just one (the first)
      P_ux = np.apply_along_axis(just_one_one, 0, P_ux) 

    # policy evaluation
    V_x_new = policy_evaluation(network, accept, P_ux, prec=prec, verbose=False)

    if (not dumb and np.amax(V_x_new - V_x) < prec) or (dumb and it > 20):
      break

    V_x = V_x_new
    it += 1

  if verbose:
    print('finished after {:.2f}s and {} iterations'.format(time.time()-start, it))

  return V_x, P_ux

def policy_evaluation(network, accept, P_ux, prec=1e-5, verbose=False):
  '''return the value of a P_ux on a reachability problem'''
  
  V_accept = accept.astype(DTYPE, copy=False)

  it = 0
  start = time.time()

  V = np.fmax(V_accept, np.zeros(network.N, dtype=DTYPE))
      
  while True:

    if verbose:
      print('iteration {}, time {:.2f}'.format(it, time.time()-start))

    # Calculate Q(u_free, x)
    V_new = np.fmax(bellman_policy(network, V, P_ux), V_accept)

    if np.amax(np.abs(V_new - V)) < prec:
      break

    V = V_new
    it += 1

  if verbose:
    print('finished after {:.2f}s and {} iterations'.format(time.time()-start, it))

  return V