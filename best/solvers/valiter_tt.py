'''Module for table value iteration'''
import time
import copy
import numpy as np

import TensorToolbox as TT

from best import DTYPE, DTYPE_ACTION, DTYPE_OUTPUT

def solve_min_cost_tt(network, costs, target, M=100, prec=1e-5, verbose =False):
  '''minimize expected cost to reach target set
      
      min   E( \sum_{t=0}^T costs(u(t), x(t)) )
      s.t.  x(t) \in Vcon_i

    for some t in 0, ..., horizon

  Inputs:
  - network: a POMDPNetwork
  - costs: black-box function [network.M + network.N] -> R
  - target: black-box function [network.N] -> {0, 1}
  - M: a priori upper bound on cost, if too small solution is invalid, if equal to inf algorithm does 
       not work unless there exists a finite horizon with P( x(T) \in target ) = 1
  - prec: termination tolerance

  Outputs::
  - val_list: array [V0 V1 .. VT] of value functions

  The infinite-horizon problem has a stationary value function and policy. In this
  case the return argument has length 2, i.e. val_list = [V0 VT]
  '''

  it = 0
  start = time.time()

  def target_f(X, params):
    if target(X):
      return 0
    else:
      return M

  V0_tw = TT.TensorWrapper(target_f, [range(n) for n in network.N], None, dtype=DTYPE)
  V0_tt = TT.TTvec(V0_tw).build(method='ttcross', eps=1e-4)

  def costs_f(X, params):
    return costs(X)

  costs_tw = TT.TensorWrapper(costs_f, [range(m) for m in network.M] + [range(n) for n in network.N], None, dtype=DTYPE)
  costs_tt = TT.TTvec(costs_tw).build(method='ttcross', eps=1e-4)

  print("Initial V size", V0_tt.size())

  # expected cost to reach target
  V = V0_tt

  while True:
    if verbose:
      print('iteration {}, time {:.2f}'.format(it, time.time()-start))

    Q = costs + network.bellman(V).reshape((-1,) + network.N)
    V_new = np.fmin(Q.min(axis=0), target0)

    if np.all(np.isfinite(V_new) == np.isfinite(V)) and \
       np.all(np.abs(V_new[np.isfinite(V)] - V[np.isfinite(V)]) < prec):
      break

    V = V_new
    it += 1

  V_new[V_new >= M] = np.Inf

  print('finished after {:.2f}s and {} iterations'.format(time.time()-start, it))

  return V_new, np.unravel_index( Q.argmin(axis=0).astype(DTYPE_ACTION, copy=False), network.M)
