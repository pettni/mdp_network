'''Module for table value iteration'''
import time
import copy
import numpy as np

from best import DTYPE, DTYPE_ACTION, DTYPE_OUTPUT
from best.logic.translate import formula_to_pomdp

def solve_reach(network, accept, horizon=np.Inf, delta=0, prec=1e-5, verbose=False):
  return solve_reach_constrained(network, accept, horizon=horizon, delta=delta, prec=prec, verbose=verbose)

def solve_reach_constrained(network, accept, constraints=[], horizon=np.Inf, delta=0, prec=1e-5, verbose=False):
  '''solve constrained reachability problem
      
      max   P( x(t) \in accept )
      s.t.  P( x(t) \in Vcon_i ) > thresh_i

    for some t in 0, ..., horizon

  Inputs:
  - network: a POMDPNetwork
  - accept: function defining target set
  - constraints: list of pairs (Vcon, thresh) s.t. P(reach Vcon) > thresh
  - horizon: reachability horizon (standard is infinite-horizon)
  - delta: failure probability in each step
  - prec: termination tolerance (inifinite-horizon case)

  Outputs::
  - val_list: array [V0 V1 .. VT] of value functions

  The infinite-horizon problem has a stationary value function and policy. In this
  case the return argument has length 2, i.e. val_list = [V0 VT]
  '''

  V_accept = accept.astype(DTYPE, copy=False)

  it = 0
  start = time.time()

  V = np.fmax(V_accept, np.zeros(network.N, dtype=DTYPE))

  # make sure that objective does not conflict with constraints
  for Vcon, thresh in constraints:
    if thresh > 0:
      V = np.fmin(V, Vcon)
      
  if len(constraints) > 0:
    Vcon_list, thresh_list = zip(*constraints)
  else:
    Vcon_list = []
    thresh_list = []

  val_list = []
  pol_list = []
  val_list.insert(0, V_accept)

  while it < horizon:

    if verbose:
      print('iteration {}, time {:.2f}'.format(it, time.time()-start))

    # Calculate constraint Qs
    Qcon_list = [network.bellman(Vcon).reshape((-1,) + network.N) for Vcon in Vcon_list]

    # Update constraints
    Vcon_list = [Qcon.max(axis=0) for Qcon in Qcon_list]

    # Calculate Q(u_free, x)
    Q = network.bellman(V).reshape((-1,) + network.N)

    # Max over accept set
    Q = np.fmax(V_accept[np.newaxis, ...], np.maximum(Q - delta, 0))

    # Max over free actions that satisfy constraints
    mask = np.ones(Q.shape)
    for Qcon, thresh in zip(Qcon_list, thresh_list):
      mask *= (Qcon >= thresh)

    # Subtract to satisfy constraint even where Q=0
    V_new = (Q * mask).max(axis=0)
    P_new = np.unravel_index((Q * mask - 2*(1-mask) ).argmax(axis=0)
                             .astype(DTYPE_ACTION, copy=False), network.M)

    if horizon < np.Inf:
      val_list.insert(0, V_new)
      pol_list.insert(0, P_new)

    if horizon == np.Inf and np.amax(np.abs(V_new - V)) < prec:
      val_list.insert(0, V_new)
      pol_list.insert(0, P_new)
      break

    V = V_new
    it += 1

  print('finished after {:.2f}s and {} iterations'.format(time.time()-start, it))

  return val_list, pol_list


def solve_min_cost(network, costs, target, M=np.Inf, prec=1e-5, verbose =False):
  '''minimize expected cost to reach target set
      
      min   E( \sum_{t=0}^T costs(u(t), x(t)) )
      s.t.  x(t) \in Vcon_i

    for some t in 0, ..., horizon

  Inputs:
  - network: a POMDPNetwork
  - costs: matrix of size network.M + network.N with instantaneous costs
  - target: set to reach
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

  # inf at target, otherwise zero
  target0 = M * np.ones(network.N, dtype=DTYPE)
  target0[target > 0] = 0

  # expected cost to reach target
  V = target0

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

  if(np.max(V_new) == M):
    print("warning: max M equal to infinity")
    V_new[V_new >= M] = np.Inf

  print('finished after {:.2f}s and {} iterations'.format(time.time()-start, it))

  return V_new, np.unravel_index( Q.argmin(axis=0).astype(DTYPE_ACTION, copy=False), network.M)


def solve_ltl_cosafe(network, formula, predicates, delta=0., horizon=np.Inf, verbose=False):
  '''synthesize a policy that maximizes the probability of
     satisfaction of formula
     Inputs:
      - network: a POMDPNetwork 
      - formula: a syntactically cosafe LTL formula over AP
      - predicates: list of triples ('output', 'ap', output -> 2^{0,1}) defining atomic propositions
                        if an atomic proposition depends on several outputs, add an intermediate logic gate

     Example: If AP = {'s1', 's2'}, then connection(x) should be a 
     value in 2^2^{'s1', 's2'} 

     Outputs:
      - pol: a LTL_Policy maximizing the probability of enforcing formula''' 

  if verbose:
    start = time.time()
    print("constructing augmented network...")

  dfsa, dfsa_init, dfsa_final = formula_to_pomdp(formula)

  network_copy = copy.deepcopy(network)

  network_copy.add_pomdp(dfsa)
  for ap, (outputs, conn) in predicates.items():
    network_copy.add_connection(outputs, ap, conn)

  Vacc = np.zeros(network_copy.N)
  Vacc[...,list(dfsa_final)[0]] = 1

  if verbose:
    print("finished constructing augmented network in {:.2f}s".format(time.time()-start))

  val, pol = solve_reach(network_copy, Vacc, delta=delta, horizon=horizon, verbose=verbose)
  
  return LTL_Policy(dfsa.input_names, dfsa._Tmat_csr, list(dfsa_init)[0], dfsa_final, val, pol)


class LTL_Policy(object):
  """control policy"""
  def __init__(self, proplist, dfsa_Tlist, dfsa_init, dfsa_final, val, pol):
    '''create a control policy object'''
    self.proplist = proplist
    self.dfsa_Tlist = dfsa_Tlist
    self.dfsa_init = dfsa_init
    self.dfsa_final = dfsa_final
    self.val = val
    self.pol = pol

    self.dfsa_state = self.dfsa_init

  def reset(self):
    '''reset controller'''
    self.dfsa_state = self.dfsa_init

  def report_aps(self, aps):
    '''report atomic propositions to update internal controller state'''
    dfsa_action = tuple(int(ap in aps) for ap in self.proplist)
    row = self.dfsa_Tlist[dfsa_action].getrow(self.dfsa_state)
    self.dfsa_state = row.indices[0]

  def __call__(self, syst_state, t=0):
    '''get input from policy'''
    if t >= len(self.val)-1:
      print('Warning: t={} larger than horizon {}. Setting t={}'.format(t, len(self.val)-1, len(self.val)-2))
      t = len(self.val)-2
    joint_state = tuple(syst_state) + (self.dfsa_state,)

    u = tuple(self.pol[t][m][joint_state] for m in range(len(self.pol[t])))

    return u, self.val[t][joint_state]

  def finished(self):
    '''check if policy reached target'''
    return self.dfsa_state in self.dfsa_final
