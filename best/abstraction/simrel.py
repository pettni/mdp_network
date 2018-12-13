# Copyright (c) 2013-2017 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

import polytope as pc
import cvxpy as cvx
import numpy as np
import math
from numpy.linalg import inv
import itertools
import scipy.optimize

def eps_err(lti, dist, lamb=.99999):
  """
  Quantify accuracy of simulation with respect to disturbance given as a polytope
  :param lti: contains dynamics matrix lti.a, lti.b
  :param dist: The disturbance given as a Polytope
  :return: Invariant set R and epsilon
  """
  n = lti.dim
  m = lti.m
  A = lti.a
  B = lti.b
  C = lti.c

  Vertices = pc.extreme(dist)

  # Introduce variables
  Minv = cvx.Variable((n, n), PSD=True)
  L    = cvx.Variable((m,n))
  eps2 = cvx.Variable((1, 1), nonneg=True)
  lam  = cvx.Parameter(nonneg=True, value=lamb)

  basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                    [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                    [A * Minv + B * L , np.zeros((n,1)), Minv]])

  cmat = cvx.bmat([[Minv, Minv * C.T],[C* Minv, np.eye(C.shape[0])]])
  constraintstup = (cmat >> 0,)

  ri =  np.zeros((n,1))
  for i in range(Vertices.shape[0]):
    ri = Vertices[i].reshape((n,1))
    rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                     [np.zeros((1, n)), np.zeros((1, 1)), ri.T],
                     [np.zeros((n, n)), ri, np.zeros((n, n))] ]   )
    constraintstup += (basic + rmat >> 0,)
  constraints = list(constraintstup)

  obj = cvx.Minimize(eps2)
  prob = cvx.Problem(obj, constraints)

  def f_opt(val):
    lam.value = val
    try:
      prob.solve()
    except cvx.error.SolverError :
      return np.inf

    return eps2.value[0,0]**.5

  lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0,1)
  lam.value = lam_opt
  prob.solve()
  eps_min = eps2.value[0,0] ** .5
  M_min = inv(Minv.value)
  K_min = L.value*Minv.value

  print ("status:", prob.status)
  print ("optimal epsilon", eps_min)
  print ("optimal M", M_min)
  print ("Optimal K", K_min)

  return M_min, K_min, eps_min


def eps_err_tune_eta(lti, grid, dist, lamb=.99999):
  """
  Tune abstraction grid size

  :param lti: contains dynamics matrix lti.a, lti.b
  :param Dist: The disturbance given as a polytope
  :return: Invariant set R and epsilon
  """
  n = lti.dim
  m = lti.m
  A = lti.a
  B = lti.b
  C = lti.c
  Vertices_grid = pc.extreme(grid)
  if type(lti.T2x) is np.ndarray:
    Apol = dist.A.dot(lti.T2x)
    dist = pc.Polytope(A=Apol, b=dist.b)
    Vertices_dist = pc.extreme(dist)
  else:
    Vertices_dist = pc.extreme(dist)

  # define variables
  Minv = cvx.Variable( (n,n), PSD=True )
  L = cvx.Variable((m,n))
  d = cvx.Parameter((2, 1))
  eps2 = cvx.Variable((1, 1), nonneg=True)
  lam = cvx.Parameter(nonneg=True, value=lamb)
  basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                    [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                    [A * Minv + B * L , np.zeros((n,1)), Minv]])

  cmat = cvx.bmat([[Minv, Minv * C.T],[C* Minv, np.eye(C.shape[0])]])
  constraintstup = (cmat >> 0,)

  ri =  np.zeros((n,1))
  for i,j in itertools.product(range(Vertices_grid.shape[0]),range(Vertices_dist.shape[0])):
    ri = Vertices_grid[i].reshape((n,1))
    rj = Vertices_dist[i].reshape((n, 1))
    rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                     [np.zeros((1, n)), np.zeros((1, 1)), ri.T*cvx.diag(d)+rj.T],
                     [np.zeros((n, n)), cvx.diag(d)*ri+rj, np.zeros((n, n))] ]   )
    constraintstup += (basic + rmat >> 0,)

  constraints = list(constraintstup)

  obj = cvx.Minimize(eps2)
  prob = cvx.Problem(obj, constraints)

  def f_opt(val):
    lam.value = val
    try:
      prob.solve()
    except cvx.error.SolverError :
      return np.inf
    return eps2.value[0,0]**.5

  def f_optd(val):
    eta = np.array([[math.cos(val)], [math.sin(val)]])
    d.value=eta
    lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0, 1, maxfun=10)
    lam.value = lam_opt
    prob.solve()
    return eta[0] ** -1 * eta[1] ** -1 * eps2.value[0,0]

  vald = scipy.optimize.fminbound(lambda val: f_optd(val), 0,math.pi/2,maxfun =40)
  eta = np.array([[math.cos(vald[0])], [math.sin(vald[0])]])
  d.value = eta

  lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0,1,maxfun =10)
  lam.value = lam_opt
  prob.solve()

  eps_min = eps2.value[0,0] ** .5
  M_min = inv(Minv.value)
  K_min = L.value*Minv.value

  print ("status:", prob.status)
  print ("optimal epsilon", eps_min)
  print ("optimal M", M_min)
  print ("Optimal K", K_min)

  return eta, M_min, K_min, eps_min