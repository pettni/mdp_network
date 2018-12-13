import numpy as np

from best.models.pomdp import POMDP, POMDPNetwork
from best.solvers.valiter import *

def test_costsolve1():

  T1 = np.array([[0, 0, 0, 1, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]])

  T2 = np.array([[0,  1,   0,  0,   0],
               [ 0,   0, 0.5,  0, 0.5],
               [ 0,   0,   0,  0,   1],
               [ 0,   0,   0,  0,   1],
               [ 0,   0,   0,  0,   1]])

  pomdp = POMDP([T1, T2])

  network = POMDPNetwork([pomdp])

  costs = np.ones([2, 5])
  costs[1,2] = 50
  costs[1,3] = 20
  costs[:,4] = 0

  target = np.array([0, 0, 0, 0, 1])

  val, pol = solve_min_cost(network, costs, target, M=1000)

  np.testing.assert_almost_equal(val, [21, 26, 50, 20, 0])

def test_costsolve2():

  T0 = np.array([[0.1, 0.9, 0], [0, 1, 0], [0, 0, 1]])
  network = POMDPNetwork([POMDP([T0])])

  costs = np.ones([1,3])
  target = np.array([0,1,0])

  val, pol = solve_min_cost(network, costs, target, M=10)

  np.testing.assert_almost_equal(val, [1/0.9, 0, np.Inf], decimal=4)

