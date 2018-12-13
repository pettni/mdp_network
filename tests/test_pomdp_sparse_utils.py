import numpy as np
import sparse
from best.models.pomdp import POMDP, POMDPNetwork
from best.models.pomdp_sparse_utils import *

def test_propagate_distr():

  T00 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0.7, 0, 0.3]])
  T01 = np.array([[0, 0,     1], [0, 0, 1], [0, 0, 1]])

  T10 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  T11 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

  Z0 = np.array([[0.5, 0.5], [0, 1], [1, 0]])

  pomdp = POMDP({ (0,0): T00, (0,1): T01, (1,0): T10, (1,1): T11 }, [Z0], input_names=['u1', 'u2'])

  D1_ux = sparse.COO([[0, 0, 1], [0, 1, 1], [0, 0, 0]], [1, 0, 0], shape = (2, 2, 3))
  D1_xz = propagate_distribution(pomdp, D1_ux)
  D1_xz_r = np.array([[0, 0], [0, 0.5], [0.5, 0]])
  np.testing.assert_almost_equal(D1_xz.todense(), D1_xz_r)

  D2_ux = sparse.COO([[0, 0, 1], [0, 1, 1], [0, 0, 0]], [0, 1, 0], shape = (2, 2, 3))
  D2_xz = propagate_distribution(pomdp, D2_ux)
  D2_xz_r = np.array([[0, 0], [0, 0], [1, 0]])
  np.testing.assert_almost_equal(D2_xz.todense(), D2_xz_r)

  D3_ux = sparse.COO([[0, 0, 1], [0, 1, 1], [0, 0, 0]], [0, 0, 1], shape = (2, 2, 3))
  D3_xz = propagate_distribution(pomdp, D3_ux)
  D3_xz_r = np.array([[0.5, 0.5], [0, 0], [0, 0]])
  np.testing.assert_almost_equal(D3_xz.todense(), D3_xz_r)

  D4_ux = sparse.COO([[0, 0, 1], [0, 1, 1], [0, 0, 0]], [0.33, 0.33, 0.34], shape = (2, 2, 3))
  D4_xz = propagate_distribution(pomdp, D4_ux)
  np.testing.assert_almost_equal(D4_xz.todense(), 0.33 * D1_xz_r + 0.33 * D2_xz_r + 0.34 * D3_xz_r)

def test_diag():

  dA = np.random.rand(5,5)
  sA = sparse.COO(dA)
  diag = diagonal(sA, axis1=0, axis2=1).todense()

  np.testing.assert_almost_equal(diag, np.diagonal(dA))

  dB = np.random.rand(5,6,5)
  sB = sparse.COO(dB)
  diag = diagonal(sB, axis1=0, axis2=2)

  np.testing.assert_equal(diag.shape, (6,5))
  np.testing.assert_almost_equal(diag.todense(), np.diagonal(dB, axis1=0, axis2=2))

def test_network_distribution():

  T1 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]])
  Z2 = np.array([[1,0], [1,0], [0,1]])
  pomdp1 = POMDP([T1], [Z2], input_names=['u1'], state_name='x1', output_name='z1')

  T21 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
  T22 = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
  pomdp2 = POMDP([T21, T22], [np.eye(3)], input_names=['u2'], state_name='x2', output_name='z2')

  network = POMDPNetwork([pomdp1, pomdp2])
  network.add_connection(['z1'], 'u2', lambda z1: {z1})

  # distribution over u1 x1 x2
  D_ux = sparse.COO([ [0], [0], [0] ], [1], shape=(1, 3, 3))

  D_xz = propagate_network_distribution(network, D_ux)

  D_xz_r = sparse.COO([ [1, 2], [1, 2], [0, 1], [1, 2] ], [0.5, 0.5], shape=(3,3,2,3))

  np.testing.assert_equal(D_xz.todense(), D_xz_r.todense())

def test_evaluate_Q():

  T1 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]])
  Z2 = np.array([[1,0], [1,0], [0,1]])
  pomdp1 = POMDP([T1], [Z2], input_names=['u1'], state_name='x1', output_name='z1')

  T21 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
  T22 = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
  pomdp2 = POMDP([T21, T22], [np.eye(3)], input_names=['u2'], state_name='x2', output_name='z2')

  network = POMDPNetwork([pomdp1, pomdp2])
  network.add_connection(['z1'], 'u2', lambda z1: {z1})

  V = np.array([[0,0,0], [0,0,0], [0,0,1]])

  np.testing.assert_almost_equal(evaluate_Q(network, (0,), (0,0), V), 0.5)
