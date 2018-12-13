import numpy as np
import polytope as pc

from best.models.lti import LTI
from best.abstraction.gridding import *

def test_grid():
  abstr = Grid([-1, -1], [1, 1], [5, 5])
  for s in range(5*5):
    np.testing.assert_equal(s, abstr.x_to_s(abstr.s_to_x(s)))

  xx = 2 * np.random.rand(2,10) - 1
  for i in range(10):
    x = xx[:,i].flatten()
    s = abstr.x_to_s(x)
    xc = abstr.s_to_x(s)

    np.testing.assert_array_less( np.abs(xc - x),  0.2)

def test_grid_cdf_1d():
  Pvec = grid_cdf_1d(0, 1, -10, 10, 10)
  np.testing.assert_almost_equal(Pvec, [0.5, 0.5])

  Pvec = grid_cdf_1d(0, 1, 0, 0.4, 0.1)
  np.testing.assert_almost_equal(Pvec, [ 0.03983, 0.07926 - 0.03983, 0.11791 - 0.07926, 0.15542 - 0.11791], decimal=3)

  Pvec = grid_cdf_1d(2.5, 0, 0, 5, 1)
  np.testing.assert_almost_equal(Pvec, [ 0, 0, 1, 0, 0])

def test_grid_cdf_nd():
  Pmat = grid_cdf_nd([0, 0], np.diag([1, 1]), [-10, -10], [10, 10], [10, 10])
  np.testing.assert_equal(Pmat.shape, (2,2))
  np.testing.assert_almost_equal(Pmat, np.array([ [0.25, 0.25], [0.25, 0.25] ]))

def test_poly_ellipse():
  poly = pc.Polytope( np.array([[1,0], [0,1], [-1,0], [0,-1]]), np.ones((4,1)) )

  M = np.eye(2)
  eps = 0.5

  np.testing.assert_equal(poly_ellipse_isect(np.array([0.49,0]), M, eps, poly),
                          {True})

  np.testing.assert_equal(poly_ellipse_isect(np.array([0.51,0]), M, eps, poly),
                          {False, True})

  np.testing.assert_equal(poly_ellipse_isect(np.array([1.55,0]), M, eps, poly),
                          {False})

  np.testing.assert_equal(poly_ellipse_isect(np.array([1,0]), M, eps, poly),
                          {False, True})

  M = np.diag([4,1])  # x-radius 0.25, y-radius 0.5 
  eps = 0.5 

  np.testing.assert_equal(poly_ellipse_isect(np.array([0.74,0.49]), M, eps, poly),
                          {True})

  np.testing.assert_equal(poly_ellipse_isect(np.array([0.76,0.49]), M, eps, poly),
                          {False, True})

  np.testing.assert_equal(poly_ellipse_isect(np.array([0.74,0.51]), M, eps, poly),
                          {False, True})

  np.testing.assert_equal(poly_ellipse_isect(np.array([-0.74,0.49]), M, eps, poly),
                          {True})

  np.testing.assert_equal(poly_ellipse_isect(np.array([-0.76,0.49]), M, eps, poly),
                          {False, True})

  np.testing.assert_equal(poly_ellipse_isect(np.array([-0.74,0.51]), M, eps, poly),
                          {False, True})

def test_tranformation():
  dim = 2
  A = np.eye(2) 
  B = np.eye(dim)
  W = np.array([[0,0],[0,0.4]])
  C = np.array([[1, 0],[0,1]]) 

  sys_lti = LTI(A, B, C, None, W=W)

  X = pc.box2poly(np.kron(np.ones((sys_lti.dim, 1)), np.array([[-10, 10]])))
  U = pc.box2poly(np.kron(np.ones((sys_lti.m, 1)), np.array([[-1, 1]])))
  sys_lti.setU(U) 
  sys_lti.setX(X)

  d = np.array([1., 1.])
  M = np.array([[ 1.00002, -0.,     ],
                [-0.,       1.00002]])
  K = np.array([[-0.99996, -0.     ],
                [-0.,      -0.99996]])
  eps = 1.4142261558177154

  abstr = LTIGrid(sys_lti, d, un=4, MKeps = (M, K, eps))


  xx = 20 * np.random.rand(2,10) - 10
  for i in range(10):
    x = xx[:,i]
    s_ab = abstr.x_to_s(x)

    x_out = abstr.mdp.transform_output(s_ab)

    np.testing.assert_equal(x_out[0], s_ab )
    np.testing.assert_array_less( np.abs(x_out[1] - x), d/2 * (1 + 1e-5) )

def test_lti():
  dim = 2
  A = np.eye(2) #np.array([[.9,-0.32],[0.1,0.9]])
  B = np.eye(dim)  #array([[1], [0.1]])
  W = np.array([[1,0],[0,1]]) #2*Tr.dot(np.eye(dim)).dot(Tr)  # noise on transitions
   
  # Accuracy
  C = np.array([[1, 0],[0,1]])  # defines metric for error (||y_finite-y||< epsilon with y= cx   )

  sys_lti = LTI(A, B, C, None, W=W)  # LTI system with   D = None

  X = pc.box2poly(np.kron(np.ones((sys_lti.dim, 1)), np.array([[-10, 10]])))
  U = pc.box2poly(np.kron(np.ones((sys_lti.m, 1)), np.array([[-1, 1]])))
  sys_lti.setU(U) # continuous set of inputs
  sys_lti.setX(X) # X space

  d = np.array([1., 1.])
  M = np.array([[ 1.00002, -0.,     ],
                [-0.,       1.00002]])
  K = np.array([[-0.99996, -0.     ],
                [-0.,      -0.99996]])
  eps = 1.4142261558177154

  abstr = LTIGrid(sys_lti, d, un=4, MKeps = (M, K, eps))

  for s in range(0, len(abstr), 7):
    np.testing.assert_equal( abstr.x_to_s(abstr.s_to_x(s) ), s )
    np.testing.assert_equal( s in abstr.x_to_all_s(abstr.s_to_x(s) ), True )

