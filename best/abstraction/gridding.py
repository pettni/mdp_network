import numpy as np
import polytope as pc
import itertools
import scipy.sparse as sp
import scipy.linalg as sl
from scipy.stats import norm

from best.utils import *
from best.models.pomdp import POMDP
from best.abstraction.simrel import eps_err

class Abstraction(object):

  def __init__(self, x_low, x_up):
    self.x_low = np.array(x_low, dtype=np.double).flatten()
    self.x_up = np.array(x_up, dtype=np.double).flatten()

  @property
  def dim(self):
    return len(self.x_low)

  def polytopic_predicate(self, x, poly):
    '''evaluate a polytopic predicate at x'''
    return {x in poly}

  @property
  def N(self):
    raise NotImplementedError

  def s_to_x(self, s):
    raise NotImplementedError

  def x_to_s(self, x):
    raise NotImplementedError

  def interface(self, u_ab, s_ab, x):
    raise NotImplementedError

class Grid(Abstraction):

  def __init__(self, x_low, x_up, n_list, name_prefix=''):
    ''' Create grid abstraction of \prod_i [x_low[i], x_up[i]] with n_list[i] discrete states 
      in dimension i, and with 1-step movements'''
    super().__init__(x_low, x_up)

    self.n_list = n_list
    self.eta_list = (self.x_up - self.x_low)/np.array(self.n_list)

    self.abstract(name_prefix)

  @property
  def N(self):
    return prod(self.n_list)

  def s_to_x(self, s):
    '''center of cell s'''
    return self.x_low + self.eta_list/2 + self.eta_list * np.unravel_index(s, self.n_list)

  def x_to_s(self, x):
    '''closest abstract state to x'''
    if not np.all(x.flatten() < self.x_up) and np.all(x.flatten() > self.x_low):
        raise Exception(x, 'x outside abstraction domain', self.x_low, self.x_up)
    midx = (np.array(x).flatten() - self.x_low)/self.eta_list
    return np.ravel_multi_index( tuple(map(np.int, midx)), self.n_list)

  def interface(self, u_ab, s_ab, x):
    '''return target point for given abstract control and action'''
    sp, _ = self.pomdp.evolve_observe(s_ab, u_ab)
    return self.s_to_x( sp )

  def plot(self, ax):
    xy_t = np.array([self.s_to_x(s) for s in range(prod(self.n_list))])

    ax.scatter(xy_t[:,0], xy_t[:,1], label='Finite states', color='k', s=10, marker="o")
    ax.set_xlim(self.x_low[0], self.x_up[0])
    ax.set_ylim(self.x_low[1], self.x_up[1])

  def abstract(self, name_prefix):

    def move(s0, dim, direction):
      # which state is in direction along dim from s0?
      midx_s0 = np.unravel_index(s0, self.n_list)
      midx_s1 = list(midx_s0)
      midx_s1[dim] += direction
      midx_s1[dim] = max(0, midx_s1[dim])
      midx_s1[dim] = min(self.n_list[dim]-1, midx_s1[dim])
      return np.ravel_multi_index(midx_s1, self.n_list)

    T_list = [sp.eye(self.N)]
    for d in range(len(self.n_list)):
      vals = np.ones(self.N)
      n0 = np.arange(self.N)
      npl = [move(s0, d,  1) for s0 in np.arange(self.N) ]
      npm = [move(s0, d, -1) for s0 in np.arange(self.N) ]

      T_pm = sp.coo_matrix((vals, (n0, npm)), shape=(self.N, self.N))
      T_list.append(T_pm)

      T_pl = sp.coo_matrix((vals, (n0, npl)), shape=(self.N, self.N))
      T_list.append(T_pl)

    self.pomdp = POMDP(T_list, 
                       input_names=[name_prefix + '_u'], 
                       state_name=name_prefix + '_s', 
                       output_trans=self.s_to_x, 
                       output_name=name_prefix + '_x')


class LTIGrid(Grid):

  def __init__(self, lti_syst, eta, un=3, T2x=None, MKeps=None):
    '''Construct a grid abstraction of a LTI Gaussian system
    :param lti_syst: A LTI system (noise matrix must be diagonal)
    :param eta: abstraction grid size (one for each dimension)
    :param un: number of discrete inputs per dimension
    :param T2x=None: transformation matrix (use for rotated systems for easy access to original coordinates)
    :param MKeps=None: tuple (M, K, eps) defining a simulation relation. if None one will be computed
    '''
    # check that W is diagonal
    if not np.all(lti_syst.W == np.diag(np.diagonal(lti_syst.W))):
      raise Exception('system noise must be diagonal')
    
    # store state transformation matrix
    if lti_syst.T2x is None:
      self.T2x = np.eye(lti_syst.dim)  # identity
    else:
      self.T2x = lti_syst.T2x

    # compute/store simulation relation
    if MKeps is None:
      dist = pc.box2poly(np.diag(eta).dot(np.kron(np.ones((lti_syst.dim, 1)), np.array([[-1, 1]]))))
      self.M, self.K, self.eps = eps_err(lti_syst, dist)
    else:
      self.M = MKeps[0]
      self.K = MKeps[1]
      self.eps = MKeps[2]

    # state discretization information
    lx, ux = pc.bounding_box(lti_syst.X)
    lx = lx.flatten()
    ux = ux.flatten()

    remainx = eta - np.remainder(ux-lx, eta)  # center slack
    lx -= remainx/2
    ux += remainx/2

    self.x_low = lx
    self.x_up = ux

    self.eta_list = eta.flatten()
    self.n_list = tuple(np.ceil((self.x_up - self.x_low)/self.eta_list).astype(int))

    # save input discretization information: place inputs on boundary
    # NOTE: bounding box may give infeasible inputs..
    lu, uu = pc.bounding_box(lti_syst.U)  

    self.u_low = lu.flatten()
    self.m_list = tuple(un for i in range(lti_syst.m))
    self.eta_u_list = (uu.flatten() - self.u_low)/(np.array(self.m_list)-1)

    transition_list = [np.zeros((self.N+1, self.N+1)) for m in range(prod(self.m_list))]  # one dummy state

    # extract all transitions
    for ud in range(prod(self.m_list)):

      Pmat = np.zeros((self.N+1, self.N+1))
      for s in range(self.N):

        s_diag = super(LTIGrid, self).s_to_x(s)
        mean = np.dot(lti_syst.a, s_diag) + np.dot(lti_syst.b, self.ud_to_u(ud))  # Ax

        P = np.ravel(grid_cdf_nd(mean, lti_syst.W, self.x_low, self.x_up, self.eta_list))

        Pmat[s, 0:self.N] = P
        Pmat[s, self.N] = 1 - sum(P) 

      Pmat[self.N, self.N] = 1

      transition_list[ud] = Pmat

    self.mdp = POMDP(transition_list, input_names=['u_d'], state_name='s', 
                     output_trans=lambda s: (s, self.s_to_x(s)), output_name='(s,xc)')

  def __len__(self):
    return prod(self.n_list)

  def ud_to_u(self, ud):
    return self.u_low + self.eta_u_list * np.unravel_index(ud, self.m_list)

  def s_to_x(self, s):
    '''return center of cell s'''
    if s == len(self):
      return None # the dummy state
    if s < 0 or s > len(self):
      raise Exception('s={} outside range'.format(s))
    return self.transform_d_o(super(LTIGrid, self).s_to_x(s))

  def x_to_s(self, x):
    '''compute closest abstract state'''
    x_diag = self.transform_o_d(x.flatten())
    if np.any(x_diag < self.x_low) or np.any(x_diag > self.x_up):
      return None  # outside domain
    return super(LTIGrid, self).x_to_s(x_diag)

  def x_to_all_s(self, x):
    '''compute abstract states that are related to x via simulation relation'''

    # we do stepping in diagonal space
    x_diag = self.transform_o_d(x.flatten())
    ret = set([super(LTIGrid, self).x_to_s(x_diag)])  # closest one

    sz = len(ret)

    # search in expanding squares
    radius = 1
    while True:

      ranges = [np.arange(x_diag[i]-radius*self.eta_list[i], 
                          x_diag[i]+radius*self.eta_list[i], 
                          self.eta_list[i] )
                for i in range(self.dim)]

      for x_t_iter in itertools.product(*ranges):
        x_t = np.array(x_t_iter)
        if np.any(x_t < self.x_low) or np.any(x_t > self.x_up):
          continue  # not in domain

        if (x_t - x_diag).dot(self.M).dot(x_t - x_diag) > self.eps**2:
          continue # not in relation
        ret |= set([ super(LTIGrid, self).x_to_s(x_t) ])
      
      if len(ret) == sz:
        # nothing new was added
        break

      sz = len(ret)

      radius += 1

    return list(ret)

  def interface(self, ud, s, x):
    '''refine abstract input ud to concrete input'''

    u = self.ud_to_u(ud)
    
    # give zero if in dummy state
    if s == self.mdp.N - 1:
      return np.zeros((len(self.input_cst[0]),1))

    x_s = self.s_to_x(s)

    return self.K.dot(x.flatten() - x_s) + u

  def transform_o_d(self, x):
    '''transform from original to diagonal coordinates'''
    return np.linalg.inv(self.T2x).dot(x)

  def transform_d_o(self, x_diag):
    '''transform from diagonal to original coordinates'''
    return self.T2x.dot(x_diag)

  def polytopic_predicate(self, x, poly):
    '''determine whether a state x is inside/outside a polytopic region poly.
       returns subset of {False, True}'''

    if x is None: 
      # capture dummy state
      return set([False])

    if (self.eps is None) or (self.eps == 0):
      # no epsilon error
      return super(LTIGrid, self).polytopic_predicate(x, poly)

    else:
      # must account for epsilon
      # move to transformed coordinates (since this is where sim relation is defined)
      poly_trans = pc.Polytope( poly.A.dot(self.T2x), poly.b )
      x_trans = self.transform_o_d(x)

      return poly_ellipse_isect(x_trans, self.M, self.eps, poly_trans)

###############################################
###############################################

def grid_cdf_1d(m, S, low, up, eta):
  '''compute how much of a 1D gaussian CDF that falls into cells on a grid
     returns p such that p[i] = P( low + eta*i <= x <= low + eta*(i+1)  ) for x ~ N(m, S)'''

  N = np.round((up - low) / eta).astype(int)

  if S < np.finfo(np.float32).eps:
    # no variance
    Pvec = np.zeros(N)
    if m > low and m < up: 
      Pvec[ int((m - low)/eta) ] = 1
  else:
    # there is variance
    edges = np.linspace(low, up, N)
    Pvec = np.diff(norm.cdf(edges, m, S ** .5))
  return Pvec

def grid_cdf_nd(m_tuple, S_mat, low_tuple, up_tuple, eta_tuple):
  '''compute how much of a nd gaussian CDF that falls into cells on a nd grid
     returns P such that p[i] = P( low + eta*i <= x <= low + eta*(i+1)  ) for x ~ N(m, S)'''

  if not np.all(S_mat == np.diag(np.diagonal(S_mat))):
    raise Exception('variance must be diagonal')

  ret = 1
  for m, S, low, up, eta in zip(m_tuple, S_mat.diagonal(), low_tuple, up_tuple, eta_tuple):
    ret = np.outer(ret, grid_cdf_1d(m, S, low, up, eta))

  print(ret.shape)
  return ret

def poly_ellipse_isect(x0, M, eps, poly):
  '''For an ellipse C = {x : (x-x0)' M (x-x0) <= eps^2 } and a polyhedron poly, return
     {True} if C \subset poly
     {False} if C \subset complement(poly)
     {True, False} otherwise  '''

  sqrtM = sl.sqrtm(M)

  # Transformed
  x0_p = sqrtM.dot(x0)

  # Normalize transformed matrix
  poly_p = pc.Polytope(poly.A.dot( sl.inv( sqrtM ) ), poly.b, normalize=True)

  ret = set()
  if np.all( poly_p.A.dot(x0_p) <= poly_p.b + eps ):
    ret |= set([True])  # at least partly inside
  if not np.all( poly_p.A.dot(x0_p) <= poly_p.b - eps ):
    ret |= set([False]) # at least partly outside

  if not len(ret):
    raise Exception('error in poly_ellipse_isect, result empty')

  return ret


