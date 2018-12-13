import sparse
import numpy as np

def diagonal(a, axis1, axis2):
  '''perform diagonal operation on tensor a:
    Ex: diagonal over x,z on A_xyzw gives Bxzy (new dimension appended at the end)
    Analogous to np.diagonal'''

  if a.shape[axis1] != a.shape[axis2]:
    raise Exception('dimensions must agree for diagonal')

  new_axis_order = [axis for axis in range(len(a.shape)) if axis != axis1 and axis != axis2] + [axis1]

  new_shape = [a.shape[axis] for axis in new_axis_order]

  idx_diag = [i for i in range(len(a.data)) if a.coords[axis1][i] == a.coords[axis2][i]]

  new_coord = [[a.coords[axis][i] for i in idx_diag] for axis in new_axis_order]
  new_data  = [a.data[i] for i in idx_diag]

  return sparse.COO(new_coord, new_data, new_shape)


def diag(a, axes):
  '''diagonalize axis by inserting new dimension at the end'''
  raise NotImplementedError


def propagate_distribution(pomdp, D_ux, u_dim=None, x_dim=None):
  '''evolve input/state distribution D_ux into output distribution D_xz
    D_xz(x', z) = \sum_{x', u) P(X+ = x, Z = z | U = u X = x' ) D_ux(u, x')
  '''
  
  if u_dim is None:
    u_dim = tuple(range(len(pomdp.M)))

  if x_dim is None:
    x_dim = (len(pomdp.M),)

  if len(u_dim) != len(pomdp.M) or len(x_dim) != 1:
    raise Exception('dimension problem')

  if len(D_ux.shape) <= max(u_dim + x_dim) or len(set(u_dim + x_dim)) < len(u_dim + x_dim) or sum(D_ux.data) != 1:
      raise Exception('D_ux not a valid distribution')
    
  T_uxXz = sparse.stack([sparse.stack([sparse.COO(pomdp.Tuz(m_tuple, z)) 
                                       for z in range(pomdp.O)],
                                      axis=-1)
                         for m_tuple in pomdp.m_tuple_iter()]) \
           .reshape(pomdp.M + (pomdp.N, pomdp.N, pomdp.O))
  
  T_zx = sparse.tensordot(D_ux, T_uxXz, axes=(u_dim + x_dim, range(len(pomdp.M)+1))  )
  
  return sparse.COO(T_zx)


def propagate_network_distribution(network, D):
  ''' evolve input/state distribution D into output distribution D_xz
    D_xz(x', z) = \sum_{x', u) P(X+ = x, Z = z | U = u X = x' ) D(u, x')
  '''
  if not network.deterministic:
    raise Exception('not possible in nondeterministic case')

  if not D.shape == network.M + network.N:
    raise Exception('wrong dimension of D')

  slice_names = list(network.input_names) + list(network.state_names)
  treated_inputs = [] 

  for name, pomdp in network.forward_iter():
    # index of current state
    u_dim = tuple(slice_names.index(u_name) for u_name in pomdp.input_names)
    x_dim = (slice_names.index(name),)

    # D_ux -> D_xz
    D = propagate_distribution(pomdp, D, u_dim=u_dim, x_dim=x_dim)   # get old  z_out
    slice_names = [n for n in slice_names if n not in (name,) + pomdp.input_names] + \
                  [name] + [pomdp.output_name]
    treated_inputs += pomdp.input_names

    # D_xz -> D_ux
    for outputs, inp, D_uz, _ in network.connections:

      # ALTERNATIVE WAY: first diagonalize D_xz to D_xzz, then use tensordot(D_xzz, D_uz) on z
      # should be faster: diagonalization is very cheap, avoids own diagonal() fcn
      # diagonalization here: https://github.com/nils-werner/sparse/compare/master...nils-werner:einsum#diff-774b84c1fc5cd6b86a14c41931aca83bR356

      if all([output in slice_names for output in outputs]) and inp not in treated_inputs:
        # outer tensor product -- D_xzuz 
        D = sparse.tensordot(D, sparse.COO(D_uz), axes=0)
        slice_names = slice_names + [inp] + [output + '_z' for output in outputs]

        # diagonal over each z -- Dxuz
        for output in outputs:
          D = diagonal(D, axis1=slice_names.index(output), axis2=slice_names.index(output + '_z') )
          slice_names = [name for name in slice_names if name != output and name != output + '_z'] + [output]

  new_order = [slice_names.index(x) for x in network.state_names] + [slice_names.index(z) for z in network.output_names]
  return D.transpose(new_order)


def evaluate_Q(network, m_tuple, n_tuple, W):
  '''calculate E[ W(x') | n_tuple, m_tuple]
  for a given n_tuple, m_tuple, and where W is ndarray-like (e.g. TensorWrapper works)
  '''

  # create delta distribution and propagate it through network
  D_ux = sparse.COO([ [i] for i in m_tuple + n_tuple], [1], shape=network.M + network.N)
  D_xz = propagate_network_distribution(network, D_ux)

  # sum over probabilities to get expected value of W
  return sum( D_xz.data[i] * W[tuple(D_xz.coords[k][i] for k in range(len(network.N)))] for i in range(len(D_xz.data)) )
