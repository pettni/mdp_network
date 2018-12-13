import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import scipy.sparse as sp
from itertools import product
from collections import OrderedDict

from best.utils import *
from best import DTYPE, DTYPE_OUTPUT

class POMDP:
  """(Partially Observable) Markov Decision Process"""
  def __init__(self,
               T, 
               Z=[],
               input_names=['u'], 
               state_name='x', 
               output_name=None,
               input_trans=None,
               output_trans=None):
    '''
    Create a POMDP

    Below, M is the number of actions, N the number of states, O the number of outputs.

    Input arguments:
      T: dict of NxN matrices such that T[m_tuple][n,n'] = P(n' | n, m_tuple). 
      Z: dict of NxO matrices such that Z[m_tuple][n,o] = P(o | n,m_tuple )
        if len(Z) = 1, outputs do not depend on inputs
        if Z = [], it is an MDP (perfect observations)
      input_fcns:  input resolution functions: input_functions[i] : U_i -> range(M_i) 
      output_fcn: output labeling function: range(O) -> Y
      input_names: identifier for inputs
      state_name: identifier for state
      output_name: identifier for output

    Alphabets:
      inputs: range(m1) x range(m2) ... x range(mM)
      states: range(N)
      outputs: range(O)

      input  alphabet: U
      output alphabet: Y

    Control flow:
      U ---> range(M)  ---> range(N) ---> range(O) ---> Y
         ^              ^             ^             ^
         |              |             |             |
  input_trans        dynamics    observation     output_trans
    '''
    
    self._input_names = input_names
    self._state_name  = state_name
    self._output_name = output_name
    self._input_trans = input_trans
    self._output_trans = output_trans

    # Transition matrices for each axis
    self._Tmat_csr = {}
    self._Tmat_coo = {}

    # Convert to dict if necessary
    if type(T) != dict:
      T = dict(zip(range(len(T)), T))

    for key in T.keys():
      if type(key) != tuple:
        m_tuple = (key,)
      else:
        m_tuple = key

      self._Tmat_csr[m_tuple] = sp.csr_matrix(T[key], dtype=DTYPE)
      self._Tmat_coo[m_tuple] = sp.coo_matrix(T[key], dtype=DTYPE)

    if Z == []:
      # MDP
      self._Zmat_csr = []
      self._Zmat_coo = []
    elif len(Z) == 1:
      # does not depend on input
      self._Zmat_csr = [sp.csr_matrix(Z[0], dtype=DTYPE)]
      self._Zmat_coo = [sp.coo_matrix(Z[0], dtype=DTYPE)]
    else:
      self._Zmat_csr = {}
      self._Zmat_coo = {}

      if type(Z) != dict:
        Z = dict(zip(range(len(Z)), Z))

      for key in Z.keys():
        if type(key) != tuple:
          m_tuple = (key,)
        else:
          m_tuple = key

        self._Zmat_csr[m_tuple] = sp.csr_matrix(Z[key], dtype=DTYPE)
        self._Zmat_coo[m_tuple] = sp.coo_matrix(Z[key], dtype=DTYPE)

    self.check()

  @property
  def M(self):
    return tuple(1 + np.amax([np.array(m_tuple) for m_tuple in self._Tmat_coo.keys()], axis=0))

  @property
  def N(self):
    return next(iter(self._Tmat_csr.values())).shape[1]

  @property
  def O(self):
    if len(self._Zmat_csr) == 0:
      return self.N
    if len(self._Zmat_csr) == 1:
      return self._Zmat_csr[0].shape[1]
    return next(iter(self._Zmat_csr.values())).shape[1]
 
  @property
  def input_names(self):
    return tuple(self._input_names)

  @property
  def state_name(self):
    return self._state_name

  @property
  def output_name(self):
    if self.observable and self._output_name is None:
      return self._state_name
    else:
      return self._output_name

  @property
  def observable(self):
    return self._Zmat_csr == []

  @property
  def nnz(self):
    '''total number of stored transitions'''
    return sum(Tm.getnnz() for _, Tm in self._Tmat_csr.items())

  @property
  def sparsity(self):
    '''percentage of transitions'''
    return float(self.nnz) / (self.N**2 * sum(self.M))

  def m_tuple_iter(self):
    '''iterate over all inputs'''
    for m_tuple in product(*[list(range(k)) for k in self.M]):
      yield m_tuple

  def transform_output(self, o):
    '''return transformed output'''
    if self._output_trans is None:
      return o
    return self._output_trans(o)

  def transform_input(self, u):
    '''return transformed input'''
    if self._input_trans is None:
      return u
    return self._input_trans(u)

  def T(self, m_tuple):
    '''transition matrix for action tuple m_tuple'''
    return self._Tmat_csr[m_tuple]

  def Z(self, m_tuple):
    '''output matrix for action tuple m_tuple'''
    if self._Zmat_csr == []:
      # MDP
      return sp.identity(self.N, dtype=DTYPE, format='csr')
    if type(self._Zmat_csr) is not dict:
      # not depending on input
      return self._Zmat_csr[0]
    else:
      # input-dependent
      return self._Zmat_csr[m_tuple]

  def Tuz(self, m_tuple, z):
    '''belief transition matrix T^{u,z} = P ( x', z | x, u )'''
    Z_row = self.Z(m_tuple).getcol(z).todense().transpose()  # [1 x n]
    return self.T(m_tuple) * sp.spdiags(Z_row, 0, self.N, self.N)

  def check(self):
    for m_tuple in self._Tmat_csr.keys():
      if not self.T(m_tuple).shape == (self.N, self.N):
        raise Exception('T matrix not N x N')

      if not self.Z(m_tuple).shape == (self.N, self.O):
        raise Exception('Z matrix not N x O')

    if len(self.M) != len(self.input_names):
      raise Exception('Input names does not equal inputs')

    if prod(self.M) != len(self._Tmat_csr.keys()):
      raise Exception('Problem with inputs')

  def __str__(self):
    po = '' if self.observable else 'PO'

    ret = '{0}MDP: {1} inputs {2} --> {3} states {4} --> {5} outputs {6}' \
          .format(po, self.M, self.input_names, self.N, self.state_name, self.O, self.output_name)
    return ret

  def prune(self, thresh=1e-8):
    '''remove transitions with probability less than thresh and re-normalize'''
    for key_m, Tm in self._Tmat_csr.items():
      data = Tm.data
      indices = Tm.indices
      indptr = Tm.indptr
      data[np.nonzero(data < thresh)] = 0

      new_mat = sp.csr_matrix((data, indices, indptr), shape=Tm.shape)

      # diagonal matrix with row sums
      norms = new_mat.dot( np.ones(new_mat.shape[1]) )
      norms_mat = sp.coo_matrix((1/norms, (range(new_mat.shape[1]), range(new_mat.shape[1])))) 

      self._Tmat_csr[key_m] = norms_mat.dot(new_mat)

  def bellman(self, W, d=0):
    '''calculate Q function via one Bellman step
       Q(u, x) = \sum_x' T(x' | x, u) W(x')  '''
    Q = np.zeros(self.M + W.shape, dtype=DTYPE)
    for m_tuple in product(*[list(range(k)) for k in self.M]):
      Q[m_tuple] += sparse_tensordot(self.T(m_tuple), W, d)
    return Q

  def evolve_observe(self, state, m_tuple):
    '''draw a successor new_state for (state, m_tuple) and get observation obs'''
    succ_prob = np.asarray(self.T(m_tuple).getrow(state).todense()).ravel()
    new_state = np.random.choice(range(self.N), size=1, p=list(succ_prob))[0]
    obs_prob = np.asarray(self.Z(m_tuple).getrow(new_state).todense()).ravel()
    obs = np.random.choice(range(self.O), 1, p=list(obs_prob))[0]
    return new_state, obs

class POMDPNetwork:

  def __init__(self, pomdp_list=[]):

    self.pomdps = OrderedDict()  # POMDP state name -> POMDP
    self.connections = []        # (outputs, input, conn_matrix, deterministic)

    for pomdp in pomdp_list:
      self.add_pomdp(pomdp)

  def __str__(self):
    po = '' if self.observable else 'PO'
    return '{}MDP network: {} inputs {}, {} states {}, {} outputs {}' \
           .format(po, self.M, self.input_names, self.N, self.state_names, 
                   self.O, self.output_names)

  @property
  def N(self):
    return tuple(pomdp.N for pomdp in self.pomdps.values())

  @property
  def O(self):
    return tuple(pomdp.O for pomdp in self.pomdps.values())

  @property
  def M(self):
    in_s = self.__input_size()
    if len(in_s):
      _, sizes = zip(*in_s)
      return tuple(sizes)
    return tuple()

  @property
  def input_names(self):
    '''return list of global inputs'''
    in_s = self.__input_size()
    if len(in_s):
      inputs, _ = zip(*in_s)
      return tuple(inputs)
    return tuple()

  @property
  def state_names(self):
    return tuple(pomdp.state_name for pomdp in self.pomdps.values())

  @property
  def output_names(self):
    '''return list of global outputs'''
    return tuple(pomdp.output_name for pomdp in self.pomdps.values())

  @property
  def observable(self):
    return all(pomdp.observable for  pomdp in self.pomdps.values())

  @property
  def deterministic(self):
    return all(det for _,_,_,det in self.connections)
  
  def transform_output(self, o_tuple):
    return tuple(pomdp.transform_output(o) for (pomdp, o) in zip(self.pomdps.values(), o_tuple))

  def transform_input(self, u_tuple):
    return tuple(pomdp.transform_input(u) for (pomdp, u) in zip(self.pomdps.values(), u_tuple))

  def __input_size(self):
    all_inputs_size = [(input, Mi) for pomdp in self.pomdps.values() for (input, Mi) in zip(pomdp.input_names, pomdp.M)]
    connected_inputs = [input for _, input, _, _ in self.connections]
    return [input_size for input_size in all_inputs_size if input_size[0] not in connected_inputs]

  def add_pomdp(self, pomdp):

    if any(input_name in self.input_names for input_name in pomdp.input_names):
      raise Exception('input name collision')

    if pomdp.output_name in self.output_names:
      raise Exception('output name collision')

    if pomdp.state_name in self.state_names:
      raise Exception('state name collision')

    self.pomdps[pomdp.state_name] = pomdp

  def remove_pomdp(self, pomdp):
    self.pomdps.pop(pomdp.state_name, None)

  def get_node_with_output(self, output):
    for name, pomdp in self.pomdps.items():
      if pomdp.output_name == output:
        return name
    raise Exception('No POMDP with output', output)

  def get_node_with_input(self, input):
    for name, pomdp in self.pomdps.items():
      if input in pomdp.input_names:
        return name
    raise Exception('No POMDP with input', input)


  def add_connection(self, outputs, input, conn_fcn=None):

    if not type(outputs) is list:
      raise Exception('output must be list')

    if input not in self.input_names:
      raise Exception('invalid connection')

    # TODO: sort outputs as in overall network
    assert(outputs == sorted(outputs, key=lambda o: self.output_names.index(o)))

    in_name = self.get_node_with_input(input)
    out_names = [self.get_node_with_output(output) for output in outputs]

    in_pomdp = self.pomdps[in_name]
    nO_list = tuple(self.pomdps[out_name].O for out_name in out_names) # number of outputs
    m = in_pomdp.input_names.index(input)       # input dimension
    nM = in_pomdp.M[m]                          # number of inputs

    # compute bool connection matrix (input x output_list)
    conn_matrix = np.zeros((nM,) + nO_list, dtype=bool)
    deterministic = True

    for o_tuple in product(*[range(nO) for nO in nO_list]):
      # for each possible output

      u_list = conn_fcn(*[self.pomdps[out_name].transform_output(o) 
                          for o, out_name in zip(o_tuple, out_names) ])

      if not type(u_list) == set:
        raise Exception('connection map must return set')

      if len(u_list) == 0:
        raise Exception('connection empty for output {}'.format(o_tuple))

      if len(u_list) > 1:
        deterministic = False

      for u in u_list:
        u_real = in_pomdp.transform_input(u)
        if u_real < 0 or u_real >= nM:
          raise Exception('connection invalid for output {}'.format(o_tuple))
        conn_matrix[(int(u_real),) + o_tuple] = True

    self.connections.append((outputs, input, conn_matrix, deterministic))

    if len(list(nx.simple_cycles(self.construct_graph()))) > 0:
      raise Exception('connection graph can not have cycles')

  def construct_graph(self):
    G = nx.MultiDiGraph()

    G.add_node('in')

    for pomdp in self.pomdps.values():
      G.add_node(pomdp.state_name)

    for input_name in self.input_names:
      G.add_edge('in', self.get_node_with_input(input_name), input=input_name, output='')

    for outputs, input, _, _ in self.connections:
      for output in outputs:
        s0 = self.get_node_with_output(output)
        s1 = self.get_node_with_input(input)
        G.add_edge( s0, s1, input=input, output=output )
    return G

  def plot(self):
    G = self.construct_graph()

    pos = graphviz_layout(G, prog='dot')
  
    edge_labels = {(n1, n2) : '{}'.format(attr['input'])
                   for n1, n2, attr in G.edges(data=True)}

    nx.draw_networkx(G, pos=pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

  def predecessors(self, node):
    all_outputs = set([])
    for outputs, input, _, _ in self.connections:
      if input in self.pomdps[node].input_names:
        all_outputs |= set(outputs)
    return [self.get_node_with_output(output) for output in all_outputs]

  def successors(self, node):
    all_inputs = [input for outputs, input, _, _ in self.connections if self.pomdps[node].output_name in outputs]
    return [self.get_node_with_input(input) for input in all_inputs]

  def backwards_iter(self):
    '''walk backwards over network'''
    mark = {n: False for n in self.pomdps.keys()}
    cond = lambda n: not mark[n] and all(mark[sc] for sc in self.successors(n))

    try:
      while True:
        next_n = next(n for n in self.pomdps.keys() if cond(n))
        mark[next_n] = True
        yield next_n, self.pomdps[next_n]

    except StopIteration as e:
      return

  def forward_iter(self):
    '''walk forward over network'''
    mark = {n: False for n in self.pomdps.keys()}
    cond = lambda n: not mark[n] and all(mark[pd] for pd in self.predecessors(n))

    try:
      while True:
        next_n = next(n for n in self.pomdps.keys() if cond(n))
        mark[next_n] = True
        yield next_n, self.pomdps[next_n]
    except StopIteration as e:
      return

  def bellman(self, W):
    '''calculate dense Q function via one Bellman step
       Q(u_free, x) = E[ W(x') | x, u_free]'''

    slice_names = list(self.state_names)

    # Iterate bottom up 
    for name, pomdp in self.backwards_iter():

      # Do backup over current state
      W = pomdp.bellman(W, slice_names.index(name))
      slice_names = list(pomdp.input_names) + slice_names

      # Resolve connections (non-free actions)
      for outputs, input, conn_mat, deterministic in self.connections:

        if input not in pomdp.input_names:
          continue

        dim_u = slice_names.index(input)
        
        # reshape to same dim as W
        new_shape = np.ones(len(W.shape), dtype=np.uint32)
        new_shape[dim_u] = conn_mat.shape[0]

        for i, output in enumerate(outputs):
          state = self.get_node_with_output(output)
          new_shape[slice_names.index(state)] = conn_mat.shape[1+i]

        # cast to higher dim
        conn_mat = conn_mat.reshape(new_shape)

        W = np.maximum(np.amax(W)*(1-conn_mat), W).min(axis=dim_u)
        slice_names.remove(input)

    # reshuffle so inputs appear in order
    order = tuple(slice_names.index(u) for u in self.input_names) + \
            tuple(range(len(self.M), len(self.M) + len(self.N)))
    return W.transpose(order)

  def evolve(self, state, inputs):

    all_inputs = dict(zip(self.input_names, inputs))
    all_states = dict()
    all_outputs = dict()

    for name, pomdp in self.forward_iter():

      # index of current state
      idx = self.state_names.index(name)

      # find inputs to current pomdp and evolve
      pomdp_input_tuple = tuple(all_inputs[input_name] for input_name in pomdp.input_names)
      new_state, new_output = pomdp.evolve_observe(state[idx], pomdp_input_tuple)
      all_outputs[pomdp.output_name] = new_output
      all_states[pomdp.state_name] = new_state

      # add any intermediate input that is a function calculated outputs
      for outputs, input, conn_mat, _ in self.connections:
        if all([output in all_outputs.keys() for output in outputs]) and input not in all_inputs.keys():
          conn_output = tuple(all_outputs[output] for output in outputs)
          possible_input_values = np.nonzero(conn_mat[ (Ellipsis,) + conn_output ])
          all_inputs[input] = np.random.choice(possible_input_values[0])

    return [all_states[state] for state in self.state_names], \
           self.transform_output([all_outputs[output] for output in self.output_names])

  # OpenAI baselines fcns
  def step(self, inputs):
    pass

  def reset(self):
    pass

