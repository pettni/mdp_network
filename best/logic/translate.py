import numpy as np
from itertools import product

from best.logic.fsa import Fsa
from best.models.pomdp import POMDP

def formula_to_logic(formula):
  '''convert propsitional logic formula to a logic gate
     represented as a special case of a POMDP'''

  fsa = Fsa()
  fsa.from_formula(formula)

  T = dict(zip(product(*[range(2) for k in range(len(fsa.props))]),
               [np.array([[1,0], [1,0]]) for k in range(2**len(fsa.props))] ))

  init_state = next(s for (s,k) in fsa.init.items() if k==1)
  final_state = next(s for s in fsa.final)

  input_names = sorted(fsa.props.keys(), key = lambda key: -fsa.props[key])

  for u in fsa.g[init_state][final_state]['input']:
    m_tuple = tuple(map(int, tuple(format(u, '0{}b'.format(len(fsa.props))))))
    T[m_tuple] = np.array([[0, 1], [0, 1]])

  return POMDP(T, input_names=input_names, state_name='_'.join(input_names))


def formula_to_pomdp(formula):
  '''convert a co-safe LTL formula to a DFSA represented as a   
  special case of a POMPD'''
  
  fsa = Fsa()
  fsa.from_formula(formula)
  fsa.add_trap_state()

  # mapping state -> state index
  N = len(fsa.g)
  dict_fromstate = dict([(sstate, s) for s, sstate in enumerate(sorted(fsa.g.nodes()))])

  inputs = set.union(*[attr['input'] for _,_,attr in fsa.g.edges(data=True)])
  M = len(inputs)
  assert(inputs == set(range(M)))

  T = dict(zip(product(*[range(2) for k in range(len(fsa.props))]),
               [np.zeros((N, N)) for k in range(M)] ))

  input_names = sorted(fsa.props.keys(), key = lambda key: -fsa.props[key])

  for (s1, s2, attr) in fsa.g.edges(data=True):
    for u in attr['input']:
      # get binary representation
      m_tuple = tuple(map(int, tuple(format(u, '0{}b'.format(len(fsa.props))))))

      # check that input_names are in correct order
      test_props = set([input_names[i] for i in range(len(input_names))
                       if m_tuple[i]])
      assert u == fsa.bitmap_of_props(test_props)

      T[m_tuple][dict_fromstate[s1], dict_fromstate[s2]] = 1

  mdp = POMDP(T, input_names=input_names, state_name='mu')

  init_states = set(map(lambda state: dict_fromstate[state], [state for (state, key) in fsa.init.items() if key == 1]))
  final_states = set(map(lambda state: dict_fromstate[state], fsa.final))

  return mdp, init_states, final_states


