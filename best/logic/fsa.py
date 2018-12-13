import networkx as nx
import re
import subprocess as subp
import tempfile
import numpy as np
import os

import pkg_resources as pr
import sys


if sys.platform[0:5] == 'linux':
  __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
  ltl2ba_binary = os.path.join(__location__, 'binaries', 'linux','ltl2ba')
  scheck_binary = os.path.join(__location__, 'binaries', 'linux','scheck2')
elif sys.platform == 'darwin':
  __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
  ltl2ba_binary = os.path.join(__location__, 'binaries', 'mac','ltl2ba')
  scheck_binary = os.path.join(__location__, 'binaries', 'mac','scheck2')
else:
  print ('%s platform not supported yet!' % sys.platform)
  exit(1)


class Fsa(object):
  """
  Base class for deterministic finite state automata.
  """

  def __init__(self):
    """
    Empty LOMAP Model object constructor.
    """
    self.g = nx.DiGraph()
    self.init = dict()
    self.final = set()

  @staticmethod
  def infix_formula_to_prefix(formula):
    # This function expects a string where operators and parantheses 
    # are seperated by single spaces, props are lower-case.
    #
    # Tokenizes and reverses the input string.
    # Then, applies the infix to postfix algorithm.
    # Finally, reverses the output string to obtain the prefix string.
    #
    # Infix to postfix algorithm is taken from:
    # http://www.cs.nyu.edu/courses/fall09/V22.0102-002/lectures/InfixToPostfixExamples.pdf
    # http://www.programmersheaven.com/2/Art_Expressions_p1
    #
    # Operator priorities are taken from:
    # Principles of Model Checking by Baier, pg.232
    # http://www.voronkov.com/lics_doc.cgi?what=chapter&n=14
    # Logic in Computer Science by Huth and Ryan, pg.177

    # Operator priorities (higher number means higher priority)
    operators = { "I": 0, "|" : 1, "&": 1, "U": 2, "G": 3, "F": 3, "X": 3, "!": 3};
    output = []
    stack = []

    # Remove leading, trailing, multiple white-space, and
    # split string at whitespaces
    formula = re.sub(r'\s+',' ',formula).strip().split()

    # Reverse the input
    formula.reverse()

    # Invert the parantheses
    for i in range(0,len(formula)):
      if formula[i] == '(':
        formula[i] = ')'
      elif formula[i] == ')':
        formula[i] = '('

    # Infix to postfix conversion
    for entry in formula:

      if entry == ')':
        # Closing paranthesis: Pop from stack until matching '('
        popped = stack.pop()
        while stack and popped != '(':
          output.append(popped)
          popped = stack.pop()

      elif entry == '(':
        # Opening paranthesis: Push to stack
        # '(' has the highest precedence when in the input
        stack.append(entry)

      elif entry not in operators:
        # Entry is an operand: append to output
        output.append(entry)

      else:
        # Operator: Push to stack appropriately
        while True:
          if not stack or stack[-1] == '(':
            # Push to stack if empty or top is '('
            # '(' has the lowest precedence when in the stack
            stack.append(entry)
            break
          elif operators[stack[-1]] < operators[entry]:
            # Push to stack if prio of top of the stack
            # is lower than the current entry
            stack.append(entry)
            break
          else:
            # Pop from stack and try again
            popped = stack.pop()
            output.append(popped)

    # Pop remaining entries from the stack
    while stack:
      popped = stack.pop()
      output.append(popped)

    # Reverse the order and join entries w/ space
    output.reverse()
    formula = ' '.join(output)

    return formula

  def from_formula(self, formula):

    # scheck expects a prefix co-safe ltl formula w/ props: p0, p1, ...

    # Get the set of propositions
    props = re.sub('[IGFX!\(\)&|U]', ' ', formula)

    # TODO: implement true/false support
    props = set(re.sub('\s+', ' ', props).strip().split())


    # Form the bitmap dictionary of each proposition
    # Note: range goes upto rhs-1
    self.props = dict(zip(props, map(lambda x: 2 ** x, range(0, len(props)))))
    self.final = set()
    self.init = {}

    # Alphabet is the power set of propositions, where each element
    # is a symbol that corresponds to a tuple of propositions
    # Note: range goes upto rhs-1
    self.alphabet = set(range(0, 2 ** len(self.props)))

    # Prepare from/to scheck conversion dictionaries
    i = 0
    to_scheck = dict()
    from_scheck = dict()
    for p in props:
      scheck_p = 'p%d'%i
      from_scheck[scheck_p] = p
      to_scheck[p] = scheck_p
      i += 1

    # Convert infix to prefix
    scheck_formula = Fsa.infix_formula_to_prefix(formula)

    # Scheck expect implies operator (I) to be lower-case
    scheck_formula = ''.join([i if i != 'I' else 'i' for i in scheck_formula])

    # Convert formula props to scheck props
    for k,v in to_scheck.items():
      scheck_formula = scheck_formula.replace(k, v)

    # Write formula to temporary file to be read by scheck
    tf = tempfile.NamedTemporaryFile()
    if sys.version_info >= (3, 0):
      tf.write(bytes(scheck_formula, 'utf-8'))
    else:
      tf.write(bytes(scheck_formula).encode("utf-8")) 
    tf.flush()

    # Execute scheck and get output
    try:
      lines = subp.check_output([scheck_binary, '-s', '-d', tf.name]).splitlines()
    except Exception as ex:
      raise Exception(__name__, "Problem running %s: '%s'" % (scheck_binary, ex))

    # Close temp file (automatically deleted)
    tf.close()

    # Convert lines to list after leading/trailing spaces
    lines = list(map(lambda x: x.decode('utf-8').strip(), lines))

    # 1st line: "NUM_OF_STATES NUM_OF_ACCEPTING_STATES"
    # if NUM_OF_ACCEPTING_STATES is 0, all states are accepting
    l = lines.pop(0)
    state_cnt, final_cnt = map(int, l.split())
    # Set of remaining states
    rem_states = set(['%s'%i for i in range(0,state_cnt)])
    # Parse state defs
    while True:
      # 1st part: "STATE_NAME IS_INITIAL -1" for regular states
      # "STATE_NAME IS_INITIAL ACCEPTANCE_SET -1" for final states
      l = lines.pop(0).strip().split()
      src = l[0]
      is_initial = True if l[1] != '0' else False
      is_final = True if len(l) > 3 else False

      # Mark as done
      rem_states.remove(src)

      # Mark as initial/final if required
      if is_initial:
        self.init[src] = 1
      if is_final:
        self.final.add(src)

      while True:
        # 2nd part: "DEST PREFIX_GUARD_FORMULA" 
        l = lines.pop(0).strip().split()
        if l == ['-1']:
          # Done w/ this state
          break
        dest = l[0]
        l.pop(0)
        guard = ''
        # Now l holds the guard in prefix form
        if l == ['t']:
          guard = '(1)'
        else:
          l.reverse()
          stack = []
          for e in l:
            if e in ['&', '|']:
              op1 = stack.pop()
              op2 = stack.pop()
              stack.append('(%s %s %s)' % (op1, e, op2))
            elif e == '!':
              op = stack.pop()
              stack.append('!%s' % (op))
            else:
              stack.append(e)
          guard = stack.pop()

        # Convert to regular props
        for k,v in from_scheck.items():
          guard = guard.replace(k,v)
        bitmaps = self.get_guard_bitmap(guard)
        self.g.add_edge(src, dest, weight=0, input=bitmaps, guard=guard, label=guard)

      if not rem_states:
        break

    # We expect a deterministic FSA
    assert(len(self.init)==1)

    return

  def get_guard_bitmap(self, guard):

    # Get sets for all props
    for key in self.props:
      guard = re.sub(r'\b%s\b' % key, "self.symbols_w_prop('%s')" % key, guard)

    # Handle (1)
    guard = re.sub(r'\(1\)', 'self.alphabet', guard)

    # Handler negated sets
    guard = re.sub('!self.symbols_w_prop', 'self.symbols_wo_prop', guard)

    # Convert logic connectives
    guard = re.sub(r'\&\&', '&', guard)
    guard = re.sub(r'\|\|', '|', guard)

    bitmaps = eval(guard)

    return bitmaps

  def add_trap_state(self):

    # Figure out if trap state is needed
    need_trap = False
    for s, s_nbrs in self.g.adjacency():
      if self.alphabet != set.union(*[e_attr['input'] for _, e_attr in s_nbrs.items()]):
        need_trap = True
        break

    # Add trap state
    if need_trap:
      self.g.add_edge('trap', 'trap', input=self.alphabet)

      for s, s_nbrs in self.g.adjacency():
        defined_actions = set.union(*[e_attr['input'] for _, e_attr in s_nbrs.items()])
        remaining_actions = self.alphabet-defined_actions
        if len(remaining_actions):
          self.g.add_edge(s, 'trap', input=remaining_actions)

    # Ensure all actions enabled at every node
    for s, s_nbrs in self.g.adjacency():
      assert self.alphabet == set.union(*[e_attr['input'] for _, e_attr in s_nbrs.items()])
      if len(s_nbrs) > 1:
        assert set([]) == set.intersection(*[e_attr['input'] for _, e_attr in s_nbrs.items()])



  def symbols_w_prop(self, prop):
    return set(filter(lambda symbol: True if self.props[prop] & symbol else False, self.alphabet))

  def symbols_wo_prop(self, prop):
    return self.alphabet.difference(self.symbols_w_prop(prop))

  def bitmap_of_props(self, props):
    prop_bitmap = 0
    for x in map(lambda p: self.props.get(p, 0), props):
      prop_bitmap |= x
    return prop_bitmap

  def next_states_of_fsa(self, q, props):
    # Get the bitmap representation of props
    prop_bitmap = self.bitmap_of_props(props)

    # Return an array of next states
    return filter(lambda x: True if x is not None else False,
              # next state if bitmap is in inputs else None
              map(lambda e: e[1] if prop_bitmap in e[2]['input'] else None,
              # Get all edges from q
              self.g.out_edges_iter(q,True)))