'''Methods for solving reachability problems via occupation measure LPs

Currently only support two-part networks (e.g., MDP + automaton) with potentially
nondeterministic connections
'''
import numpy as np
import sparse
import scipy.sparse as sp

from best.models.pomdp_sparse_utils import *
from best.solvers.optimization_wrappers import Constraint, solve_ilp


def solve_exact(P_asS, P_lqQ, conn_mat, s0, q0, q_target):

  na = P_asS.shape[0]
  ns = P_asS.shape[1]
  nl = P_lqQ.shape[0]
  nq = P_lqQ.shape[1]
  nk = nl*2+2

  if q_target != nq-1:
    raise Exception("case q_target not equal to highest q not implemented")

  idx_notarget = [i for i in range(nq) if i != q_target]
  idx_target = [q_target]
  P_lqQ_notarget = P_lqQ[:, :, idx_notarget]
  P_lqQ_notarget = P_lqQ_notarget[:, idx_notarget, :]  # transitions from Q \ T to Q \ T
  P_lqQ_sep = P_lqQ[:, idx_notarget, :] 
  P_lqQ_sep = P_lqQ_sep[:, :, idx_target]              # transitions from Q \ T to T

  nq_notarget = nq - len(idx_target)

  num_varP = 1                      # objective (reach probability)
  num_varX = na * ns * nq_notarget  # x variable (occupation measures)

  P_lS = sparse.COO(conn_mat)

  ##################
  # Constraint 12b #
  ##################

  # right-hand side
  sum_a = sparse.COO([range(na)], np.ones(na))
  R_aSsQq = sparse.tensordot(sum_a, sparse.tensordot(sparse.eye(ns), sparse.eye(nq_notarget), axes=-1), axes=-1) #
  R_b_QS_asq = R_aSsQq.transpose([3, 1, 0, 2, 4]).reshape([ns * nq_notarget, na * ns * nq_notarget])

  # left-hand side
  L_SlQasq = sparse.tensordot(P_asS, P_lqQ_notarget, axes=-1).transpose([2, 3, 5, 0, 1, 4]) #
  L_SQasqS = sparse.tensordot(L_SlQasq, P_lS, axes=[[1], [0]])
  L_QSasq = diagonal(L_SQasqS, axis1=0, axis2=5).transpose([0,4,1,2,3])
  L_QS_asq_sp = L_QSasq.reshape([nq_notarget * ns, na * ns * nq_notarget]).to_scipy_sparse()  

  # TODO: indexing needs fix to have q_target not being the last one
  b_iq_b = np.zeros(ns * nq_notarget)
  b_iq_b[np.ravel_multi_index((s0, q0), (ns, nq_notarget))] = 1.

  c = Constraint(A_iq=sp.bmat([[sp.coo_matrix((ns * nq_notarget, num_varP)), 
                                R_b_QS_asq.to_scipy_sparse() - L_QS_asq_sp]]), 
                 b_iq=b_iq_b)

  ##################
  # Constraint 12c #
  ##################

  # right-hand side
  R_e_asSlqQ = sparse.tensordot(P_asS, P_lqQ_sep, axes=-1)     #
  R_e_Slasq = R_e_asSlqQ.sum(axis=[5]).transpose([2,3,0,1,4])  #
  R_asq = sparse.tensordot(R_e_Slasq, P_lS, axes=[[1, 0], [0, 1]])
  R_asq_sp = R_asq.reshape([1, na*ns*nq_notarget]).to_scipy_sparse()

  c &= Constraint(A_iq=sp.bmat([[1, -R_asq_sp]]), b_iq=[0])

  ##################
  #### Solve it ####
  ##################

  objective = np.zeros(num_varP + num_varX)
  objective[0] = -1  # maximize P

  sol = solve_ilp(objective, c, J_int=[])

  if sol['status'] == 'optimal':
    return sol['x'][0], sol['x'][num_varP: num_varP+num_varX].reshape((na, ns, nq_notarget))
  else:
    print("solver returned {}".format(sol['status']))
    return -1, -1


def solve_robust(P_asS, P_lqQ, conn_mat, s0, q0, q_target):
  # formulate and solve robust LP for system 
  # consisting of two mdps P_{uxx'} and Q_{vyy'}, 
  # where x' yields (nondeterministic) inputs v

  na = P_asS.shape[0]
  ns = P_asS.shape[1]
  nl = P_lqQ.shape[0]
  nq = P_lqQ.shape[1]
  nk = nl*2+2

  if q_target != nq-1:
    raise Exception("case q_target not equal to highest q not implemented")

  idx_notarget = [i for i in range(nq) if i != q_target]
  idx_target = [q_target]
  P_lqQ_notarget = P_lqQ[:, :, idx_notarget]
  P_lqQ_notarget = P_lqQ_notarget[:, idx_notarget, :]  # transitions from Q \ T to Q \ T
  P_lqQ_sep = P_lqQ[:, idx_notarget, :] 
  P_lqQ_sep = P_lqQ_sep[:, :, idx_target]              # transitions from Q \ T to T

  nq_notarget = nq - len(idx_target)

  num_varP = 1                      # objective (reach probability)
  num_varX = na * ns * nq_notarget  # x variable (occupation measures)
  num_var1 = nk * ns * nq_notarget  # dummy variable 1
  num_var2 = nk * ns                # dummy variable 2

  # Extract uncertainty polyhedron
  A_poly_Skl = sparse.stack([sparse.COO(np.vstack([np.eye(nl), -np.eye(nl), np.ones([1, nl]), -np.ones([1, nl])])) for state in range(ns)])
  b_poly_Sk = sparse.stack([sparse.COO(np.hstack([conn_mat[:, state], np.zeros(nl), 1, -1])) for state in range(ns)])

  ##################
  # Constraint 15b #
  ##################
  num_iq_b = ns * nq_notarget
  # Left-hand side
  L_SkS = diag(b_poly_Sk, axis=0)
  L_SkSQQ = sparse.tensordot(L_SkS, sparse.eye(nq_notarget), axes=-1)
  L_b_QS_QSk = L_SkSQQ.transpose([3, 0, 4, 2, 1]).reshape([nq_notarget*ns, nq_notarget*ns*nk])     # <--
  # Right-hand side
  sum_a = sparse.COO([range(na)], np.ones(na))
  R_aSsQq = sparse.tensordot(sum_a, sparse.tensordot(sparse.eye(ns), sparse.eye(nq_notarget), axes=-1), axes=-1)
  R_b_QS_asq = R_aSsQq.transpose([3, 1, 0, 2, 4]).reshape([ns * nq_notarget, na * ns * nq_notarget])

  A_iq_b = sp.bmat([[sp.coo_matrix((num_iq_b, num_varP)),      # P 
                     R_b_QS_asq.to_scipy_sparse(),             # X
                     L_b_QS_QSk.to_scipy_sparse(),             # V1
                     sp.coo_matrix((num_iq_b, num_var2)) ]])   # V2
  # TODO: indexing needs fix to have q_target not being the last one
  b_iq_b = np.zeros(num_iq_b)
  b_iq_b[np.ravel_multi_index((s0, q0), (ns, nq_notarget))] = 1.
  c = Constraint(A_iq=A_iq_b, b_iq=b_iq_b)

  ##################
  # Constraint 15c #
  ##################
  num_eq_c = ns * nl * nq_notarget
  # Left-hand side
  L_SklS = diag(A_poly_Skl, axis=0)
  L_SklSQQ = sparse.tensordot(L_SklS, sparse.eye(nq_notarget), axes=-1)
  L_c_SlQ_QSk = L_SklSQQ.transpose([0,2,4,5,3,1]).reshape([ns * nl * nq_notarget, nq_notarget * ns * nk])
  # Right-hand side
  R_SlQasq_c = sparse.tensordot(P_asS, P_lqQ_notarget, axes=-1).transpose([2, 3, 5, 0, 1, 4])
  R_c_SlQ_asq = R_SlQasq_c.reshape([ns * nl * nq_notarget, na * ns * nq_notarget])

  A_eq_c = sp.bmat([[sp.coo_matrix((num_eq_c, num_varP)),     # P 
                     R_c_SlQ_asq.to_scipy_sparse(),           # X
                     L_c_SlQ_QSk.to_scipy_sparse(),           # V1
                     sp.coo_matrix((num_eq_c, num_var2))]])   # V2
  c &= Constraint(A_eq=A_eq_c, b_eq=np.zeros(num_eq_c))

  ##################
  # Constraint 15d #
  ##################
  num_iq_d = 1
  L_d_sk = b_poly_Sk.reshape((1, ns * nk))
  A_iq_d = sp.bmat([[1,                                      # P
                     sp.coo_matrix((num_iq_d, num_varX)),    # X
                     sp.coo_matrix((num_iq_d, num_var1)),    # V1
                     L_d_sk.to_scipy_sparse()]])             # V2
  c &= Constraint(A_iq=A_iq_d, b_iq=np.zeros(num_iq_d))

  ##################
  # Constraint 15e #
  ##################
  num_eq_e = nl * ns;
  # Left-hand side
  L_e_Sl_sk = diag(A_poly_Skl, axis=0).transpose([3, 2, 0, 1]).reshape([ns * nl, ns * nk]) 
  # Right-hand side
  R_e_asSlqQ = sparse.tensordot(P_asS, P_lqQ_sep, axes=-1)
  R_e_Slasq = R_e_asSlqQ.sum(axis=[5]).transpose([2,3,0,1,4])
  R_e_Sl_asq_e = R_e_Slasq.reshape([ns * nl, na * ns * nq_notarget])

  A_eq_e = sp.bmat([[sp.coo_matrix((num_eq_e, num_varP)),    # P
                     R_e_Sl_asq_e.to_scipy_sparse(),         # X
                     sp.coo_matrix((num_eq_e, num_var1)),    # V1
                     L_e_Sl_sk.to_scipy_sparse()]])          # V2
  c &= Constraint(A_eq=A_eq_e, b_eq=np.zeros(num_eq_e))

  ##################
  #### Solve it ####
  ##################

  objective = np.zeros(num_varP + num_varX + num_var1 + num_var2)
  objective[0] = -1  # maximize P

  sol = solve_ilp(objective, c, J_int=[])

  ########################################################

  if sol['status'] == 'optimal':
    return sol['x'][0], sol['x'][num_varP: num_varP+num_varX].reshape((na, ns, nq_notarget))
  else:
    print("solver returned {}".format(sol['status']))
    return -1, -1