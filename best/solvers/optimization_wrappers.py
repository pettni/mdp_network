import sys
import scipy.sparse as sp
import numpy as np
from dataclasses import dataclass

default_solver = 'gurobi'

# Try to import gurobi
try:
    from gurobipy import *
    TIME_LIMIT = 10 * 3600

except Exception as e:
    print("warning: gurobi not found")
    default_solver = 'mosek'

# Try to import mosek/cvxopt
try:
    import mosek

except Exception as e:
    print("warning: mosek not found")
    default_solver = 'gurobi'

return_codes = {1: 'unknown', 2: 'optimal', 3: 'infeasible', 5: 'dual infeasible'}

#Class for constraints----------------------------------------------------------

@dataclass
class Constraint(object):
    A_eq: sp.coo_matrix = None
    b_eq: np.array = None
    A_iq: sp.coo_matrix = None
    b_iq: np.array = None

    @property
    def has_iq(self):
        return self.A_iq is not None

    @property
    def has_eq(self):
        return self.A_eq is not None

    def __and__(self, other):
        A_eq = None
        b_eq = None

        if self.has_eq or other.has_eq:
            A_eq = sp.bmat([[c.A_eq] for c in [self, other] if c.A_eq is not None])
            b_eq = np.hstack([c.b_eq for c in [self, other] if c.b_eq is not None])

        A_iq = None
        b_iq = None

        if self.has_iq or other.has_iq:
            A_iq = sp.bmat([[c.A_iq] for c in [self, other] if c.A_iq is not None])
            b_iq = np.hstack([c.b_iq for c in [self, other] if c.b_iq is not None])

        return Constraint(A_eq=A_eq, b_eq=b_eq, A_iq=A_iq, b_iq=b_iq)

    def __iand__(self, other):
        if self.has_eq and other.has_eq:
            self.A_eq = sp.bmat([[self.A_eq], [other.A_eq]])
            self.b_eq = np.hstack([self.b_eq, other.b_eq])
        elif other.has_eq:
            self.A_eq = other.A_eq
            self.b_eq = other.b_eq
        if self.has_iq and other.has_iq:
            self.A_iq = sp.bmat([[self.A_iq], [other.A_iq]])
            self.b_iq = np.hstack([self.b_iq, other.b_iq])
        elif other.has_iq:
            self.A_iq = other.A_iq
            self.b_iq = other.b_iq

        return self

def solve_ilp(c, constraint, J_int=None, J_bin=None, solver=default_solver, output=0):
    """
    Solve the ILP
        min c' x
        s.t. Aiq x <= biq
             Aeq x == beq
             x[J_int] are integers in N
             x[J_bin] are binary {0, 1}
             x >= 0
    using the solver `solver`.
    If `J_int` and `J_bin` are not given, all variables are treated as integers.

    Returns a dict sol with the fields
      'status': solver status
      'rcode': return code (2: optimal, 3: infeasible, 5: dual infeasible, 1: unknown)
      'x': the primary solution
    """
    if solver is None:
        solver = default_solver

    if J_int is None and J_bin is None:
        J_int = range(Aiq.shape[1])
        J_bin = []
    elif J_int is None:
        J_int = []
    elif J_bin is None:
        J_bin = []

    if len(set(J_bin) & set(J_int)):
        raise Exception('J_int and J_bin overlap')

    if constraint.has_eq and not constraint.has_iq:
        constraint.A_iq = sp.coo_matrix((0, constraint.A_eq.shape[1]))
        constraint.b_iq = np.array([])

    if constraint.has_iq and not constraint.has_eq:
        constraint.A_eq = sp.coo_matrix((0, constraint.A_iq.shape[1]))
        constraint.b_eq = np.array([])

    if solver == 'gurobi':
        sol = _solve_gurobi(c, constraint.A_iq, constraint.b_iq,
                             constraint.A_eq, constraint.b_eq,
                             J_int, J_bin, output)
    elif solver == 'mosek':
        sol = _solve_mosek(c, constraint.A_iq, constraint.b_iq,
                            constraint.A_eq, constraint.b_eq,
                            J_int, J_bin, output)

    sol['status'] = return_codes[sol['rcode']]
    return sol

def _solve_mosek(c, Aiq, biq, Aeq, beq, J_int, J_bin, output):
    """
        Solve optimization problem
        min c' x
        s.t. Aiq x <= biq
             Aeq x == beq
             x[J] are integers
             x >= 0
        using the Mosek ILP solver
    """
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    inf = 0.0  # for readability

    num_var = Aiq.shape[1]
    num_iq = Aiq.shape[0]
    num_eq = Aeq.shape[0]

    env = mosek.Env()
    env.set_Stream(mosek.streamtype.log, streamprinter)

    task = env.Task(0, 0)
    task.set_Stream(mosek.streamtype.log, streamprinter)
    task.putintparam(mosek.iparam.log, 10 * output)

    task.appendvars(num_var)
    task.appendcons(num_iq + num_eq)

    # Coefficients
    task.putcslice(0, num_var, c)

    # Positivity
    task.putvarboundslice(0, num_var, [mosek.boundkey.lo] * num_var,
                                      [0.] * num_var,
                                      [+inf] * num_var)

    # Constrain binary to [0, 1]
    task.putvarboundlist(J_bin, [mosek.boundkey.ra] * len(J_bin),
                         [0.] * len(J_bin),
                         [1.] * len(J_bin))

    # Integers
    task.putvartypelist(J_int+J_bin, [mosek.variabletype.type_int] * len(J_int+J_bin))

    # Inequality constraints
    task.putaijlist(Aiq.row, Aiq.col, Aiq.data)
    task.putconboundslice(0, num_iq,
                          [mosek.boundkey.up] * num_iq,
                          [-inf] * num_iq,
                          biq)

    # Equality constraints
    task.putaijlist(num_iq + Aeq.row, Aeq.col, Aeq.data)
    task.putconboundslice(num_iq, num_iq + num_eq,
                          [mosek.boundkey.fx] * num_eq,
                          beq,
                          beq)

    task.putobjsense(mosek.objsense.minimize)
    task.optimize()

    sol = {}
    sol['x'] = np.zeros(num_var, float)

    if len(J_int+J_bin) > 0:
        solsta = task.getsolsta(mosek.soltype.itg)
        task.getxx(mosek.soltype.itg, sol['x'])
    else:
        solsta = task.getsolsta(mosek.soltype.bas)
        task.getxx(mosek.soltype.bas, sol['x'])

    if solsta in [solsta.optimal,
                  solsta.near_optimal,
                  solsta.integer_optimal,
                  solsta.near_integer_optimal]:
        sol['rcode'] = 2
    elif solsta in [solsta.dual_infeas_cer,
                    solsta.near_dual_infeas_cer]:
        sol['rcode'] = 5
    elif solsta in [solsta.prim_infeas_cer,
                    solsta.near_prim_infeas_cer]:
        sol['rcode'] = 3
    elif solsta == solsta.unknown:
        sol['rcode'] = 1

    return sol


def solCallback(model, where):
    if where == GRB.callback.MIPSOL:
        solcnt = model.cbGet(GRB.callback.MIPSOL_SOLCNT)
        runtime = model.cbGet(GRB.callback.RUNTIME)
        if solcnt > 0 and runtime > TIME_LIMIT:
            model.terminate()


def _solve_gurobi(c, Aiq, biq, Aeq, beq, J_int, J_bin, output):
    """
        Solve optimization problem
        min c' x
        s.t. Aiq x <= biq
             Aeq x == beq
             x[J] are integers
             x >= 0
        using the Gurobi solver
    """
    num_var = Aiq.shape[1]

    Aiq = Aiq.tocsr()
    Aeq = Aeq.tocsr()

    m = Model()

    # Enable/disable output
    m.setParam(GRB.Param.OutputFlag, output)

    # Some solver parameters, see
    # http://www.gurobi.com/documentation/6.0/refman/mip_models.html
    m.setParam(GRB.Param.TimeLimit, TIME_LIMIT)
    m.setParam(GRB.Param.MIPFocus, 1)

    x = []
    for i in range(num_var):
        if i in J_int:
            x.append(m.addVar(vtype=GRB.INTEGER, obj=c[i]))
        elif i in J_bin:
            x.append(m.addVar(vtype=GRB.BINARY, obj=c[i]))
        else:
            x.append(m.addVar(obj=c[i]))
    m.update()

    for i in range(Aiq.shape[0]):
        start = Aiq.indptr[i]
        end = Aiq.indptr[i + 1]
        variables = [x[j] for j in Aiq.indices[start:end]]
        coeff = Aiq.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        m.addConstr(lhs=expr, sense=gurobipy.GRB.LESS_EQUAL, rhs=biq[i])

    for i in range(Aeq.shape[0]):
        start = Aeq.indptr[i]
        end = Aeq.indptr[i + 1]
        variables = [x[j] for j in Aeq.indices[start:end]]
        coeff = Aeq.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        m.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=beq[i])

    m.update()
    m.optimize(solCallback)

    sol = {}
    if m.status == gurobipy.GRB.status.OPTIMAL:
        sol['x'] = np.array([var.x for var in x])
        sol['primal objective'] = m.objVal
    if m.status in [2,3,5]:
        sol['rcode'] = m.status
    else:
        sol['rcode'] = 1

    return sol
