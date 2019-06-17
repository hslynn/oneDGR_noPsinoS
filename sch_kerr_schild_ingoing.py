"""
initial and gauge conditons for Schwarzschild spacetime in Kerr-Schild coordinate.
"""

from dolfin import *

def get_exact_var_list(func_space):
    g00_exp = Expression("-(1-2/x[0])", degree=10)
    g01_exp = Expression("2/x[0]", degree=10)
    g11_exp = Expression("1+2/x[0]", degree=10)
    
    Pi00_exp = Expression("-2/(x[0]+2)*sqrt(1+2/x[0])*2/x[0]/x[0]", degree=10)
    Pi01_exp = Expression("-2/(x[0]+2)*sqrt(1+2/x[0])*2/x[0]/x[0]", degree=10)
    Pi11_exp = Expression("-2/(x[0]+2)*sqrt(1+2/x[0])*2/x[0]/x[0]", degree=10)
    
    Phi00_exp = Expression("-2/pow(x[0],2)", degree=10)
    Phi01_exp = Expression("-2/pow(x[0],2)", degree=10)
    Phi11_exp = Expression("-2/pow(x[0],2)", degree=10)
    
    exp_list = (g00_exp, g01_exp, g11_exp,
                Pi00_exp, Pi01_exp, Pi11_exp,
                Phi00_exp, Phi01_exp, Phi11_exp)
    return tuple([project(exp, func_space) for exp in exp_list])
     
def get_H_list(func_space):
    H0_exp = Expression("2/x[0]/x[0]", degree=10)
    H1_exp = Expression("2*(1+x[0])/x[0]/x[0]", degree=10)
    
    exp_list = [H0_exp, H1_exp]
    return tuple([project(exp, func_space) for exp in exp_list])

def get_deriH_list(func_space):
    deriH00_exp = Expression("0", degree=10)
    deriH01_exp = Expression("0", degree=10)
    deriH10_exp = Expression("-4/pow(x[0], 3)", degree=10)
    deriH11_exp = Expression("-2*(x[0]+2)/pow(x[0], 3)", degree=10)
    
    exp_list = [deriH00_exp, deriH01_exp, deriH10_exp, deriH11_exp]
    return tuple([project(exp, func_space) for exp in exp_list])



