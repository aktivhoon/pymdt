import numpy as np
from scipy.optimize import minimize
import math

def compute_bound_class(x0, LB, UB):
    """
    For each parameter, determine its bound type:
    0: Unconstrained
    1: Lower bound only (finite LB, UB=inf)
    2: Upper bound only (LB=-inf, finite UB)
    3: Both bounds finite (LB and UB)
    4: Fixed variable (LB==UB)
    """
    n = len(x0)
    bound_class = np.zeros(n, dtype=int)
    for i in range(n):
        lb = LB[i] if LB is not None else -np.inf
        ub = UB[i] if UB is not None else np.inf
        if np.isfinite(lb) and np.isfinite(ub):
            if lb == ub:
                bound_class[i] = 4
            else:
                bound_class[i] = 3
        elif np.isfinite(lb) and not np.isfinite(ub):
            bound_class[i] = 1
        elif not np.isfinite(lb) and np.isfinite(ub):
            bound_class[i] = 2
        else:
            bound_class[i] = 0
    return bound_class

def transform(u, lb, ub, bclass):
    """
    Transforms an unconstrained variable u into x (in original space) 
    depending on the bound class.
    """
    if bclass == 0:  # unconstrained
        return u
    elif bclass == 1:  # lower bound only: x = lb + u^2
        return lb + u**2
    elif bclass == 2:  # upper bound only: x = ub - u^2
        return ub - u**2
    elif bclass == 3:  # both bounds finite
        # MATLAB: x = lb + ((sin(u - 2*pi)+1)/2)*(ub - lb)
        # return lb + ((np.sin(u - 2*np.pi) + 1) / 2) * (ub - lb)
        transformed_value = lb + ((np.sin(u) + 1) / 2) * (ub - lb)
        return max(lb, min(ub, transformed_value))
    else:  # fixed variable: return the fixed value (lb==ub)
        return lb

def inv_transform(x, lb, ub, bclass):
    """
    Computes the inverse transformation: given x in original space,
    returns u in the unconstrained space.
    """
    if bclass == 0:
        return x
    elif bclass == 1:
        # Add handling for infeasible starting values
        if x <= lb:
            # Infeasible starting value. Use bound.
            return 0
        else:
            return np.sqrt(x - lb)
    elif bclass == 2:
        # Add handling for infeasible starting values
        if x >= ub:
            # Infeasible starting value. Use bound.
            return 0
        else:
            return np.sqrt(ub - x)
    elif bclass == 3:
        if x <= lb:
            return -np.pi/2
        elif x >= ub:
            return np.pi/2
        else:
            # Compute y = 2*(x-lb)/(ub-lb)-1, ensure it is in [-1,1]
            y = 2*(x - lb)/(ub - lb) - 1
            y = max(-1, min(1, y))
            return 2*np.pi + np.arcsin(y)
    else:
        # fixed variable; not used in the optimization
        return None

def fminsearchbnd(fun, x0, LB=None, UB=None, options=None, *args):
    """
    A Python implementation of MATLAB's fminsearchbnd.
    
    Parameters:
      fun: objective function; expects a 1D numpy array x and any additional args.
      x0: initial guess (1D numpy array)
      LB: lower bound (1D array-like, same size as x0); use -np.inf if no lower bound.
      UB: upper bound (1D array-like, same size as x0); use np.inf if no upper bound.
      options: dictionary of options passed to scipy.optimize.minimize.
      *args: additional arguments passed to fun.
      
    Returns:
      x_opt: the minimizer (in original space)
      fval: objective function value at x_opt
      exitflag: optimization exit flag (integer status code)
      output: full OptimizeResult object from scipy.optimize.minimize.
    """
    x0 = np.asarray(x0).flatten()
    n = len(x0)
    if LB is None:
        LB = -np.inf * np.ones(n)
    else:
        LB = np.asarray(LB).flatten()
    if UB is None:
        UB = np.inf * np.ones(n)
    else:
        UB = np.asarray(UB).flatten()
    
    # Determine bound class for each parameter.
    bound_class = compute_bound_class(x0, LB, UB)
    
    # Identify free and fixed variables.
    free_idx = [i for i in range(n) if bound_class[i] != 4]
    fixed_idx = [i for i in range(n) if bound_class[i] == 4]
    
    # Create the initial guess for the free parameters in the unconstrained space.
    u0 = []
    for i in free_idx:
        u0.append(inv_transform(x0[i], LB[i], UB[i], bound_class[i]))
    u0 = np.array(u0)
    
    # Define a helper function to map free variables u to full x.
    def u_to_x(u):
        x = np.array(x0, copy=True)
        for j, i in enumerate(free_idx):
            x[i] = transform(u[j], LB[i], UB[i], bound_class[i])
        # Fixed indices remain unchanged.
        return x
    
    # Define the objective in terms of u.
    def obj(u):
        x = u_to_x(u)
        return fun(x, *args)
    
    # Call scipy's Nelder-Mead on the unconstrained parameters.
    res = minimize(obj, u0, method='Nelder-Mead', options=options)
    
    # Map optimal u back to the original parameter space.
    x_opt = u_to_x(res.x)
    fval = res.fun
    exitflag = res.status  # status code from minimize
    
    return x_opt, fval, exitflag, res