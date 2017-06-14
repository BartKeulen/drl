"""
Original by Christian Balcom

https://github.com/computer-whisperer/integrated-dynamics
"""
from numpy import *
import logging
logger = logging.getLogger("boxQP")

#function [x,result,Hfree,free,trace] = boxQP(H,g,lower,upper,x0,options)
# Minimize 0.5*x'*H*x + x'*g  s.t. lower<=x<=upper
#
#  inputs:
#     H            - positive definite matrix   (n * n)
#     g            - bias vector                (n)
#     lower        - lower bounds               (n)
#     upper        - upper bounds               (n)
#
#   optional inputs:
#     x0           - initial state              (n)
#     options      - see below                  (7)
#
#  outputs:
#     x            - solution                   (n)
#     result       - result type (roughly, higher is better, see below)
#     Hfree        - subspace cholesky factor   (n_free * n_free)
#     free         - set of free dimensions     (n)

def boxQP(H, g, lower, upper, x0=None, options_in=None):
    
    n = H.shape[0]
    clamped = zeros(n)
    oldvalue = 0
    result = 0
    gnorm = 0
    nfactor = 0
    Hfree = zeros(n)

    # initial state
    if x0 is not None and x0.size == n:
        x = clip(x0, lower, upper)
    else:
        LU = [lower, upper]
        #LU[~isfinite(LU)] = nan
        x = nanmean(LU,1)
    x[isinf(x)] = 0
    
    # options
    options = {
        "maxIter": 100,     # maximum number of iterations
        "minGrad": 1e-8,     # minimum norm of non-fixed gradient
        "minRelImprove": 1e-8,     # minimum relative improvement
        "stepDec": 0.6,     # factor for decreasing stepsize
        "minStep": 1e-22,     # minimal stepsize for linesearch
        "Armijo": 0.1, 	# Armijo parameter (fraction of linear improvement required)
        "verbose":  0, # verbosity
    }

    if options_in is not None:
        options.update(options_in)
    
    # initial objective value
    value = dot(x.T, g) + 0.5*dot(dot(x.T, H),x)

    # main loop
    for iter in range(options["maxIter"]):
        
        # check relative improvement
        if iter>0 and ((oldvalue - value) < options["minRelImprove"]*abs(oldvalue)).any():
            #print('Improvement smaller than tolerance')
            break
        oldvalue = value
        
        # get gradient
        grad = g + dot(H, x)
        
        # find clamped dimensions
        old_clamped = clamped
        clamped = (x == lower) & (grad > 0) | (x == upper) & (grad < 0)
        free = ~clamped
        
        # check for all clamped
        
        if clamped.all():
            #print('All dimensions are clamped')
            break
        
        # factorize if clamped has changed
        if iter == 0:
            factorize = True
        else:
            factorize = (old_clamped != clamped).any()
        
        if factorize:
            val = H[free, :][:, free]
            try:
                Hfree = linalg.cholesky(val).T
            except Exception as e:
                raise e
            nfactor += 1
        
        # check gradient norm
        gnorm = linalg.norm(grad[free])
        if gnorm < options["minGrad"]:
            #print('Gradient norm smaller than tolerance')
            break
        
        # get search direction
        grad_clamped = g + dot(H, (x*clamped))
        search = zeros(n)
        search[free] = linalg.solve(-Hfree, linalg.solve(Hfree.T, grad_clamped[free])) - x[free]
        
        # check for descent direction
        sdotg = sum(search*grad)
        if sdotg >= 0: # (should not happen)
            break
        
        # armijo linesearch
        step  = 1
        nstep = 0
        xc = clip(x+step*search, lower, upper)
        vc = dot(xc.T, g) + 0.5*dot(dot(xc.T, H), xc)
        while ((vc - oldvalue)/(step*sdotg) < options["Armijo"]).all():
            step = step*options["stepDec"]
            nstep += 1
            xc = clip(x+step*search, lower, upper)
            vc = dot(xc.T, g) + 0.5*dot(dot(xc.T, H), xc)
            if step<options["minStep"]:
                print('Maximum line-search iterations exceeded')
                break
        
        if options["verbose"] > 1:
            print('iter #-3d  value # -9.5g |g| #-9.3g  reduction #-9.3g  linesearch #g^#-2d  n_clamped #d\n', 
                iter, vc, gnorm, oldvalue-vc, options["stepDec"], nstep, sum(clamped))
        
        # accept candidate
        x     = xc
        value = vc
    
    if iter >= options["maxIter"]:
        print('Maximum main iterations exceeded')
    
    # results = { 'Hessian is not positive definite',          # result = -1
    #            'No descent direction found',                # result = 0    SHOULD NOT OCCUR
    #            ,          # result = 1
    #            ,   # result = 2
    #            'No bounds, returning Newton point',         # result = 3
    #                    # result = 4
    #            ,     # result = 5
    #            }                  # result = 6

    if options["verbose"] > 0:
        print('RESULT: {}\niterations {}  gradient {} final value {}  factorizations {}\n'.format(
            result, iter, gnorm, value, nfactor))

    return x, result, Hfree, free
