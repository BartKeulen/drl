"""
Original by Christian Balcom

https://github.com/computer-whisperer/integrated-dynamics
"""

from numpy import *
from .boxQP import boxQP
from drl.utilities import function_derivatives, func_serializer
import logging
logger = logging.getLogger("iLQG")


def ilqg(dynamics_fun_in, cost_fun_in, x0, u0, options_in=None):
    """
    PORTED FROM MATLAB CODE
    http://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization


    iLQG - solve the deterministic finite-horizon optimal control problem.
               minimize sum_i CST(x(:,i),u(:,i)) + CST(x(:,end))
               u
           s.t.  x(:,i+1) = DYN(x(:,i),u(:,i))

    Inputs
    ======
    DYNCST - A combined dynamics and cost function. It is called in
    three different formats.
     1) step:
      [xnew,c] = DYNCST(x,u,i) is called during the forward pass.
      Here the state x and control u are vectors: size(x)==[n 1],
      size(u)==[m 1]. The cost c and time index i are scalars.
      If Op.parallel==true (the default) then DYNCST(x,u,i) is be
      assumed to accept vectorized inputs: size(x,2)==size(u,2)==K

     2) final:
      [~,cnew] = DYNCST(x,nan) is called at the end the forward pass to compute
      the final cost. The nans indicate that no controls are applied.

     3) derivatives:
      [~,~,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = DYNCST(x,u,I) computes the
      derivatives along a trajectory. In this case size(x)==[n N+1] where N
      is the trajectory length. size(u)==[m N+1] with NaNs in the last column
      to indicate final-cost. The time indexes are I=(1:N).
      Dimensions match the variable names e.g. size(fxu)==[n n m N+1]
      note that the last temporal element N+1 is ignored for all tensors
      except cx and cxx, the final-cost derivatives.

    x0 - The initial state from which to solve the control problem.
      Should be a column vector. If a pre-rolled trajectory is available
      then size(x0)==[n N+1] can be provided and Op.cost set accordingly.
    u0 - The initial control sequence. A matrix of size(u0)==[m N]
      where m is the dimension of the control and N is the number of state
      transitions.
    Op - optional parameters, see below

    Outputs
    =======
    x - the optimal state trajectory found by the algorithm.
        size(x)==[n N+1]
    u - the optimal open-loop control sequence.
        size(u)==[m N]
    L - the optimal closed loop control gains. These gains multiply the
        deviation of a simulated trajectory from the nominal trajectory x.
        size(L)==[m n N]
    Vx - the gradient of the cost-to-go. size(Vx)==[n N+1]
    Vxx - the Hessian of the cost-to-go. size(Vxx)==[n n N+1]
    cost - the costs along the trajectory. size(cost)==[1 N+1]
           the cost-to-go is V = fliplr(cumsum(fliplr(cost)))
    lambda - the final value of the regularization parameter
    trace - a trace of various convergence-related values. One row for each
            iteration, the columns of trace are
            [iter lambda alpha g_norm dcost z sum(cost) dlambda]
            see below foe details.
    """

    # user-adjustable parameters
    options = {
        'lims':           None,  # control limits
        'parallel':       True,  # use parallel line-search?
        'Alpha':          10**linspace(0, -3, 8),  # backtracking coefficients
        'tolFun':         1e-7,  # reduction exit criterion
        'tolGrad':        1e-5,  # gradient exit criterion
        'maxIter':        500,  # maximum iterations
        'lambdaInit':     1,  # initial value for lambda
        'dlambdaInit':    1,  # initial value for dlambda
        'lambdaFactor':   1.6,  # lambda scaling factor
        'lambdaMax':      1e10,  # lambda maximum value
        'lambdaMin':      1e-6,  # below this value lambda = 0
        'regType':        1,  # regularization type 1: q_uu+lambda*eye(); 2: V_xx+lambda*eye()
        'zMin':           0,  # minimal accepted reduction ratio
        'plot':           1,  # 0: no;  k>0: every k iters; k<0: every k iters, with derivs window
        'print':          2,  # 0: no;  1: final; 2: iter; 3: iter, detailed
        'cost':           None,  # initial cost for pre-rolled trajectory
        'dyn_first_der':  True,  # Use analytical first derivative of dynamics
        'dyn_sec_der':    False,  # Use analytical second derivative of dynamics
        'cost_first_der': True,  # Use analytical first derivative of cost
        'cost_sec_der':   True  # Use analytical second derivative of cost
    }

    # serialize dynamics and cost function calls
    dynamics_fun = lambda x, u: func_serializer(x, u, dynamics_fun_in, first=options["dyn_first_der"], second=options["dyn_sec_der"])
    cost_fun = lambda x, u: func_serializer(x, u, cost_fun_in, first=options["cost_first_der"], second=options["cost_sec_der"])

    # --- initial sizes and controls
    n = x0.shape[-1]          # dimension of state vector
    m = u0.shape[1]          # dimension of control vector
    N = u0.shape[0]         # number of state transitions
    u = u0[:]

    fx = fu = fxx = fuu = fxu = NaN
    cx = cu = cxx = cuu = cxu = NaN

    # -- process options
    if options_in is not None:
        options.update(options_in)

    lamb = options["lambdaInit"]
    dlamb = options["dlambdaInit"]

    # Initial trajectory
    if x0.shape[0] == n:
        diverge = True
        for alpha in options["Alpha"]:
            xn, un, costn, fxn, fun, fxxn, fuun, fxun, cxn, cun, cxxn, cuun, cxun = forward_pass(dynamics_fun, cost_fun, x0, alpha*u, None, None, None, array([1]), options["lims"])
            # simplistic divergence random_scripts
            if (abs(xn) < 1e8).all():
                u = un[:, 0]
                x = xn[:, 0]
                cost = costn[:, 0]
                if fxn is not None:
                    fx = fxn[:, 0]
                    fu = fun[:, 0]
                    fxx = fxxn[:, 0]
                    fuu = fuun[:, 0]
                    fxu = fxun[:, 0]
                if cxn is not None:
                    cx = cxn[:, 0]
                    cu = cun[:, 0]
                    cxx = cxxn[:, 0]
                    cuu = cuun[:, 0]
                    cxu = cxun[:, 0]
            else:
                logger.info("\nEXIT: Initial control sequence caused divergence\n")
                return xn[:, 0], un[:, 0], None, None, None, costn[:, 0]

    elif x0.shape[0] == N+1: # already did initial fpass
        x = x0
        if options["cost"] is None:
            raise ValueError("pre-rolled initial trajectory requires cost")
        else:
            cost = options["cost"]

    flgChange = 1
    dcost = 0
    z = 0
    expected = 0
    L = zeros((N, n, m))

    logger.info("\n============== begin iLQG ===============\n")

    for alg_iter in range(options["maxIter"]):

        # ==== STEP 1: differentiate dynamics along new trajectory
        if flgChange:
            if isnan(fx).any() or isnan(fxx).any():
                fx, fu, fxx, fxu, fuu = function_derivatives(x, vstack((u, full([1, m], nan))), dynamics_fun, first=(fx, fu), use_second=options["dyn_sec_der"])
            if isnan(cx).any() or isnan(cxx).any():
                cx, cu, cxx, cxu, cuu = function_derivatives(x, vstack((u, full([1, m], nan))), cost_fun, first=(cx, cu), use_second=True)
            flgChange = 0

        # ==== STEP 2: backward pass, compute optimal control law and cost-to-go
        backPassDone = 0
        while not backPassDone:
            diverge, Vx, Vxx, l, L, dV = back_pass(cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu, lamb, options["regType"], options["lims"], u)

            if diverge:
                logger.debug("Cholesky failed at timestep {}.".format(diverge))
                dlamb = max(dlamb * options["lambdaFactor"], options["lambdaFactor"])
                lamb = max(lamb * dlamb, options["lambdaMin"])
                if lamb > options["lambdaMax"]:
                    break
                continue
            backPassDone = 1

        #Check for termination due to small gradient
        g_norm = mean((abs(l) / (abs(u)+1)).max(1))
        if g_norm < options["tolGrad"] and lamb < 1e-5:
            dlamb = min(dlamb / options["lambdaFactor"], 1/options["lambdaFactor"])
            lamb = lamb * dlamb * (lamb > options["lambdaMin"])
            logger.info("SUCCESS: gradient norm < tolGrad")
            break

        # ==== STEP 3: line-search to find new control sequence, trajectory, cost
        fwdPassDone = 0
        if backPassDone:
            if options["parallel"]: # parallel line-search
                xnew, unew, costnew, *_ = forward_pass(dynamics_fun, cost_fun, x0, u, L, x[:N], l, options["Alpha"], options["lims"])
                dcost = cost.sum(axis=0) - costnew.sum(axis=0)
                w = argmax(dcost)
                dcost = dcost[w]
                alpha = options["Alpha"][w]
                expected = -alpha*(dV[0] + alpha*dV[1])
                if expected > 0:
                    z = dcost/expected
                else:
                    z = sign(dcost)
                    logger.info("WARNING: non-positive expected reduction: should not occur")
                if z > options["zMin"]:
                    fwdPassDone = 1
                    costnew = costnew[:,w]
                    xnew = xnew[:,w]
                    unew = unew[:,w]

            else: # serial backtracking line-search
                for alpha in options["Alpha"]:
                    xnew, unew, costnew, *_ = forward_pass(dynamics_fun, cost_fun, x0, u+l*alpha, L, x[:N], None, array([]), options["lims"])
                    dcost = cost.sum(axis=0) - costnew.sum(axis=0)
                    expected = -alpha*(dV[0] + alpha*dV[1])
                if expected > 0:
                    z = dcost/expected
                else:
                    z = sign(dcost)
                    logger.info("WARNING: non-positive expected reduction: should not occur")
                if z > options["zMin"]:
                    fwdPassDone = 1
                    break

        # ==== STEP 4: accept (or not)
        if fwdPassDone:
            # print status
            logger.info('iter: {} cost: {} reduction: {} gradient: {} log10lam: {}'.format(alg_iter, sum(cost.flatten(1)), dcost, g_norm, nan if lamb == 0 else log10(lamb)))

            # decrease lambda
            dlamb = min(dlamb / options["lambdaFactor"], 1/options["lambdaFactor"])
            lamb = lamb * dlamb * (lamb > options["lambdaMin"])

            # accept changes
            u = unew
            x = xnew
            cost = costnew
            flgChange = 1

            # terminate ?
            if dcost < options["tolFun"]:
                logger.info("\nSUCCESS: cost change < tolFun")
                break

        else: # No cost improvement

            # increase lambda
            dlamb = max(dlamb * options["lambdaFactor"], options["lambdaFactor"])
            lamb = max(lamb * dlamb, options["lambdaMin"])

            # print status
            logger.info('iter: {} REJECTED expected: {} actual: {} log10lam: {}'.format(alg_iter, expected, dcost, log10(dlamb)))

            # terminate ?
            if lamb > options["lambdaMax"]:
                logging.info("\nEXIT: lambda > lambdaMax")
                break
    else:
        logger.warn("\nEXIT: Maximum iterations reached.\n")

    return x, u, L, Vx, Vxx, cost


def forward_pass(dynamics_fun, cost_fun, x0, u, L, x, du, alpha, lims):
    n = x0.shape[0]
    K = alpha.shape[0]
    N = u.shape[0]
    m = u.shape[1]

    xnew = zeros((N+1, K, n))
    xnew[0, :, :] = x0
    unew = zeros((N, K, m))
    cnew = zeros((N+1, K))

    fx = zeros((N, K, n, n))
    fu = zeros((N, K, m, n))
    fxx = zeros((N, K, n, n, n))
    fuu = zeros((N, K, m, m, n))
    fxu = zeros((N, K, m, n, n))

    cx = zeros((N+1, K, n))
    cu = zeros((N+1, K, m))
    cxx = zeros((N+1, K, n, n))
    cuu = zeros((N+1, K, m, m))
    cxu = zeros((N+1, K, n, m))

    for i in range(N):
        unew[i] = u[i]

        if du is not None:
            unew[i] = unew[i] + dot(alpha[:, None], du[None, i, :])

        if L is not None:
            dx = xnew[i] - x[i]
            unew[i] = unew[i] + dot(dx, L[i].T)

        if lims is not None:
            unew[i] = clip(unew[i], lims[:, 0], lims[:, 1])

        xnew[i+1], fx[i], fu[i], fxx[i], fuu[i], fxu[i] = dynamics_fun(xnew[i], unew[i])
        cnew[i], cx[i], cu[i], cxx[i], cuu[i], cxu[i] = cost_fun(xnew[i], unew[i])

    cnew[N], cx[N], cu[N], cxx[N], cuu[N], cxu[N] = cost_fun(xnew[N], full([K, m], nan))

    return xnew, unew, cnew, fx, fu, fxx, fuu, fxu, cx, cu, cxx, cuu, cxu


def back_pass(cx, cu, cxx, cxu, cuu, fx, fu, fxx, fxu, fuu, lamb, regType, lims, u):
    """
    Perform the Ricatti-Mayne backward pass
    """

    # tensor = lambda a, b: sum(a*b, 0).T

    N = cx.shape[0]
    n = cx.shape[1]
    m = cu.shape[1]

    k = zeros((N-1, m))
    K = zeros((N-1, m, n))
    Vx = zeros((N, n))
    Vxx = zeros((N, n, n))
    dV = array([0, 0])

    Vx[N-1] = cx[N-1]
    Vxx[N-1] = cxx[N-1]

    diverge = 0
    for i in reversed(range(N-1)):
        Qu = cu[i] + dot(fu[i], Vx[i+1])
        Qx = cx[i] + dot(fx[i], Vx[i+1])

        Quu = cuu[i].T + dot(dot(fu[i], Vxx[i+1]), fu[i].T)
        if not isnan(fuu).any():
            fuuVx = sum(Vx[i+1, :, None, None]*fuu[i].T, 0)
            Quu = Quu + fuuVx

        Qux = cxu[i].T + dot(dot(fu[i], Vxx[i+1]), fx[i].T)
        if not isnan(fxu).any():
            fxuVx = sum(Vx[i+1, :, None, None]*fxu[i].T, 0)
            Qux = Qux + fxuVx

        Qxx = cxx[i].T + dot(dot(fx[i], Vxx[i+1]), fx[i].T)
        if not isnan(fxx).any():
            fxxVx = sum(Vx[i+1, :, None, None]*fxx[i].T, 0)
            Qxx = Qxx + fxxVx

        Vxx_reg = (Vxx[i+1] + lamb*eye(n)*(regType==2))

        Qux_reg = cxu[i].T + dot(dot(fu[i], Vxx_reg), fx[i].T)
        if not isnan(fxu).any():
            Qux_reg = Qux_reg + fxuVx

        QuuF = cuu[i] + dot(dot(fu[i], Vxx_reg), fu[i].T) + lamb*eye(m)*(regType == 1)

        if not isnan(fuu).any():
            QuuF = QuuF + fuuVx

        if lims is None or lims[0,0] > lims[0, 1]:
            # no control limits: Cholesky decomposition, check for non-PD
            try:
                R = linalg.cholesky(QuuF).T
            except linalg.LinAlgError as e:
                # logger.warning(e)
                diverge = i
                return diverge, Vx, Vxx, k, K, dV

            # find control law
            kK = linalg.solve(-R, linalg.solve(R.T, concatenate((Qu[:, None], Qux_reg), axis=1)))
            k_i = kK[:,0]
            K_i = kK[:,1:n+1]

        else:   # Solve Quadratic Program
            lower = lims[:,0]-u[i, :]
            upper = lims[:,1]-u[i, :]

            try:
                k_i, _, R, free = boxQP(QuuF, Qu, lower, upper, k[min((i+1, N-2))])
            except linalg.LinAlgError as e:
                # logger.warning(e)
                diverge = i
                return diverge, Vx, Vxx, k, K, dV

            K_i = zeros((m, n))
            if free.any():
                K_i[free,:] = linalg.solve(-R, linalg.solve(R.T, Qux_reg[free,:]))

        # update cost-to-go approximation
        v1 = dot(k_i.T, Qu)
        v2 = dot(dot(.5*k_i.T, Quu), k_i)
        dV = dV + [v1, v2]
        Vx[i]  = Qx  + dot(dot(K_i.T, Quu), k_i) + dot(K_i.T, Qu) + dot(Qux.T, k_i)
        Vxx[i] = Qxx + dot(dot(K_i.T, Quu), K_i) + dot(K_i.T, Qux) + dot(Qux.T, K_i)
        Vxx[i] = .5*(Vxx[i] + Vxx[i].T)

        # save controls/gains
        k[i] = k_i
        K[i] = K_i

    return diverge, Vx, Vxx, k, K, dV