import numpy as np
from numpy import matrix as mat
from matplotlib import pyplot as plt
import random

n = 100
a1, b1, c1 = 1, 3, 2

h = np.linspace(0,1,n)
y = [np.exp(a1*i**2 + b1*i + c1) + random.gauss(0,4) for i in h]
y = mat(y)


def Func(abc, iput):
    a = abc[0,0]
    b = abc[1,0]
    c = abc[2,0]
    return np.exp(a*iput**2 + b*iput + c)

def Deriv(abc, iput, n):
    x1 = abc.copy()
    x2 = abc.copy()
    x1[n,0] -= 0.000001
    x2[n,0] += 0.000001
    p1 = Func(x1, iput)
    p2 = Func(x2, iput)
    d = (p2-p1)*1.0 / (0.000002)
    return d

J = mat(np.zeros((n, 3))) # function*variable
fx = mat(np.zeros((n,1)))
fx_tmp = mat(np.zeros((n,1)))
xk = mat([[11.8],[20.7],[1.5]])
lase_mse = 0
step = 0
u,v = 1,2
conve = 1000

while(conve):
    mse, mse_tmp = 0,0
    step += 1
    # compute total mse error use current xk
    for i in range(n):
        fx[i] = Func(xk, h[i]) - y[0,i] # error
        mse += fx[i,0]**2

        for j in range(3):
            J[i,j] = Deriv(xk, h[i], j) # funtion*variable
    mse /= n

    H = J.T*J + u*np.eye(3) # u is lambda
    dx = -H.I * J.T * fx # H*delta(x) = g     g is J.T
    xk_tmp = xk.copy()
    xk_tmp += dx

    # compute total mse error use updated xk_tmp
    for i in range(n):
        fx_tmp[i] = Func(xk_tmp, h[i]) - y[0,i]
        mse_tmp += fx_tmp[i,0]**2
    mse_tmp /= n

    # compute rho
    q = (mse-mse_tmp) / ((dx.T*(u*dx - J.T*fx))[0,0])

    # rho > 0
    if q > 0:
        s = 1.0/3.0
        v = 2
        mse = mse_tmp
        xk = xk_tmp
        temp = 1 - pow(2*q-1, 3)

        if s > temp:
            u = u*s
        else:
            u = u*temp
    else:
        u = u*v
        v = v*2

    print("step = %d, abs(mse) = %.8f" % (step, abs(mse)))
    if abs(mse) < 0.000001:
        break

    lase_mse = mse
    conve -= 1

    z = [Func(xk, i) for i in h]

    print(xk)

plt.figure()
y = np.array(y)
plt.scatter(h, y[0], s=4)
plt.plot(h,z,'r')
plt.show()

