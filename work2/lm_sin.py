import numpy as np
from numpy import matrix as mat
from matplotlib import pyplot as plt
import random


def func(input_1, input_2):
    h = [0.,0.05128205,0.1025641,0.15384615,0.20512821,0.25641026,
     0.30769231,0.35897436,0.41025641,0.46153846,0.51282051,0.56410256,
     0.61538462,0.66666667,0.71794872,0.76923077,0.82051282,0.87179487,
     0.92307692,0.97435897,1.02564103,1.07692308,1.12820513,1.17948718,
     1.23076923,1.28205128,1.33333333,1.38461538,1.43589744,1.48717949,
     1.53846154,1.58974359,1.64102564,1.69230769,1.74358974,1.79487179,
     1.84615385,1.8974359,1.94871795,2.        ]

    n = len(h)

    y = [  50.14205309,74.52945195,-125.60406092,50.73876215,-78.06990572,
       69.07600961,72.59307969,-89.85886577,49.63661953,-116.99899809,
       75.61742457,54.64519908,-47.2762507,53.40895121,-143.71858796,
       72.98163061,26.37395524,-2.60377048,65.1767561,-148.47687307,
       64.41096274,-10.09721052,35.12977766,71.16446338,-138.68616457,
       55.83174324,-55.22601629,61.38557638,75.7080723,-109.84633432,
       48.68186317,-95.74894683,71.69891297,66.86944822,-70.12822282,
       52.36352661,-130.17602848,74.33260689,44.69661749,-26.65394757]
    y = mat(y)


    def Func(abc, iput):
        a = abc[0,0]
        b = abc[1,0]
        return a*np.cos(b*iput) + b*np.sin(a*iput)

    def Deriv(abc, iput, n):
        x1 = abc.copy()
        x2 = abc.copy()
        x1[n,0] -= 0.000001
        x2[n,0] += 0.000001
        p1 = Func(x1, iput)
        p2 = Func(x2, iput)
        d = (p2-p1)*1.0 / (0.000002)
        return d

    J = mat(np.zeros((n, 2))) # function*variable
    fx = mat(np.zeros((n,1)))
    fx_tmp = mat(np.zeros((n,1)))
    xk = mat([[input_1],[input_2]]) # initial
    lase_mse = 0
    step = 0
    u,v = 1,2
    conve = 100

    while(conve):
        mse, mse_tmp = 0,0
        step += 1
        # compute total mse error use current xk
        for i in range(n):
            fx[i] = Func(xk, h[i]) - y[0,i] # error
            mse += fx[i,0]**2

            for j in range(2):
                J[i,j] = Deriv(xk, h[i], j) # funtion*variable
        mse /= n

        H = J.T*J + u*np.eye(2) # u is lambda

        dx = -H.I * J.T * fx # H*delta(x) = g     g is J.T*fx
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

        print("step = %d, abs(mse-lase_mse) = %.8f" % (step, abs(mse-lase_mse)))

        lase_mse = mse
        conve -= 1


    print('xk:')
    print(xk)
    print('mse:')
    print(mse)

    plt.figure()
    y = np.array(y)
    plt.scatter(h, y[0], s=4)


    h = np.linspace(0, 2, 2000)
    z = [Func(xk, i) for i in h]
    plt.plot(h,z,'r')
    plt.show()

if __name__ == '__main__':
    #mse = 999999999
    #for i in range(0, 200):
    #    for j in range(0, 200):
    #        mse_tmp = func(float(i),float(j))
    #        if mse >  func(float(i),float(j)):
    #            mse = mse_tmp
    #            a=i
    #            b=j

    #print((a,b))
    func(float(52), float(99))
