import numpy as np
import cv2


def get_mat(src, dst):
    x = src[0]
    y = src[1]
    x_ = dst[0]
    y_ = dst[1]
    return np.array([[x,y,1,0,0,0,-x*x_, -x_*y, -x_],
                    [0,0,0,x,y,1,-y_*x, -y*y_, -y_]])


def init():
    img1 = cv2.imread('./img1.jpg')
    img2 = cv2.imread('./img2.jpg')

    ret, corners1 = cv2.findChessboardCorners(img1, (9,6), None)
    ret, corners2 = cv2.findChessboardCorners(img2, (9,6), None)

    cv2.drawChessboardCorners(img1, (9,6), corners1, ret)
    cv2.drawChessboardCorners(img2, (9,6), corners2, ret)

    #cv2.imshow('img1', img1)
    #cv2.imshow('img2', img2)
    #cv2.waitKey()

    corners1 = np.squeeze(corners1)
    corners2 = np.squeeze(corners2)

    homo, _ = cv2.findHomography(corners1, corners2)
    print('opencv h:')
    print(homo)

    mat0 = get_mat(corners1[0], corners2[0])
    mat1 = get_mat(corners1[9], corners2[9])
    mat2 = get_mat(corners1[21], corners2[21])
    mat3 = get_mat(corners1[36], corners2[36])

    mat = np.matrix(np.concatenate((mat0,mat1,mat2,mat3), axis=0))
    A = mat[:,:8]
    b = -mat[:,8]

    h = A.I*b
    h = np.concatenate((h, np.ones((1,1))), axis=0)
    return (corners1, corners2, h)

def Func(h, data):
    A = data[:, :8]
    b = -data[:,8]
    err = np.sum(b - A*h)
    return err

def Derive(h, data, n):
    h1 = h.copy()
    h2 = h.copy()
    h1[n,0] -= 0.000001
    h2[n,0] += 0.000001
    p1 = Func(h1, data)
    p2 = Func(h2, data)
    d = (p2-p1)*1.0 / 0.000002
    return d

if __name__ == '__main__':
    corners1, corners2, h = init()

    print('init:')
    print(h.reshape((3,3)))

    h = h[:8]
    # print(Func(h, corners1[0], corners2[0]))
    n = 54

    J = np.matrix(np.zeros((n,8)))
    fx = np.matrix(np.zeros((n,1)))
    fx_tmp = np.matrix(np.zeros((n,1)))
    xk = h
    lase_mse = 0
    step = 0
    u,v = 1,2
    conve = 100

    while(conve):
        mse, mse_tmp = 0, 0
        step += 1
        for i in range(n):
            data = get_mat(corners1[i], corners2[i])
            fx[i, 0] = Func(xk, data)
            mse += fx[i,0]**2

            for j in range(8):
                J[i,j] = Derive(xk, data, j)
        mse /= n

        H = J.T*J + u*np.eye(8)
        dx = -H.I * J.T * fx

        xk_tmp = xk.copy()
        xk_tmp += dx

        for i in range(n):
            data = get_mat(corners1[i], corners2[i])
            fx_tmp[i] = Func(xk_tmp, data)
            mse_tmp += fx_tmp[i,0]**2
        mse_tmp /= n

        q = (mse-mse_tmp) / ((dx.T*(u*dx - J.T*fx))[0,0])

        if q > 0:
            s = 1.0/3.0
            v = 2
            mse = mse_tmp
            xk = xk_tmp
            temp = 1 - pow(2*q-1,3)
            if s > temp:
                u*=s
            else:
                u*=temp
        else:
            u *= v
            v *= 2

        u = min(u, 1e200)

        if(abs(mse - lase_mse) < 0.0000001):
            break

        #print('step = %d, abs(mse-lase_mse) = %.8f' % (step, abs(mse-lase_mse)))
        lase_mse = mse
        conve -= 1


    h = np.concatenate((xk, np.ones((1,1))), axis=0)

    print('optimized h:')
    print(h.reshape((3,3)))

    print('mse:')
    print(mse)
