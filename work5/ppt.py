import cv2
import numpy as np
from numpy import matrix as mat
import matplotlib.pyplot as plt


def get_I(im1, im2, im3, im4):
    for i in range(90):
        for j in range(90):
            i1 = im1[i,j]
            i2 = im2[i,j]
            i3 = im3[i,j]
            i4 = im4[i,j]
            tmp = np.array([[i1],[i2],[i3],[i4]])
            if i == 0 and j == 0:
                I = tmp
            else:
                I = np.concatenate([I, tmp], -1)
    return I

def normalize_N(N):
    for i in range(90*90):
        tmp = N[:,i]
        tmp /= np.linalg.norm(tmp)
        N[:,i] = tmp
    return N

#def normalize_im(im):
#    im_min = np.min(im)
#    im -= im_min
#    im_max = np.min(im)
#    im /= im_max


def get_height(u,v,fx,fy, fz):
    sum_y = 0
    sum_x = 0
    for j in range(v):
        sum_y += (fy[0, j]/fz[0,j])
    for i in range(u):
        sum_x += (fx[i, v]/fz[i,v])
    return -(sum_y+sum_x)



def process_N(N):
    fx = np.zeros((90,90))
    fy = np.zeros((90,90))
    fz = np.zeros((90,90))
    for i in range(90):
        for j in range(90):
            fx[i,j] = N[0, i*90+j]
            fy[i,j] = N[1, i*90+j]
            fz[i,j] = N[2, i*90+j]
    return fx,fy,fz


if __name__ == '__main__':
    if 1:
        im1 = cv2.imread('(0 0 -1).png', cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread('(0 0.2 -1).png', cv2.IMREAD_GRAYSCALE)
        im3 = cv2.imread('(0 -0.2 -1).png', cv2.IMREAD_GRAYSCALE)
        im4 = cv2.imread('(0.2 0 -1).png', cv2.IMREAD_GRAYSCALE)

        #s1 = np.mat([0.2, 0, -1])
        #s2 = np.mat([0, -0.2, -1])
        #s3 = np.mat([0, 0, -1])
        #s4 = np.mat([0, 0.2, -1])

        s1 = np.mat([0, 0, -1])
        s2 = np.mat([0, 0.2, -1])
        s3 = np.mat([0, -0.2, -1])
        s4 = np.mat([0.2, 0, -1])
    else:
        im1 = cv2.imread('(0 0 -1).png', cv2.IMREAD_COLOR)
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im1 = np.array(im1)
        im2 = cv2.imread('(0 0.2 -1).png', cv2.IMREAD_COLOR)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        im2 = np.array(im2)
        im3 = cv2.imread('(0 -0.2 -1).png', cv2.IMREAD_COLOR)
        im3 = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)
        im3 = np.array(im3)
        im4 = cv2.imread('(0.2 0 -1).png', cv2.IMREAD_COLOR)
        im4 = cv2.cvtColor(im4, cv2.COLOR_RGB2GRAY)
        im4 = np.array(im4)

        s1 = np.mat([0, 0, -1])
        s2 = np.mat([0, 0.2, -1])
        s3 = np.mat([0, -0.2, -1])
        s4 = np.mat([0.2, 0, -1])



    I = get_I(im1, im2, im3, im4)
    S = np.concatenate((s1,s2,s3,s4))

    N = (S.T*S).I*S.T*I
    N = normalize_N(N)

    fx,fy,fz = process_N(N)

    # show
    x = np.arange(90)
    y = np.arange(90)

    x,y = np.meshgrid(x,y)
    z = -np.ones((90,90))
    z[0,0] = 0

    for i in range(89):
        for j in range(89):
            #z[i,j] = get_height(i,j, fx,fy, fz)
            if z[i,j] == -1:
                print('error')
            z[i,j+1] = -fy[i,j]/fz[i,j] + z[i,j]
            z[i+1,j] = -fx[i,j]/fz[i,j] + z[i,j]

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x,y,z)

    plt.show()

