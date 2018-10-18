import cv2
import numpy as np
from numpy import matrix as mat
import matplotlib.pyplot as plt

#cv2.imshow("img1", img1)
#cv2.waitKey(0)

N = 400
zone = 10.0

def get_pq(u, v, s, img):
    p = np.linspace(-zone, zone, N)
    q = np.linspace(-zone, zone, N)
    p, q = np.meshgrid(p,q)
    I = (p*s[0]+q*s[1]+1) / (np.sqrt(p*p+q*q+1)*np.sqrt(s[0]*s[0]+s[1]*s[1]+1))
    i = img[u,v]

    print(i)

   # plt.figure()
   # contour = plt.contour(p,q,I, 15) # 20 layer
   # plt.clabel(contour, fontsize=10)

    I = abs(I-i)
    idx = I < 0.018
    q_idx, p_idx = np.where(idx)

    return p_idx, q_idx


def process_one(u,v):
    count = np.zeros((N,N))
    p_s1, q_s1 = get_pq(u,v, s1, img1)
    p_s2, q_s2 = get_pq(u,v, s2, img2)
    p_s3, q_s3 = get_pq(u,v, s3, img3)
    p_s4, q_s4 = get_pq(u,v, s4, img4)

    count[p_s1, q_s1] += 1
    count[p_s2, q_s2] += 1
    count[p_s3, q_s3] += 1
    count[p_s4, q_s4] += 1

    if np.max(count) != 3:
        print(np.max(count))

    p_idx, q_idx = np.where(count == 3)

    p_idx = (p_idx / N - 0.5) * zone * 2
    q_idx = (q_idx / N - 0.5) * zone * 2

    q_idx = q_idx[p_idx < 0]
    p_idx = p_idx[p_idx < 0]

    p_idx = np.mean(p_idx)
    q_idx = np.mean(q_idx)


    #p_s1 = (p_s1/ N - 0.5) * zone * 2
    #q_s1 = (q_s1/ N - 0.5) * zone * 2
    #p_s2 = (p_s2/ N - 0.5) * zone * 2
    #q_s2 = (q_s2/ N - 0.5) * zone * 2
    #p_s3 = (p_s3/ N - 0.5) * zone * 2
    #q_s3 = (q_s3/ N - 0.5) * zone * 2
    #p_s4 = (p_s4/ N - 0.5) * zone * 2
    #q_s4 = (q_s4/ N - 0.5) * zone * 2

    #plt.figure()
    #plt.scatter(p_s1, q_s1)
    #plt.scatter(p_s2, q_s2)
    #plt.scatter(p_s3, q_s3)
    #plt.scatter(p_s4, q_s4)
    #plt.scatter(p_idx, q_idx)

    #plt.show()

    return p_idx, q_idx

if __name__ == '__main__':
    reuse  = True
    if reuse == False:
        img1 = cv2.imread('./img1_0.2_0_-1.png', cv2.IMREAD_GRAYSCALE) / 255.0
        img2 = cv2.imread('./img2_0_-0.2_-1.png', cv2.IMREAD_GRAYSCALE) / 255.0
        img3 = cv2.imread('./img3_0_0_-1.png', cv2.IMREAD_GRAYSCALE) / 255.0
        img4 = cv2.imread('./img4_0_0.2_-1.png', cv2.IMREAD_GRAYSCALE) / 255.0


        s1 = np.array([0.2, 0, -1])
        s1 = -s1
        s2 = np.array([0, -0.2, -1])
        s2 = -s2
        s3 = np.array([0, 0, -1])
        s3 = -s3
        s4 = np.array([0, 0.2, -1])
        s4 = -s4

        img1 = img1[4:94, 2:92]
        img2 = img2[2:92, 2:92]
        img3 = img3[2:92, 2:92]
        img4 = img4[2:92, 2:92]

        print(np.min(img1))
        print(np.min(img2))
        print(np.max(img3))
        print(np.max(img4))

        p = np.zeros((90,90))
        q = np.zeros((90,90))

        print('------------')
        p,q = process_one(45,45)
        print((p,q))
        p,q = process_one(75,5)
        print((p,q))
        exit(0)
        for i in range(0, 90):
            for j in range(0, 90):
                p_idx, q_idx = process_one(i,j)
                p[i,j] = p_idx
                q[i,j] = q_idx

        np.save('p_idx.npy', p)
        np.save('q_idx.npy', q)
    else:
        p = np.load('p_idx.npy')
        q = np.load('q_idx.npy')
        x = np.arange(90)
        y = np.arange(90)

        x,y = np.meshgrid(x,y)
        z = np.zeros((90,90))
        for i in range(90):
            for j in range(90):
                z[i,j] = -p[i,j]*(i-45) - q[i,j]*(j-45)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        print(x.shape)
        print(z.shape)
        print(y.shape)
        ax.plot_surface(x,y,z)

        plt.show()
