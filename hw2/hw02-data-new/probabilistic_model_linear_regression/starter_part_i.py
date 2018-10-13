import numpy as np
import matplotlib.pyplot as plt

Samples=[5,25,125]
Sigma = [0.5**2,1.5**2] #plotting for two different variance
w1_s = [0.4,1.3] # true W1 drawn from the gaussian with s.d 0.5 and 1.5 respectively
w2_s = [0.5,-1.7] # true W2 drawn from the gaussian with s.d 0.5 and 1.5 respectively

for s in range(2):
    sigma=Sigma[s]
    A=w1_s[s]
    B=w2_s[s]
    plt.figure()
    for count in range(3):
        n = Samples[count]
        # X = (X1,X2) Y = AX1+AX2+Z
        X1 = np.random.normal(0,1, n) # generating random samples for X1 from normal dist
        X2 = np.random.normal(0,1, n)
        Z = np.random.normal(0,1, n)
        Y = A*X1 +B*X2 +Z
        N=201
        W = np.linspace(-5,5,N)
        prob = np.ones([N,N])
        for i1 in range(N):
              w1 = W[i1] #taking different values for w1 from -5 to 5 to compute something 􏰁→ proportional to the posteriori probability
              for i2 in range(N):
                w2 = W[i2]
                L=1
                for i in range(n): # this part can vectorized as well
                    L = L*np.exp(-0.5*(Y[i]-X1[i]*w1-X2[i]*w2)**2)
                    L= L*np.exp(-0.5*(w1**2+w2**2)/sigma)
                    prob[i1][i2]=L
    plt.subplot(1,3,count+1)
    plt.imshow(np.flipud(prob), cmap='hot', aspect='auto',extent=[-5,5,-5,5])
    plt.show()
