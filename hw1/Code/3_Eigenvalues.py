import numpy as np
from scipy import linalg
A   = np.array([[2,-4],[-1,-1]])
B   = np.array([[3, 1],[ 1, 3]])
A_2 = A.dot(A)
B_2 = B.dot(B)
AB  = A.dot(B)
BA  = B.dot(A)
C   = np.array([[3, 1],[ 1, 3],[ 2,-4],[-1,-1]])

matrices = [A,B,A_2,B_2,AB,BA,C]
names = ['A','B','A_2','B_2','AB','BA','C']
for i in len(0,len(matrices),1):
    U_A, s_A, Vh_A = linalg.svd(A)
    print ('For matrix ' + str(matrices(i)))
    print ('The value of U = ')
    print U_A
    print ('The value of S = ')
    print s_A
    print ('The value of V = ')
    print Vh_A
U_B, s_B, Vh_B = linalg.svd(B)
print ('The value of U = ')
print U_B
print ('The value of S = ')
print s_B
print ('The value of V = ')
print Vh_B
U_A_2, s_A_2, Vh_A_2 = linalg.svd(A_2)

U_B_2, s_B_2, Vh_B_2 = linalg.svd(B_2)

U_AB, s_AB, Vh_AB = linalg.svd(AB)

U_BA, s_BA, Vh_BA = linalg.svd(BA)

U_C, s_C, Vh_C = linalg.svd(C)
U.shape,  s.shape, Vh.shape
((9, 9), (6,), (6, 6))
