#! /usr/bin/env python3

import numpy as np



def cholesky(A):
    '''
    A: square,hermitan matrix
    L: cholesky separation
    '''
    if(np.shape(A)[0] != np.shape(A)[1]):
        raise(Exception('Matrix has to be square'))
    elif(np.any(A-np.conj(A.T))):
        raise(Exception('Matrix has to be hermitian'))
    L = np.zeros((np.shape(A)),dtype=np.complex)
    for row in range(np.shape(A)[0]):
        for column in range(row):
            L[row][column] = (1./L[column][column])*(A[row][column] - np.sum([L[row][mu]*np.conj(L[column][mu]) for mu in range(column)]))
        L[row][row] = np.sqrt(A[row][row] - np.sum([np.abs(L[row][mu])**2 for mu in range(row)]))
    return L


def solve_les_cholesky(A,b):
    L = cholesky(A)
    L_H = np.conj(L.T)
    y = np.zeros(([np.shape(A)[0],1]),dtype=np.complex)
    x = np.zeros(([np.shape(A)[0],1]),dtype=np.complex)
    for row in range(np.shape(A)[0]):
        for column in range(row):
            y[row] += y[column]*L[row][column]
        y[row] = (b[row]-y[row])/L[row][row]
    for row in reversed(range(np.shape(A)[0])):
        for column in range(row,np.shape(A)[1]):
            x[row] += L_H[row][column]*x[column]
        x[row] = (y[row]-x[row])/L_H[row][row]
    return x




def main():
    test_A = np.array(([9,6,-15,3],[6,5,-12,2],[-15,-12,45,-13],[3,2,-13,6]),dtype=np.complex)
    test_A2 = np.zeros(([3,5]),dtype=np.complex)
    test_A3 = np.array(([1,1],[-11,-1]),dtype=np.complex)
    test_b = np.array(([0],[-1],[10],[-5]),dtype=np.complex)
    tests = [test_A,test_A2,test_A3]
    for A in tests:
        try:
            L = cholesky(A)
            print(L)
            test_x = solve_les_cholesky(A,test_b)
            print(test_x)
        except Exception as e:
            print("{}".format(e.message))

if __name__  == "__main__":
    main()
