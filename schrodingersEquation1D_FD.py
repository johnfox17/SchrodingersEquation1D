'''
This solution is based on book Computational Quantum Mechanics Joshua Izaac and Jingbo Wang P9.7 pg 385
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

#Defining constants
L = 30
dx = 1
k = 1/(2*dx**2)
w=1
n = 2 #Ground State
En = w*(n+0.5)
xCoords = np.arange(-L/2,L/2+dx, dx) #create the discrete x and y grids
numNodes = len(xCoords)

def calculatePotentialEnergy():
    V = 0.5*w**2*xCoords**2
    return V

def createKMatrix(V):
    data0 = np.array([(numNodes-3)*[-k]])
    data1 = np.array([(numNodes-2)*[2*k-En]])
    data2 = np.array([(numNodes-2)*[-k]])
    K = spdiags(data0, -1, numNodes-2, numNodes-2 ).toarray()\
            +spdiags(data1, 0, numNodes-2, numNodes-2 ).toarray()\
            +spdiags(data2, 1, numNodes-2, numNodes-2 ).toarray()\
            +spdiags(V[1:numNodes-1], 0, numNodes-2, numNodes-2 ).toarray()
            #+spdiags(data2, 1, numNodes-2, numNodes-2 ).toarray()
    
    np.savetxt("/home/doctajfox/Documents/Thesis_Research/SolidStatePhysics/SchrodingersEquation1D/data/K.csv", K, delimiter=",")
    np.savetxt("/home/doctajfox/Documents/Thesis_Research/SolidStatePhysics/SchrodingersEquation1D/data/v.csv", V, delimiter=",")
    print(K)
    return csc_matrix(K)

def main():
    V = calculatePotentialEnergy() 
    print(np.shape(V[1:29]))
    K = createKMatrix(V)

    phi = spsolve(K,np.zeros(numNodes-2))

    b = np.zeros(numNodes-2) 
    phi = spsolve(K,b)

    #a = input('').split(" ")[0]
    fig, ax = plt.subplots()
    ax.plot(phi)
    ax.grid()
    plt.show()

if __name__ == "__main__":
    main()
