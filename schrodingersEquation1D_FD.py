'''
This solution is based on book Computational Quantum Mechanics Joshua Izaac and Jingbo Wang P9.7 pg 385
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
from sklearn.neighbors import KDTree
from numpy.linalg import solve
#Defining constants
L = 10
dx = 0.01
k = 1/(2*dx**2)
w = 1
n = 0 #Ground State
xCoords = np.arange(-L/2,L/2+dx, dx) #create the discrete x and y grids
numNodes = len(xCoords)
#Peridynamics constants
horizon = 3.015 
delta = horizon * dx
bVec = np.array([0,2])
diffOrder = 2
BC = np.array([[0,0],[100,0]])
diffOrderBC = 0
bVecBC = np.array([1,0])
numBC = 2
#np.savetxt("/home/doctajfox/Documents/Thesis_Research/SchrodingersEquation1D/data/sysMatrix.csv", sysMatrix, delimiter=",")
#np.savetxt("C:\\Users\\johnf\\Documents\\Thesis\\SchrodingersEquation1D\\data\\K.csv", K, delimiter = ",")
#np.savetxt("C:\\Users\\johnf\\Documents\\Thesis\\SchrodingersEquation1D\\data\\V.csv", V, delimiter = ",")

def calculatePotentialEnergy():
    V = 0.5*w**2*xCoords**2
    return V

##################################################################################################
#Finite Difference
##################################################################################################
def createKMatrix(V):
    data0 = np.array([(numNodes-3)*[-k]])
    data1 = np.array([(numNodes-2)*[2*k]])
    data2 = np.array([(numNodes-2)*[-k]])
    K = spdiags(data0, -1, numNodes-2, numNodes-2 ).toarray()\
            +spdiags(data1, 0, numNodes-2, numNodes-2 ).toarray()\
            +spdiags(data2, 1, numNodes-2, numNodes-2 ).toarray()
            #+spdiags(V[1:numNodes-1], 0, numNodes-2, numNodes-2 ).toarray()
    return K

##################################################################################################
#PDDO
##################################################################################################
def findFamilyMembers():
    tree = KDTree(xCoords.reshape((numNodes,1)), leaf_size=2)
    familyMembers, xis = tree.query_radius(xCoords.reshape((numNodes,1)), r = delta, sort_results=True, return_distance=True) 
    return familyMembers, xis

def calcSysMatrix(familyMembers, xis):
    sysMatrix = np.zeros([numNodes,numNodes])
    #Differential Equation Part
    for iNode in range(numNodes):
        family = familyMembers[iNode]
        xi = xis[iNode]
        diffMat = np.zeros([2,2])
        for iFamilyMember in range(len(family)):    
            currentFamilyMember = family[iFamilyMember]
            currentXi = xi[iFamilyMember]
            if iNode != currentFamilyMember:
                pList = np.array([1, (currentXi/delta)**2])
                weight = np.exp(-4*(np.absolute(currentXi)/delta)**2)
                diffMat += weight*np.outer(pList,pList)*dx/delta**diffOrder
        for iFamilyMember in range(len(family)):
            currentFamilyMember = family[iFamilyMember]
            currentXi = xi[iFamilyMember]
            if iNode != currentFamilyMember:
                pList = np.array([1, (currentXi/delta)**2])
                weight = np.exp(-4*(np.absolute(currentXi)/delta)**2);
                sysMatrix[iNode][ currentFamilyMember] = -0.5*weight*np.inner(solve(diffMat,bVec),pList)*dx/delta**diffOrder
    
    #Boundary Condition 
    sysMatrix[0][0] = 1
    sysMatrix[0][1] = 0
    sysMatrix[0][2] = 0
    sysMatrix[0][3] = 0
    sysMatrix[numNodes-1][numNodes-1] = 1
    sysMatrix[numNodes-1][numNodes-2] = 0
    sysMatrix[numNodes-1][numNodes-3] = 0
    sysMatrix[numNodes-1][numNodes-4] = 0
    np.savetxt("/home/doctajfox/Documents/Thesis_Research/SchrodingersEquation1D/data/sysMatrix.csv", sysMatrix, delimiter=",")

    return sysMatrix

def main():
    V = calculatePotentialEnergy() 
    #PDDO
    familyMembers, xis = findFamilyMembers()
    sysMatrix = calcSysMatrix(familyMembers, xis)
    EnPDDO, PhiPDDO = eigs(sysMatrix)
    K = createKMatrix(V)
    EnFD, PhiFD = eigs(K)

    print(np.shape(PhiFD))
    print(np.shape(PhiPDDO))
    #a = input('').split(" ")[0]
    fig, ax = plt.subplots()
    ax.plot(np.absolute(PhiFD[:,n]),marker='o')
    ax.plot(np.absolute(PhiPDDO[:,n]), marker='*')
    ax.grid()
    plt.show()

if __name__ == "__main__":
    main()
