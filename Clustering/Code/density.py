import numpy as np
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



"""
    Data Pre-Processing Implementation
"""
def pre_processing(filename):
    contents = open(filename, 'r')
    firstLine = contents.readline().split()
    contents.close()
    num = [i for i in range(2,len(firstLine))]

    gene = np.genfromtxt(filename, usecols=num)
    result = np.genfromtxt(filename, usecols=(1), dtype='str').astype(int)
    id = np.genfromtxt(filename, usecols=(0), dtype='str').astype(int)

    return gene, id, result

"""
    PCA_plot Implement
"""
def PCA_plot(gene, geneGroup, cluster, filename):
    pca = PCA(n_components=2)
    pca.fit(gene)
    M = pca.transform(gene)

    for idx, item in enumerate(geneGroup):
        group = [[M[i][0], M[i][1]] for i in item]

        # for x in group:
        group = np.array(group)
        if(idx == len(geneGroup)-1):
            if(len(geneGroup[-1])!=0):
                plt.scatter(group[:, 0], group[:, 1], label=("Outlier"))
        else:
            plt.scatter(group[:, 0], group[:, 1], label=("Group " + str(idx+1)))

    plt.title(cluster +" with data : "+ filename + ' - PCA Plot')
    plt.legend(loc=2)
    plt.show()

"""
    Jaccard Coefficient and Rand Index Calculate implement
"""
def Jaccard_Coefficient_and_Rand_Index(test, result):
    # JC = a / (a + b + c)
    groundTrue, check_matrix = np.zeros((len(result),len(result))), np.zeros((len(result),len(result)))
    geneGroup_arr = np.zeros((len(result)))

    for index, item in enumerate(test):
        for x in item:
            geneGroup_arr[x] = index

    M_00, M_01, M_10, M_11 = 0 ,0 ,0 ,0
    for i, iter1 in enumerate(result):
        for j, iter2 in enumerate(result):
            # groundTrue[i,j] = 1 if iter1 == iter2 else 0
            # check_matrix[i,j] = 1 if geneGroup_arr[i] == geneGroup_arr[j] else 0
            if iter1 == iter2:
                if geneGroup_arr[i] == geneGroup_arr[j]:
                    M_11 += 1
                else:
                    M_10 += 1
            else:
                if geneGroup_arr[i] == geneGroup_arr[j]:
                    M_01 += 1
                else:
                    M_00 += 1

    JC = M_11 / (M_11 + M_10 + M_01)

    RandIndex = (M_11 + M_00) / (M_11 + M_00 + M_10 + M_01)

    return JC , RandIndex

"""
    Density-Based Clustering Implementation
"""

def DBSCAN(gene,id,epsilon,minPts):
    visited, in_cluster, noise, core, border, geneGroup = set(), set(), set(), set(), set(), []

    for index, item in enumerate(gene):
        if index not in visited:
            visited.add(index)                                       # Add point into visited set
            neighborPts = regionQuery(gene,gene[index],epsilon)      # Find all the neighbor points that distance less or equal to epsilon

            if(len(neighborPts) < minPts):                    # If size of neighborPts less than minPts, Mark p as noise
                noise.add(index)
            else:
                geneGroup.append([])
                geneGroup, visited, in_cluster, core, noise, border = expandCluster(visited, in_cluster, noise, core, border, gene, index, neighborPts, geneGroup, epsilon,minPts)

    arr = []
    for i in noise:                             # Add all noise point as one cluster
        arr.append(i)
    geneGroup.append(arr)

    return geneGroup, noise

def expandCluster(visited, in_cluster, noise, core, border, gene,index, neighborPts, geneGroup, eps, minPts):
    geneGroup[len(geneGroup)-1].append(index)
    core.add(index)
    in_cluster.add(index)

    for neighbor_index in neighborPts:
        if neighbor_index not in visited :                           # Check neighbor point is not visited
            visited.add(neighbor_index)                                   # add point to visited set
            neighborPts_2 = regionQuery(gene,gene[neighbor_index],eps)

            if len(neighborPts_2) >= minPts :
                core.add(neighbor_index)                                  # Add point to core set

                for item in neighborPts_2:                                # NeighborPts is a set that all the point has a core point within epsilon radius
                    if item not in neighborPts:
                        neighborPts.append(item)
            else:                                                         # If it is not core then it is noise
                noise.add(neighbor_index)

        if neighbor_index not in in_cluster:                                     # If point not belong to any cluster
            geneGroup[len(geneGroup)-1].append(neighbor_index)                              # Add it to current cluster
            in_cluster.add(neighbor_index)                                               # Set it have already belonged to a cluster
            if(neighbor_index in noise):        # If this point is noise then set it as border point
                noise.remove(neighbor_index)
                border.add(neighbor_index)

    return geneGroup, visited, in_cluster, core, noise, border,

def regionQuery(gene,p,eps):
    retVal = []
    for idx, item in enumerate(gene):
        if(eps >= np.linalg.norm(item - p)):
            retVal.append(idx)
    return retVal

if __name__ == '__main__':
    """
       Data Pre-Processing
    """
    filename = "../cho.txt"
    gene, id, result = pre_processing(filename)

    """
            Density-Based Clustering
    """
    cluster = "Density-Based Clustering"

    minPts = 4
    epsilon = 1

    geneGroup, noise = DBSCAN(gene, id, epsilon, minPts)
    PCA_plot(gene, geneGroup, cluster, filename)

    JC, RD = Jaccard_Coefficient_and_Rand_Index(geneGroup, result)
    print("Jaccard Coefficient is : " + str(JC))
    print("Rand Index is : " + str(RD))