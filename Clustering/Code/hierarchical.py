import numpy as np
import random
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
    Hierarchical Agglomerative clustering with Min Approach Implementation
"""
def hierarchical_clustering_min (k,gene,id):
    length = len(id)
    matrix = np.zeros((length,length,3))                # initial Distance matrix (p1_id, p2_id, distance)
    geneGroup = np.array([i for i in range(length)])    # initial geneGroup, 1*n array, each index contain a value of group.

    for i in range(len(gene)):
        for j in range(len(gene)):
            matrix[i,j,2] = sys.maxsize if i== j else np.linalg.norm(gene[i] - gene[j])             # calculate the distance between two points and add into matrix
            matrix[i, j, 0], matrix[i, j, 1] = i , j                                                # If i == j then distance is 0, ex. the distance from point 0 to itself is 0.

    col,row,d  = matrix.shape             # Get the number of cols, rows and dimensions of distance matrix
    while(col != k):                      # If number of cols not equal to k (final group number) then runs while loop, each time will decrease the cols by 1.

        x,y = np.unravel_index(matrix[:,:,2].argmin(), matrix[:,:,2].shape)             # Find index of smallest distance value

        i , j = int(matrix[x,y,0]), int(matrix[x,y,1])                                  # To get two points' id

        geneGroup[geneGroup == geneGroup[j]] = geneGroup[i]                             # Set second point's group number as first point's group number in geneGroup array.

        x_array = matrix[x,:,:]
        y_array = matrix[y,:,:]

        for num in range(len(x_array)):
            if(num != x and num != y):
                x_array[num,2] = min(x_array[num,2],y_array[num,2])             # Combine this two points cols and rows into one that each index is smallest value between two

        matrix = np.delete(matrix, y, 0)                           # Delete second point's row
        matrix = np.delete(matrix, y, 1)                           # Delete second point's col

        col, row, d = matrix.shape                              # Update cols, rows and dimensions of distance matrix

    record = np.unique(geneGroup)               # Get all unique group number
    geneGroup_sub = [[] for _ in range(k)]      # Initial return matrix.

    for idx,item in enumerate(geneGroup):       # Add points into same array if their group number is same.
        geneGroup_sub[int(np.where(record==item)[0])].append(idx)

    return geneGroup_sub


if __name__ == '__main__':
    """
       Data Pre-Processing
    """
    filename = "../new_dataset_2.txt"
    gene, id, result = pre_processing(filename)
    k = 7

    """
        Hierarchical Agglomerative clustering with Min Approach
     """
    cluster = "Hierarchical clustering (Min Approach)"
    geneGroup = hierarchical_clustering_min(k, gene, id)

    PCA_plot(gene, geneGroup,cluster,filename)

    JC, RD = Jaccard_Coefficient_and_Rand_Index(geneGroup, result)
    print("Jaccard Coefficient is : " + str(JC))
    print("Rand Index is : " + str(RD))