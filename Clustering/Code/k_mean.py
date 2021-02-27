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

# def PCA_plot(gene,geneGroup,center):
#     pca = PCA(n_components=2)
#     pca.fit(gene)
#     M = pca.transform(gene)
#
#     pca.fit(center)
#     k = pca.transform(center)
#
#     for idx, item in enumerate(geneGroup):
#         group = [[M[i][0],M[i][1]] for i in item]
#
#         # for x in group:
#         group = np.array(group)
#         plt.scatter(group[:,0], group[:,1], label=("Group "+str(idx)))
#
#     c = np.array([[i[0],i[1]] for i in k])
#     plt.scatter(c[:, 0], c[:, 1], label=("Center"))
#
#     plt.title("Dataset : " + filename + ' - PCA Plot')
#     plt.legend(loc=2)
#     plt.show()
"""
    PCA_plot Implement
"""
def PCA_plot(gene, geneGroup, cluster, filename):
    pca = PCA(n_components=2)
    pca.fit(gene)
    M = pca.transform(gene)

    for idx, item in enumerate(geneGroup):
        group = [[M[i][0], M[i][1]] for i in item]
        group = np.array(group)
        plt.scatter(group[:, 0], group[:, 1], label=("Group " + str(idx+1)))

    plt.title(cluster +" with data : "+ filename + ' - PCA Plot')
    plt.legend(loc=2)
    plt.show()

"""
    Jaccard Coefficient and Rand Index Calculate implement
"""
def Jaccard_Coefficient_and_Rand_Index(test, result):
    geneGroup_arr = np.zeros((len(result)))

    for index, item in enumerate(test):
        for x in item:
            geneGroup_arr[x] = index

    M_00, M_01, M_10, M_11 = 0 ,0 ,0 ,0
    for i, iter1 in enumerate(result):
        for j, iter2 in enumerate(result):
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
    K_Mean Implementation
"""
def SSE(gene,centers):
    error = [np.linalg.norm(gene - i) for i in centers]
    return min(error), error.index(min(error))

def k_mean(k,gene,central,iteration):
    retVal = [[] for _ in range(k)]
    geneGroup = [[] for _ in range(k)]
    # initial = np.sort(np.random.choice(id,k,replace=False))
    total_error = sys.maxsize
    center = np.array([gene[i-1]  for i in central])

    iter_time = 0
    while(iter_time < iteration):
        total_error_2 = 0
        retVal = [[] for _ in range(k)]
        geneGroup = [[] for _ in range(k)]

        for idx, item in enumerate(gene):
            error, group = SSE(item,center)
            total_error_2 += error
            retVal[group].append(item)
            geneGroup[group].append(idx)

        iter_time += 1

        if(total_error != total_error_2):
            total_error = total_error_2
        else:
            break



        center = [np.mean(i,axis=0) for i in retVal]

    # print(iter_time)

    return total_error,geneGroup,center


if __name__ == '__main__':
    """
       Data Pre-Processing
    """

    filename = "new_dataset_1.txt"
    gene, id, result = pre_processing(filename)
    k = 3
    iteration = 10
    central = [3,5,9]

    """
    K - Mean Clustering
    """
    cluster = "K_mean"
    error,geneGroup,center = k_mean(k,gene,central,iteration)

    PCA_plot(gene,geneGroup,cluster,filename)

    JC , RD = Jaccard_Coefficient_and_Rand_Index(geneGroup,result)
    print("Jaccard Coefficient is : " + str(JC))
    print("Rand Index is : " + str(RD))




