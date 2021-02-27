import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def matrix(file):
    # contents = open(file).read()
    # M = np.array([item.split() for item in contents.split('\n')[:-1]])

    contents = open(file, 'r')
    firstLine = contents.readline().split()
    contents.close()

    num = [i for i in range(len(firstLine)-1)]

    data = np.genfromtxt(file, usecols=num)
    N = np.genfromtxt(file,usecols=(len(firstLine)-1),dtype='str')

    return data, N

def PCA(M,N,dimension):
    mean_vector = np.mean(M, axis=0)                 # Get Each Colums Mean Value

    m = M - mean_vector                              # Subtreacting the Mean of each Column

    cov_matrix = np.cov(m.transpose())
    # cov_matrix = np.dot(m.transpose(),m) / (len(m) - 1)                            # Get Convariance Matrix

    eigen_value, eigen_vector = LA.eig(cov_matrix)      # Find Eigen Value and Eigen Vector

    eigen_combine = [(eigen_value[i], eigen_vector.transpose()[i]) for i in range(len(eigen_value))]   # Conbine Eigen Value and Eigen Vector together

    eigen_combine.sort(reverse=True)                      # Sort by Eigen Value

    new_feature = np.array([eig_v[1] for eig_v in eigen_combine[:dimension]])   # Reduce Diemention To K Diemention

    rdm = np.dot(m,new_feature.transpose())                             # To get new feature's number

    name_list = np.unique(N)                                            # Get All Disease labels

    for i, name in enumerate(name_list):                                # Draw Scatter Plot

        l = np.where(N == name)                                         # Get index list of same name
        arr = [rdm[i] for i in l]

        plt.scatter(arr[0][:,0],arr[0][:,1],label = name)

    plt.title("Dataset : "+filename+' - PCA Plot')
    plt.legend(loc=2)
    plt.show()

def SVD(M,N,dimension):
    # u,sigma,vt = LA.svd(M)                                # Get U, Sigma, V_T value from Matrix M.Transpose
    #
    # new_feature = sigma[:dimension]                         # Reduce Dimension with new Sigma
    #
    # svd = np.dot(u[:,:dimension],np.diag(new_feature))
    #
    # print(u)
    # points = np.dot(M,svd)                                  # Dot with SVD value to get new features.
    u ,sigma, vt = LA.svd(M)

    points = u[:,:dimension]

    name_list = np.unique(N)                                # Get All Disease labels

    for i, name in enumerate(name_list):                     # Draw Scatter Plot

        l = np.where(N == name)                              # Get index list of same name
        arr = [points[i] for i in l]

        plt.scatter(arr[0][:, 0], arr[0][:, 1], label=name)

    plt.title("Dataset : "+filename+' - SVD Plot')
    plt.legend(loc=2)
    plt.show()

def t_SNE(M,N,dimension):
    points = TSNE(n_components=dimension).fit_transform(M)

    name_list = np.unique(N)  # Get All Disease labels

    for i, name in enumerate(name_list):  # Draw Scatter Plot

        l = np.where(N == name)  # Get index list of same name
        arr = [points[i] for i in l]

        plt.scatter(arr[0][:, 0], arr[0][:, 1], label=name)

    plt.title("Dataset : "+filename+' - t-SNE Plot')
    plt.legend(loc=2)
    plt.show()

    print(points.shape)
if __name__ == '__main__':
    filename = 'pca_demo.txt'
    M,N = matrix(filename)

    PCA(M,N,2)
    SVD(M,N,2)
    t_SNE(M,N,2)