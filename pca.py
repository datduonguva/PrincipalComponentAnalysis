import numpy as np
import matplotlib.pyplot as plt

class PCA():
	def __init__(self, n_components):
		self.n_components = n_components
	#This method calculates and returns the variance matrix
	def covariance(self, matrix):
		n_features = len(matrix[0])
		result = np.array([[0.0 for i in range(n_features)] for j in range(n_features)])
		for i in range(n_features):
			for j in range(i, n_features):
				result[i, j] = self.covavirance2(matrix[:, i], matrix[:, j])
				result[j, i] = result[i, j]
		return result
	#This method calculates and return the covariance between two features a and b.
	#a and b must be in numpy array format
	def covavirance2(self, a, b):
		return a.dot(b)/(len(a)-1)
	#This method finds the principal components a matrix. The number of components (features)
	#can be set manually by setting n_components.
	def fit_transform(self, data):
		if self.n_components > len(data[0]):
			self.n_components = len(data[0])
		matrix = np.array(data)
		self.mean = matrix.mean(axis = 0)
		matrix = matrix - self.mean
		covariance_matrix = self.covariance(matrix)
		weights, vectors = np.linalg.eig(covariance_matrix)
		principal_weights_index = np.argsort(weights)[-self.n_components:]
		principal_weights_index = [principal_weights_index[-1-i] for i in range(len(principal_weights_index))]
		self.explained_variances_ = weights[principal_weights_index]
		self.principal_vectors = np.array([vectors[:, i] for i in principal_weights_index]).T
		result =  matrix.dot(self.principal_vectors)
		return result
	# This method can only be called after self.fit_transform
	def transform(self, data):
		matrix = np.array(data)
		matrix = matrix -self.mean
		return matrix.dot(self.principal_vectors)


if __name__=='__main__':
	x = np.random.rand(250)
	y = 3*np.random.rand(250)-1 + 4*x
	
	
	pca = PCA(n_components = 2)
	data = np.array([x, y]).T
	principal_data = pca.fit_transform(data)

	print('Variances in decreasing order:\n\t', pca.explained_variances_)
	print('Initial shape: ', data.shape)
	print('Final shape  : ', principal_data.shape)

	#Let try to look at how it performed:
	plt.figure('Principal Component Analysis')
	plt.subplot(2, 1, 1)
	plt.title('Initial data')
	plt.scatter(data[:, 0], data[:, 1], s = 5, c = 'blue')
	plt.grid()
	plt.subplot(2, 1, 2)
	plt.title('Final data')
	plt.scatter(principal_data[:, 0], principal_data[:, 1], s = 5, c = 'red')
	plt.grid()
	plt.show()
