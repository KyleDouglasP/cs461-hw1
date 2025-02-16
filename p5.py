import numpy as np
import matplotlib.pyplot as plt 

x=np.load("x.npz")["x"]

mean = np.sum(x, axis=1, keepdims=True)/x.shape[1]
print(f'Mean vector:\n{mean}')

x_centered = x-mean
cov = np.matmul(x_centered,x_centered.T)/(x.shape[1]-1)
print(f'Covariance matrix:\n{cov}')
print()

# Spectral decomposition of the covariance matrix
E, L, ET = np.linalg.svd(cov)

# Performing the inverse root of the eigenvalues
L=1/(L**.5)

# .svd returns the eigenvalues in a vector format, but we need them in a diagonal matrix for the computations
L=np.diag(L)

A = np.matmul(L,ET)
print(f'A:\n{A}')

b = -np.matmul(A, mean)
print(f'b:\n{b}')
print()

W = np.matmul(A,x)+b

mean_W = np.sum(W, axis=1, keepdims=True)/W.shape[1]
print(f'W mean vector:\n{mean_W}')

# Determine if E[W] is approximately close enough to a vector of 0s 
print(f'Approximately close to mean=0? {np.allclose(mean_W, np.zeros_like(mean_W))}')

cov_W = np.matmul(W,W.T)/(W.shape[1]-1)
print(f'W covariance matrix:\n{cov_W}')

# Determine if COV[W,W] is approximately close enough to the identity matrix
print(f'Approximately close to identity matrix? {np.allclose(cov_W, np.eye(cov_W.shape[0]))}')

# For showcasing the whitening of the data visually with a 3D plot

fig = plt.figure(figsize=(12, 6))

# Plotting the original matrix X
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x[0], x[1], x[2], c=x[0]+x[2], s=1, alpha=0.5)
ax1.set_title("Original Data X")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("X3")

# Plotting the whitened matrix W
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(W[0], W[1], W[2], c=W[0]+W[2], s=1, alpha=0.5)
ax2.set_title("Whitened Data W")
ax2.set_xlabel("W1")
ax2.set_ylabel("W2")
ax2.set_zlabel("W3")

# Reducing the viewbox to make whitening more obvious
for ax in [ax1, ax2]:
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-3.5, 3.5)

plt.show()