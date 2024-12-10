import scipy.linalg

# Define a test matrix
matrix = [[1, 2], [3, 4]]  # Simple 2x2 matrix

# Convert the test matrix to a dense array
dense_matrix = matrix  # This is already a dense matrix in this case
values, vectors = scipy.linalg.eig(dense_matrix)  # Calculate eigenvalues and eigenvectors

# Print the results
print("Eigenvalues:", values)
