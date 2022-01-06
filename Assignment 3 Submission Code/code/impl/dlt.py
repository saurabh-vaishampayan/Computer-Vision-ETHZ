import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))
  points3D_new = np.append(points3D, np.ones((points3D.shape[0], 1)), 1)

  for i in range(num_corrs):
      constraint_matrix[i*2][0:4] = points3D_new[i]
      constraint_matrix[i*2][8:] = -points2D[i][0]*points3D_new[i]
      
      constraint_matrix[2*i+1][4:8] = -1*points3D_new[i]
      constraint_matrix[2*i+1][8:] = points2D[i][1]*points3D_new[i]
      
    # TODO Add your code here

  return constraint_matrix