import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib as mpl
import random

def get_rotation_matrix(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    # Normalize vector length
    i_v = np.divide(i_v, np.sqrt(np.dot(i_v, i_v)))
    # Get axis
    u, v, w = np.cross(i_v, unit)
    axis = np.array([u, v, w])
    u, v, w = np.divide(axis, np.sqrt(np.dot(axis, axis)))
    # Get angle
    phi = np.arccos(np.dot(i_v, unit))
    # Precompute trig values
    rcos = np.cos(phi)
    rsin = np.sin(phi)
    # Compute rotation matrix
    matrix = np.zeros((3, 3))
    matrix[0][0] = rcos + u * u * (1.0 - rcos)
    matrix[1][0] = w * rsin + v * u * (1.0 - rcos)
    matrix[2][0] = -v * rsin + w * u * (1.0 - rcos)
    matrix[0][1] = -w * rsin + u * v * (1.0 - rcos)
    matrix[1][1] = rcos + v * v * (1.0 - rcos)
    matrix[2][1] = u * rsin + w * v * (1.0 - rcos)
    matrix[0][2] = v * rsin + u * w * (1.0 - rcos)
    matrix[1][2] = -u * rsin + v * w * (1.0 - rcos)
    matrix[2][2] = rcos + w * w * (1.0 - rcos)
    return matrix

# Example Vector
origv = np.array([0.47404573,  0.78347482,  0.40180573])

# Get the 3D figure
fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(0, 100):
    origv = np.array([random.random(), random.random(), random.random()])

    # Compute the rotation matrix
    R = get_rotation_matrix(origv)

    # Apply the rotation matrix to the vector
    newv = np.dot(origv.T, R.T)

    # Plot the original and rotated vector
    ax.plot(*np.transpose([[0, 0, 0], origv]), label="original vector", color="r")
    ax.plot(*np.transpose([[0, 0, 0], newv]), label="rotated vector", color="b")

# Plot some axes for reference
ax.plot([0, 1], [0, 0], [0, 0], color='k')
ax.plot([0, 0], [0, 1], [0, 0], color='k')
ax.plot([0, 0], [0, 0], [0, 1], color='k')

# Show the plot and legend
ax.legend()
plt.show()
