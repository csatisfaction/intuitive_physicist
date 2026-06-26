import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Your optimized coordinates from the run
positions = np.array([
    [-0.27,  -0.123,  1.726], # Alpha 0
    [-2.22,  -0.371,  1.359], # Alpha 1
    [-1.628,  0.625,  2.988]  # Alpha 2
])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the center points
ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='red', s=200, label='Alpha Cores')

# Draw lines between them to show the triangle mesh
triangle = np.vstack([positions, positions[0]]) # Loop back to start
ax.plot(triangle[:,0], triangle[:,1], triangle[:,2], color='blue', linestyle='-', linewidth=2)

ax.set_title("Carbon-12 Alpha Cluster Geometry (Equilateral Triangle)")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()