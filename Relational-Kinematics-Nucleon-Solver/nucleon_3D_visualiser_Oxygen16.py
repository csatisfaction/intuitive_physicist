import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Your optimized coordinates from the run
positions = np.array([
    [-0.056,  0.198,  0.86], # Alpha 0
    [-1.371, -1.102,  1.621], # Alpha 1
    [-1.967,  0.736,  1.105],  # Alpha 2
    [-1.475, -0.59,  -0.309]  # Alpha 3
])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the center points
ax.scatter(positions[:,0], positions[:,1], positions[:,2], color='red', s=200, label='Alpha Cores')

# Draw lines between them to show the triangle mesh
triangle = np.vstack([positions, positions[0]]) # Loop back to start
ax.plot(triangle[:,0], triangle[:,1], triangle[:,2], color='blue', linestyle='-', linewidth=2)

ax.set_title("Oxygen-16 Alpha Cluster Geometry")
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()