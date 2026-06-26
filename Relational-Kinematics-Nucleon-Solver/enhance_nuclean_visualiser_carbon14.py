import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Paste your optimized data arrays here
positions = np.array([
    [ 0.253,  1.370,  0.721],  # Entity 0 (alpha)
    [-1.240, -1.466,  1.350],  # Entity 1 (alpha)
    [-2.418,  1.289,  0.943],  # Entity 2 (alpha)
    [-1.303, -0.477,  0.208],  # Entity 3 (neutron)
    [ 0.093, -1.027, -0.232]   # Entity 4 (neutron)
])

spins = np.array([
    [ 0.042, -0.183,  0.982],  # Vector 0
    [ 0.281,  0.333,  0.900],  # Vector 1
    [-0.009,  0.368,  0.930],  # Vector 2
    [-0.234,  0.263,  0.936],  # Vector 3
    [-0.062,  0.007,  0.998]   # Vector 4
])



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
num_entities = len(positions)

# 1. Plot Vortex Cores
ax.scatter(positions[:,0], positions[:,1], positions[:,2], 
           color='crimson', s=350, alpha=0.85, edgecolors='black', label='Alpha Nodes')

# Add numeric identifiers above each node
for i, pos in enumerate(positions):
    ax.text(pos[0], pos[1], pos[2] + 0.15, f'[{i}]', 
            color='black', weight='bold', fontsize=12, ha='center')

# 2. Map Inter-Entity Relations and Calculate Separation Radii
plotted_pairs = set()
for i in range(num_entities):
    for j in range(num_entities):
        if i != j and (j, i) not in plotted_pairs:
            p1, p2 = positions[i], positions[j]
            distance = np.linalg.norm(p1 - p2)
            
            # Draw relational boundary link
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='royalblue', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Calculate midpoint to anchor distance readout
            midpoint = (p1 + p2) / 2.0
            ax.text(midpoint[0], midpoint[1], midpoint[2], f'{distance:.3f}', 
                    color='darkblue', fontsize=9, weight='semibold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            plotted_pairs.add((i, j))

# 3. Project Vector Spin Axes (Quiver Engine)
# 'length' controls the visual extension of the arrow vectors
ax.quiver(positions[:,0], positions[:,1], positions[:,2],
          spins[:,0], spins[:,1], spins[:,2],
          length=0.6, color='black', linewidth=2.5, arrow_length_ratio=0.3,
          pivot='tail', label='Spin Vector Axis')

# 4. Global Scene Housekeeping
ax.set_title("TIR Relational Topography Visualizer", fontsize=14, weight='bold', pad=20)
ax.set_xlabel("X Space Matrix")
ax.set_ylabel("Y Space Matrix")
ax.set_zlabel("Z Space Matrix")

# Force equal axis aspect ratios to prevent distorting the tetrahedral geometry
max_range = np.array([positions[:,0].max()-positions[:,0].min(), 
                      positions[:,1].max()-positions[:,1].min(), 
                      positions[:,2].max()-positions[:,2].min()]).max() / 2.0

mid_x = (positions[:,0].max()+positions[:,0].min()) / 2.0
mid_y = (positions[:,1].max()+positions[:,1].min()) / 2.0
mid_z = (positions[:,2].max()+positions[:,2].min()) / 2.0

ax.set_xlim(mid_x - max_range - 0.5, mid_x + max_range + 0.5)
ax.set_ylim(mid_y - max_range - 0.5, mid_y + max_range + 0.5)
ax.set_zlim(mid_z - max_range - 0.5, mid_z + max_range + 0.5)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
