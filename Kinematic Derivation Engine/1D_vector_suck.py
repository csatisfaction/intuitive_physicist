import numpy as np
import matplotlib.pyplot as plt

class ActiveSubstrateGrid:
    def __init__(self, x_bounds=(-2.5, 2.5), y_bounds=(-2.5, 2.5), resolution=40):
        """
        An active space fabric where coordinates are real fluid vector cells 
        capable of dragging, aligning, and entraining their neighbors.
        """
        self.res = resolution
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Space cells hold active 2D fluid velocity vectors
        self.VX = np.zeros_like(self.X)
        self.VY = np.zeros_like(self.Y)
        
    def inject_toroid_core_drive(self, center_x, center_y, radius, spin_velocity):
        """
        Forces the substrate grid cells directly touching the 'train track' 
        of the toroid core to match the coordinated forward wheel sprint.
        """
        # Calculate distances from each cell to the ring center
        dx = self.X - center_x
        dy = self.Y - center_y
        r_cells = np.sqrt(dx**2 + dy**2)
        
        # Mask identifying the cells sitting right on the core diameter
        core_mask = np.abs(r_cells - radius) < 0.2
        
        if np.any(core_mask):
            # Tangential unit vectors for the circular stream drive
            angle = np.arctan2(dy, dx)
            self.VX[core_mask] = -spin_velocity * np.sin(angle)[core_mask]
            self.VY[core_mask] = spin_velocity * np.cos(angle)[core_mask]

    def execute_entrainment_relaxation(self, relaxation_passes=12, alpha=0.25, c=1.0):
        """
        The repeating simple function: Cells sideswipe their neighbors, 
        sharing momentum and forcing chaotic vectors to align along the track.
        """
        for _ in range(relaxation_passes):
            # Create copies to update field synchronously
            next_vx = self.VX.copy()
            next_vy = self.VY.copy()
            
            # 4-Neighbor averaging matrix (Viscous Shear relaxation)
            for i in range(1, self.res - 1):
                for j in range(1, self.res - 1):
                    # Skip the hard-driven core cells to maintain the 'train pump'
                    if self.VX[i, j]**2 + self.VY[i, j]**2 > 0.8 * (c**2):
                        continue
                        
                    # Sample neighbors
                    avg_vx = (self.VX[i+1, j] + self.VX[i-1, j] + self.VX[i, j+1] + self.VX[i, j-1]) * 0.25
                    avg_vy = (self.VY[i+1, j] + self.VY[i-1, j] + self.VY[i, j+1] + self.VY[i, j-1]) * 0.25
                    
                    # Relax cell state toward its neighbors' alignment profile
                    next_vx[i, j] = (1.0 - alpha) * self.VX[i, j] + alpha * avg_vx
                    next_vy[i, j] = (1.0 - alpha) * self.VY[i, j] + alpha * avg_vy
            
            # Enforce the unyielding medium ceiling (v <= c)
            v_mags = np.sqrt(next_vx**2 + next_vy**2)
            clamp_mask = v_mags > c
            if np.any(clamp_mask):
                next_vx[clamp_mask] = (next_vx[clamp_mask] / v_mags[clamp_mask]) * c
                next_vy[clamp_mask] = (next_vy[clamp_mask] / v_mags[clamp_mask]) * c
                
            self.VX, self.VY = next_vx, next_vy

    def compute_bernoulli_pressure(self):
        v_sq = self.VX**2 + self.VY**2
        return 1.0 - v_sq

# --- FIRE UP THE ACTIVE SUBSTRATE SANDBOX ---
if __name__ == "__main__":
    # Create the space medium grid
    fabric = ActiveSubstrateGrid(resolution=50)
    
    # Inject a single spinning particle core ('The Train') centered at the origin
    fabric.inject_toroid_core_drive(center_x=0.0, center_y=0.0, radius=0.9, spin_velocity=0.95)
    
    # Run the simple repeating entrainment function to let vectors drag their neighbors
    fabric.execute_entrainment_relaxation(relaxation_passes=20, alpha=0.3)
    
    # Evaluate the resulting structural pressure landscape
    P = fabric.compute_bernoulli_pressure()
    
    # --- RENDER THE TRAIN-SUCK PROFILE ---
    plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    # 1. Background contour mapping the emergent suction valleys
    contour = plt.contourf(fabric.X, fabric.Y, P, levels=60, cmap='viridis')
    cbar = plt.colorbar(contour, label='Substrate Static Pressure Floor (P)')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    # 2. Overlay vector arrows showing the drag/alignment flow spreading out
    # Skips cells with zero velocity to keep the canvas razor-sharp
    mask = (fabric.VX**2 + fabric.VY**2) > 0.005
    plt.quiver(fabric.X[mask], fabric.Y[mask], fabric.VX[mask], fabric.VY[mask], 
               color='white', alpha=0.6, scale=22, width=0.003)
    
    plt.title("Emergent 'Train-Suck' Valley via Active Substrate Entrainment", color='white', fontsize=13, pad=15)
    plt.xlabel("Substrate X", color='white'); plt.ylabel("Substrate Y", color='white')
    ax.tick_params(colors='white'); ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()