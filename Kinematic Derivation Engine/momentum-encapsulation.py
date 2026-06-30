import numpy as np
import matplotlib.pyplot as plt

class EncapsulatedAtomFabric:
    def __init__(self, x_bounds=(-2.5, 2.5), y_bounds=(-2.5, 2.5), resolution=65, c=1.0):
        self.res = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Substrate Field Arrays
        self.VX = np.zeros_like(self.X)
        self.VY = np.zeros_like(self.Y)
        
        # Hard boundary condition mask
        self.core_mask = np.zeros_like(self.X, dtype=bool)

    def inject_nested_cores(self, p_radius=0.35, p_spin=0.95, e_radius=1.3, e_spin=0.70):
        """
        Embeds both the proton and electron concentrically at origin (0,0).
        Both are driven in the EXACT SAME global counter-clockwise direction.
        """
        dx = self.X
        dy = self.Y
        r = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # 1. Proton Core boundary layout (Inner Ring Track)
        p_mask = np.abs(r - p_radius) < 0.10
        self.core_mask[p_mask] = True
        self.VX[p_mask] = -p_spin * np.sin(angle)[p_mask]
        self.VY[p_mask] = p_spin * np.cos(angle)[p_mask]
        
        # 2. Electron Core boundary layout (Outer Enclosure Ring Track)
        e_mask = np.abs(r - e_radius) < 0.12
        self.core_mask[e_mask] = True
        self.VX[e_mask] = -e_spin * np.sin(angle)[e_mask]
        self.VY[e_mask] = e_spin * np.cos(angle)[e_mask]

    def step_momentum_entrainment(self, passes=50, alpha=0.32):
        """
        Applies local neighbor-to-neighbor relaxation (\alpha \nabla^2 v).
        Propagates velocity vectors smoothly across the concentric gap.
        """
        for _ in range(passes):
            next_vx = self.VX.copy()
            next_vy = self.VY.copy()
            
            for i in range(1, self.res - 1):
                for j in range(1, self.res - 1):
                    if self.core_mask[i, j]:
                        continue  # Protect the hard-driven core current tracks
                        
                    # 4-Neighbor Laplacian blending matrix
                    avg_vx = (self.VX[i+1, j] + self.VX[i-1, j] + self.VX[i, j+1] + self.VX[i, j-1]) * 0.25
                    avg_vy = (self.VY[i+1, j] + self.VY[i-1, j] + self.VY[i, j+1] + self.VY[i, j-1]) * 0.25
                    
                    next_vx[i, j] = (1.0 - alpha) * self.VX[i, j] + alpha * avg_vx
                    next_vy[i, j] = (1.0 - alpha) * self.VY[i, j] + alpha * avg_vy
            
            # Enforce unyielding medium speed saturation limit (v <= c)
            v_mags = np.sqrt(next_vx**2 + next_vy**2)
            clamp = v_mags > self.c
            if np.any(clamp):
                next_vx[clamp] = (next_vx[clamp] / v_mags[clamp]) * self.c
                next_vy[clamp] = (next_vy[clamp] / v_mags[clamp]) * self.c
                
            self.VX, self.VY = next_vx, next_vy

    def get_bernoulli_pressure(self):
        return 1.0 - (self.VX**2 + self.VY**2)

# --- EXECUTION SandBox TIMELINE ---
if __name__ == "__main__":
    atom = EncapsulatedAtomFabric(resolution=65)
    
    # Inject nested, co-rotating primitives centered at origin (0,0)
    atom.inject_nested_cores(p_radius=0.4, p_spin=0.95, e_radius=1.4, e_spin=0.65)
    
    print("Executing first-principles concentric relaxation passes...")
    atom.step_momentum_entrainment(passes=60, alpha=0.35)
    
    P = atom.get_bernoulli_pressure()
    
    # --- RENDER MAP PROFILE ---
    plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    # Pressure floor contours mapping the emergent orbital containment trench
    contour = plt.contourf(atom.X, atom.Y, P, levels=80, cmap='inferno')
    cbar = plt.colorbar(contour, label='Substrate Hydrostatic Pressure (P)')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    # Coordinated velocity conveyor vectors
    mask = (atom.VX**2 + atom.VY**2) > 0.004
    plt.quiver(atom.X[mask], atom.Y[mask], atom.VX[mask], atom.VY[mask],
               color='white', alpha=0.55, scale=26, width=0.0022)
    
    # Geometrical annotations
    plt.text(0.0, 0.0, "Proton Engine Core\n(CCW)", color='magenta', fontsize=9, ha='center', va='center', fontweight='bold')
    plt.text(0.0, 0.85, "Co-Shearing Suction Channel\n(Low Friction Slipstream)", color='lime', fontsize=9, ha='center', fontweight='bold')
    plt.text(0.0, 1.6, "Encapsulating Electron Shield\n(CCW)", color='cyan', fontsize=9, ha='center', fontweight='bold')
    
    plt.title("TIR Substrate Engine: Emergent Hydrogen Soliton via Concentric Encapsulation", color='white', fontsize=12, pad=15)
    plt.xlabel("X Spatial Coordinate", color='white'); plt.ylabel("Y Spatial Coordinate", color='white')
    ax.tick_params(colors='white'); ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()