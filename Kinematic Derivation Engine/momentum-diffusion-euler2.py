import numpy as np
import matplotlib.pyplot as plt

class CoRotatingSubstrateFabric:
    def __init__(self, x_bounds=(-3.0, 3.0), y_bounds=(-3.0, 3.0), resolution=60, c=1.0):
        self.res = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Space Fabric Velocity Fields
        self.VX = np.zeros_like(self.X)
        self.VY = np.zeros_like(self.Y)
        
        # Hard boundary tracking matrix
        self.core_mask = np.zeros_like(self.X, dtype=bool)

    def inject_proton_anchor(self, cx=-0.6, cy=0.0, radius=0.45, spin_v=0.95):
        """ Embeds a tight, high-torque central proton core spinning Counter-Clockwise """
        dx = self.X - cx
        dy = self.Y - cy
        r = np.sqrt(dx**2 + dy**2)
        
        mask = np.abs(r - radius) < 0.12
        self.core_mask[mask] = True
        
        angle = np.arctan2(dy, dx)
        self.VX[mask] = -spin_v * np.sin(angle)[mask]
        self.VY[mask] = spin_v * np.cos(angle)[mask]

    def inject_open_electron(self, cx=1.2, cy=0.0, radius=0.85, spin_v=0.65):
        """ 
        Embeds a wide, open electron disk. 
        CRITICAL: Globally CO-ROTATING (Counter-Clockwise) matching the proton!
        """
        dx = self.X - cx
        dy = self.Y - cy
        r = np.sqrt(dx**2 + dy**2)
        
        # Wide-open toroid track boundary
        mask = np.abs(r - radius) < 0.15
        self.core_mask[mask] = True
        
        angle = np.arctan2(dy, dx)
        # Counter-Clockwise spin axis matches the proton's global orientation
        self.VX[mask] = -spin_v * np.sin(angle)[mask]
        self.VY[mask] = spin_v * np.cos(angle)[mask]

    def step_momentum_entrainment(self, passes=40, alpha=0.3):
        """ Local neighbor relaxation stencil (carrier wave propagation mechanics) """
        for _ in range(passes):
            next_vx = self.VX.copy()
            next_vy = self.VY.copy()
            
            for i in range(1, self.res - 1):
                for j in range(1, self.res - 1):
                    if self.core_mask[i, j]:
                        continue  # Keep the particle source drives locked
                        
                    # 4-Neighbor Viscous Momentum Diffusion
                    avg_vx = (self.VX[i+1, j] + self.VX[i-1, j] + self.VX[i, j+1] + self.VX[i, j-1]) * 0.25
                    avg_vy = (self.VY[i+1, j] + self.VY[i-1, j] + self.VY[i, j+1] + self.VY[i, j-1]) * 0.25
                    
                    next_vx[i, j] = (1.0 - alpha) * self.VX[i, j] + alpha * avg_vx
                    next_vy[i, j] = (1.0 - alpha) * self.VY[i, j] + alpha * avg_vy
            
            # Enforce unyielding medium ceiling (v <= c)
            v_mags = np.sqrt(next_vx**2 + next_vy**2)
            clamp = v_mags > self.c
            if np.any(clamp):
                next_vx[clamp] = (next_vx[clamp] / v_mags[clamp]) * self.c
                next_vy[clamp] = (next_vy[clamp] / v_mags[clamp]) * self.c
                
            self.VX, self.VY = next_vx, next_vy

    def get_bernoulli_pressure(self):
        return 1.0 - (self.VX**2 + self.VY**2)

# --- EXECUTION SANBOX ---
if __name__ == "__main__":
    fabric = CoRotatingSubstrateFabric(resolution=60)
    
    # Initialize the subatomic geometries
    fabric.inject_proton_anchor(cx=-0.7, cy=0.0, radius=0.45, spin_v=0.95)
    fabric.inject_open_electron(cx=1.2, cy=0.0, radius=0.85, spin_v=0.70)
    
    # Cascade the vectors through local cell interaction
    fabric.step_momentum_entrainment(passes=45, alpha=0.38)
    P = fabric.get_bernoulli_pressure()
    
    # Render Plot Layout
    plt.figure(figsize=(12, 9), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    # Substrate pressure topography
    contour = plt.contourf(fabric.X, fabric.Y, P, levels=75, cmap='inferno')
    cbar = plt.colorbar(contour, label='Substrate Hydrostatic Pressure Profile (P)')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    # Entrainment direction vector arrows
    mask = (fabric.VX**2 + fabric.VY**2) > 0.005
    plt.quiver(fabric.X[mask], fabric.Y[mask], fabric.VX[mask], fabric.VY[mask],
               color='white', alpha=0.6, scale=25, width=0.0025)
    
    plt.text(-0.7, 0.65, "Proton Engine\n(CCW Whirlpool)", color='magenta', fontsize=10, ha='center', fontweight='bold')
    plt.text(1.2, 1.15, "Electron Throat\n(Open CCW Slipstream)", color='cyan', fontsize=10, ha='center', fontweight='bold')
    plt.text(0.25, -0.3, "Continuous Low-Pressure\nVacuum Tunnel", color='lime', fontsize=10, ha='center', fontweight='bold')
    
    plt.title("TIR Quantum Engine: Co-Rotating Open-Hole Electron Slipstream Lock-In", color='white', fontsize=12, pad=15)
    plt.xlabel("X Spatial Horizon", color='white'); plt.ylabel("Y Spatial Horizon", color='white')
    ax.tick_params(colors='white'); ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()