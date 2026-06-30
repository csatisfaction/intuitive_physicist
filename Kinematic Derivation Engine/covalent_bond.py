import numpy as np
import matplotlib.pyplot as plt

class CovalentSubstrateFabric:
    def __init__(self, x_bounds=(-3.5, 3.5), y_bounds=(-3.0, 3.0), resolution=85, c=1.0):
        self.res_x = resolution + 15  # Slightly wider for the diatomic shape
        self.res_y = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res_x)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.VX = np.zeros_like(self.X)
        self.VY = np.zeros_like(self.Y)
        self.core_mask = np.zeros_like(self.X, dtype=bool)

    def embed_proton(self, cx, cy, radius=0.4, spin_v=0.95):
        """ Inject a high-torque CCW proton anchor """
        dx = self.X - cx
        dy = self.Y - cy
        r = np.sqrt(dx**2 + dy**2)
        
        mask = np.abs(r - radius) < 0.12
        self.core_mask[mask] = True
        
        angle = np.arctan2(dy, dx)
        self.VX[mask] = -spin_v * np.sin(angle)[mask]
        self.VY[mask] = spin_v * np.cos(angle)[mask]

    def embed_covalent_shield(self, focus1_x, focus2_x, cy=0.0, major_axis=2.8, minor_axis=1.4, spin_v=0.65):
        """ 
        Embed an encapsulating electron track stretched around both protons.
        Modeled as a Cassini oval / Stadium geometry.
        """
        # Distance to focal points
        d1 = np.sqrt((self.X - focus1_x)**2 + (self.Y - cy)**2)
        d2 = np.sqrt((self.X - focus2_x)**2 + (self.Y - cy)**2)
        
        # Product of distances creates the continuous peanut/stadium shape
        product_dist = d1 * d2
        target_iso = (major_axis/2)**2
        
        mask = np.abs(product_dist - target_iso) < 0.5  # Boundary thickness
        self.core_mask[mask] = True
        
        # The vector flow must follow the contour of the stadium (CCW globally)
        # We calculate the perpendicular gradient of the Cassini field
        grad_y, grad_x = np.gradient(product_dist)
        
        # Normalize the flow track
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-9
        
        # Steer fluid CCW along the boundary
        self.VX[mask] = -spin_v * (grad_y[mask] / grad_mag[mask])
        self.VY[mask] = spin_v * (grad_x[mask] / grad_mag[mask])

    def step_momentum_diffusion(self, passes=60, alpha=0.35):
        """ Allow the space fabric to sort the velocities and find the lowest impedance state """
        for _ in range(passes):
            next_vx = self.VX.copy()
            next_vy = self.VY.copy()
            
            for i in range(1, self.res_y - 1):
                for j in range(1, self.res_x - 1):
                    if self.core_mask[i, j]:
                        continue 
                        
                    avg_vx = (self.VX[i+1, j] + self.VX[i-1, j] + self.VX[i, j+1] + self.VX[i, j-1]) * 0.25
                    avg_vy = (self.VY[i+1, j] + self.VY[i-1, j] + self.VY[i, j+1] + self.VY[i, j-1]) * 0.25
                    
                    next_vx[i, j] = (1.0 - alpha) * self.VX[i, j] + alpha * avg_vx
                    next_vy[i, j] = (1.0 - alpha) * self.VY[i, j] + alpha * avg_vy
            
            v_mags = np.sqrt(next_vx**2 + next_vy**2)
            clamp = v_mags > self.c
            if np.any(clamp):
                next_vx[clamp] = (next_vx[clamp] / v_mags[clamp]) * self.c
                next_vy[clamp] = (next_vy[clamp] / v_mags[clamp]) * self.c
                
            self.VX, self.VY = next_vx, next_vy

    def get_pressure_field(self):
        return 1.0 - (self.VX**2 + self.VY**2)


# --- EXECUTE ---
if __name__ == "__main__":
    space = CovalentSubstrateFabric(resolution=80)
    
    # 1. Place the two isolated Proton Anchors
    space.embed_proton(cx=-1.0, cy=0.0)
    space.embed_proton(cx=1.0, cy=0.0)
    
    # 2. Stretch the shared Electron wave-shield around both
    space.embed_covalent_shield(focus1_x=-1.0, focus2_x=1.0)
    
    print("Executing Figure-8 Substrate Entrainment...")
    space.step_momentum_diffusion(passes=70, alpha=0.38)
    
    P = space.get_pressure_field()
    
    # --- RENDER ---
    plt.figure(figsize=(14, 9), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    contour = plt.contourf(space.X, space.Y, P, levels=85, cmap='inferno')
    cbar = plt.colorbar(contour, label='Substrate Hydrostatic Pressure (P)')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    # Render the vector flow paths showing the shared slipstream
    mask = (space.VX**2 + space.VY**2) > 0.005
    plt.quiver(space.X[mask], space.Y[mask], space.VX[mask], space.VY[mask],
               color='white', alpha=0.6, scale=28, width=0.002)
    
    plt.text(-1.0, 0.0, "Proton A\n(CCW)", color='magenta', fontsize=9, ha='center', va='center', fontweight='bold')
    plt.text(1.0, 0.0, "Proton B\n(CCW)", color='magenta', fontsize=9, ha='center', va='center', fontweight='bold')
    plt.text(0.0, 1.8, "Shared Electron Wave-Shield\n(Covalent Encapsulation)", color='cyan', fontsize=10, ha='center', fontweight='bold')
    
    plt.title("TIR Substrate Engine: Covalent Bonding via Shared Kinematic Encapsulation (H2)", color='white', fontsize=13, pad=15)
    plt.xlabel("X Substrate Plane", color='white'); plt.ylabel("Y Substrate Plane", color='white')
    ax.tick_params(colors='white'); ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()