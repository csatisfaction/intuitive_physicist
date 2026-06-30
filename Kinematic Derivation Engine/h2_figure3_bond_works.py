import numpy as np
import matplotlib.pyplot as plt

class TrueDipoleBondingFabric:
    def __init__(self, x_bounds=(-3.0, 3.0), y_bounds=(-2.5, 2.5), resolution=95, c=1.0):
        self.res_x = resolution + 20
        self.res_y = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res_x)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.VX = np.zeros_like(self.X)
        self.VY = np.zeros_like(self.Y)
        self.core_mask = np.zeros_like(self.X, dtype=bool)

    def generate_superposition_bond(self, cx_A=-0.8, cx_B=0.8, p_rad=0.3, e_rad=1.0, p_vel=0.95, e_vel=0.65):
        """
        Calculates the true, frictionless fluid superposition of a Spin-Paired binary system.
        Atom A is CCW. Atom B is CW.
        """
        # Distance arrays
        dx_A = self.X - cx_A
        dy_A = self.Y
        r_A = np.sqrt(dx_A**2 + dy_A**2) + 1e-9
        
        dx_B = self.X - cx_B
        dy_B = self.Y
        r_B = np.sqrt(dx_B**2 + dy_B**2) + 1e-9

        # Raw Kinematic Field for Atom A (CCW)
        vx_A = -np.sin(np.arctan2(dy_A, dx_A))
        vy_A = np.cos(np.arctan2(dy_A, dx_A))

        # Raw Kinematic Field for Atom B (CW)
        vx_B = np.sin(np.arctan2(dy_B, dx_B))
        vy_B = -np.cos(np.arctan2(dy_B, dx_B))

        # 1. Embed the rigid Proton Core Engines
        mask_pA = r_A < p_rad
        mask_pB = r_B < p_rad
        self.core_mask[mask_pA | mask_pB] = True
        
        self.VX[mask_pA] = p_vel * vx_A[mask_pA]
        self.VY[mask_pA] = p_vel * vy_A[mask_pA]
        
        self.VX[mask_pB] = p_vel * vx_B[mask_pB]
        self.VY[mask_pB] = p_vel * vy_B[mask_pB]

        # 2. Superimpose the fields for the Electron Figure-8 Track
        # We add the vectors directly (weighted by distance) to find the path of zero shear
        vx_tot = (vx_A / r_A) + (vx_B / r_B)
        vy_tot = (vy_A / r_A) + (vy_B / r_B)
        mag_tot = np.sqrt(vx_tot**2 + vy_tot**2) + 1e-9
        
        # Normalize and set to electron speed
        vx_e = e_vel * (vx_tot / mag_tot)
        vy_e = e_vel * (vy_tot / mag_tot)

        # Apply this perfectly blended vector ONLY to the overlapping electron radii
        mask_e_tracks = (np.abs(r_A - e_rad) < 0.1) | (np.abs(r_B - e_rad) < 0.1)
        valid_e_mask = mask_e_tracks & ~self.core_mask # Don't overwrite the protons!
        
        self.core_mask[valid_e_mask] = True
        self.VX[valid_e_mask] = vx_e[valid_e_mask]
        self.VY[valid_e_mask] = vy_e[valid_e_mask]

    def step_momentum_diffusion(self, passes=65, alpha=0.35):
        """ Run the active substrate relaxation to let the pressure map emerge """
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
            
            # Enforce unyielding fluid capacity ceiling
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
    space = TrueDipoleBondingFabric(resolution=95)
    
    # Generate the pristine superimposed vector bond
    space.generate_superposition_bond(cx_A=-0.8, cx_B=0.8)
    
    print("Diffusing substrate to map zero-shear covalent topology...")
    space.step_momentum_diffusion(passes=85, alpha=0.38)
    
    P = space.get_pressure_field()
    
    # --- RENDER ---
    plt.figure(figsize=(14, 9), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    contour = plt.contourf(space.X, space.Y, P, levels=90, cmap='inferno')
    cbar = plt.colorbar(contour, label='Substrate Hydrostatic Pressure (P)')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    mask = (space.VX**2 + space.VY**2) > 0.005
    plt.quiver(space.X[mask], space.Y[mask], space.VX[mask], space.VY[mask],
               color='white', alpha=0.6, scale=28, width=0.002)
    
    plt.text(-0.8, 0.0, "Proton A\n(CCW)", color='magenta', fontsize=9, ha='center', va='center', fontweight='bold')
    plt.text(0.8, 0.0, "Proton B\n(CW)", color='cyan', fontsize=9, ha='center', va='center', fontweight='bold')
    plt.text(0.0, -1.3, "Continuous Figure-8 Dipole\n(Zero-Shear Covalent Encapsulation)", color='lime', fontsize=10, ha='center', fontweight='bold')
    
    plt.title("TIR Engine: Perfect Vector Superposition (H2 Spin-Paired Bond)", color='white', fontsize=13, pad=15)
    plt.xlabel("X Substrate Plane", color='white'); plt.ylabel("Y Substrate Plane", color='white')
    ax.tick_params(colors='white'); ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()