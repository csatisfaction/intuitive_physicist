import numpy as np
import matplotlib.pyplot as plt

class SubstrateFluidGrid3D:
    def __init__(self, nx=44, ny=44, nz=44, dx=0.25):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        
        # 1. Initialize the 3D continuous substrate field fabric
        self.vx = np.zeros((nx, ny, nz), dtype=float)
        self.vy = np.zeros((nx, ny, nz), dtype=float)
        self.vz = np.zeros((nx, ny, nz), dtype=float)
        self.P  = np.zeros((nx, ny, nz), dtype=float)
        
        # Bedrock Kinematic Constants
        self.alpha = 0.45       # Viscous coupling constant
        self.iterations = 350   # Steps for vector propagation cascade
        
        # ---------------------------------------------------------------------
        # FLEX TEST CONFIGURATION: Changing from Horn Torus to Ring Torus
        # R_major (6.5) > r_minor (3.5) -> Leaves a 3.0 pixel open throat corridor
        # ---------------------------------------------------------------------
        self.R_major = 6.5      
        self.r_minor = 3.5      
        self.spin_vtheta = 8.5  # Tangential whirlpool strength
        self.pump_vr = 4.0      # Poloidal jetting strength

        # Pre-compute coordinate grid centered around origin (0,0,0)
        self.X, self.Y, self.Z = np.meshgrid(
            np.arange(nx) - nx//2, 
            np.arange(ny) - ny//2, 
            np.arange(nz) - nz//2, 
            indexing='ij'
        )
        
        # Define the ring torus core boundary mask
        self.d_xy = np.sqrt(self.X**2 + self.Y**2) + 1e-9
        self.r_dist = np.sqrt((self.d_xy - self.R_major)**2 + self.Z**2)
        self.core_mask = self.r_dist <= self.r_minor

    def apply_ring_torus_boundaries(self):
        """ Imposes the helical driving vectors strictly inside the ring torus mantle """
        # Component A: Tangential Circulation (v_theta)
        v_theta_x = -self.spin_vtheta * (self.Y / self.d_xy)
        v_theta_y =  self.spin_vtheta * (self.X / self.d_xy)
        
        # Component B: Poloidal Pumping (v_r) mapped relative to the ring filament axis
        phi = np.arctan2(self.Z, self.d_xy - self.R_major)
        
        v_r_axial_z =  self.pump_vr * np.cos(phi)
        v_r_radial_x = -self.pump_vr * np.sin(phi) * (self.X / self.d_xy)
        v_r_radial_y = -self.pump_vr * np.sin(phi) * (self.Y / self.d_xy)
        
        # Apply exclusively to the separated ring mask
        self.vx[self.core_mask] = v_theta_x[self.core_mask] + v_r_radial_x[self.core_mask]
        self.vy[self.core_mask] = v_theta_y[self.core_mask] + v_r_radial_y[self.core_mask]
        self.vz[self.core_mask] = v_r_axial_z[self.core_mask]

    def propagate_vector_fabric(self):
        """ Executes cell-by-cell 3D momentum relaxation """
        for step in range(self.iterations):
            self.apply_ring_torus_boundaries()
            
            for field in [self.vx, self.vy, self.vz]:
                left  = np.roll(field,  1, axis=0); right = np.roll(field, -1, axis=0)
                down  = np.roll(field,  1, axis=1); up    = np.roll(field, -1, axis=1)
                back  = np.roll(field,  1, axis=2); front = np.roll(field, -1, axis=2)
                
                neighbors_avg = (left + right + down + up + back + front) / 6.0
                
                # Active background updates including the newly opened central throat cells
                field[~self.core_mask] = (1.0 - self.alpha) * field[~self.core_mask] + self.alpha * neighbors_avg[~self.core_mask]
        
        # Map emergent pressure profile
        v_sq = self.vx**2 + self.vy**2 + self.vz**2
        self.P = 1.0 - 0.015 * v_sq  

    def render_3d_field_dashboard(self):
        """ Cuts 2D slices through the ring volume to test structural field stability """
        mid_y = self.ny // 2
        mid_z = self.nz // 2
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 7), facecolor='#111116')
        
        # PANEL 1: Vertical X-Z Plane Slice (Showing the Open central gate)
        axs[0].set_facecolor('#111116')
        P_xz_slice = self.P[:, mid_y, :].T
        im1 = axs[0].imshow(P_xz_slice, origin='lower', cmap='twilight_shifted',
                            extent=[-self.nx//2, self.nx//2, -self.nz//2, self.nz//2])
        
        vx_xz = self.vx[:, mid_y, :].T
        vz_xz = self.vz[:, mid_y, :].T
        grid_x, grid_z = np.meshgrid(np.arange(self.nx) - self.nx//2, np.arange(self.nz) - self.nz//2)
        axs[0].streamplot(grid_x, grid_z, vx_xz, vz_xz, color='cyan', density=1.3, linewidth=0.8, arrowsize=0.8)
        
        # Visual contour verifying the separated torus rings
        core_xz = self.core_mask[:, mid_y, :].T
        axs[0].contour(grid_x, grid_z, core_xz, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
        
        axs[0].set_title("Vertical X-Z Slice: The Open Central Gate\n" +
                         r"Un-choked Substrate Bypass Tunnel", color='white', pad=15, fontweight='bold')
        axs[0].tick_params(colors='white')
        
        # PANEL 2: Horizontal Toroidal Plane Slice (Cutting directly through the hole)
        axs[1].set_facecolor('#111116')
        P_xy_slice = self.P[:, :, mid_z].T
        im2 = axs[1].imshow(P_xy_slice, origin='lower', cmap='twilight_shifted',
                            extent=[-self.nx//2, self.nx//2, -self.ny//2, self.ny//2])
        
        vx_xy = self.vx[:, :, mid_z].T
        vy_xy = self.vy[:, :, mid_z].T
        grid_x2, grid_y2 = np.meshgrid(np.arange(self.nx) - self.nx//2, np.arange(self.ny) - self.ny//2)
        axs[1].streamplot(grid_x2, grid_y2, vx_xy, vy_xy, color='lime', density=1.3, linewidth=0.8, arrowsize=0.8)
        
        axs[1].set_title("Horizontal X-Y Slice: Central Hole Footprint\n" +
                         r"Emergent Low-Velocity/High-Pressure Core Vent", color='white', pad=15, fontweight='bold')
        axs[1].tick_params(colors='white')
        
        # Fixed syntax passing the axis array directly to prevent errors
        cbar = fig.colorbar(im1, ax=axs, location='bottom', pad=0.1, shrink=0.7)
        cbar.set_label("Substrate Pressure Landscape Field Score", color='white', labelpad=10)
        cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white')
        
        plt.show()

if __name__ == "__main__":
    print("Launching Ring Torus Flex Test...")
    sim = SubstrateFluidGrid3D()
    sim.propagate_vector_fabric()
    sim.render_3d_field_dashboard()