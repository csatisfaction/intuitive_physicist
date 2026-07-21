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
        self.alpha = 0.45       # Viscous coupling constant (substrate stiffness parameter)
        self.iterations = 350   # Steps allowed for the vector propagation cascade
        
        # 2. Establish pristine Horn Torus Invariants (R_major == r_minor)
        self.R_major = 5.0      # Major Radius (measured in grid pixels)
        self.r_minor = 5.0      # Minor Radius (Pinch Limit: creates zero-bypass point at origin)
        self.spin_vtheta = 8.5  # Tangential circulation strength (DC current component)
        self.pump_vr = 4.0      # Poloidal recirculation strength (AC jetting component)

        # Pre-compute coordinate grid centered around the origin (0,0,0)
        self.X, self.Y, self.Z = np.meshgrid(
            np.arange(nx) - nx//2, 
            np.arange(ny) - ny//2, 
            np.arange(nz) - nz//2, 
            indexing='ij'
        )
        
        # Define the exact core boundary mask
        self.d_xy = np.sqrt(self.X**2 + self.Y**2) + 1e-9
        self.r_dist = np.sqrt((self.d_xy - self.R_major)**2 + self.Z**2)
        self.core_mask = self.r_dist <= self.r_minor

    def apply_horn_torus_boundaries(self):
        """ Imposes the compound 3D helical velocity vectors inside the core torus boundary """
        # --- Component A: Tangential Circulation (v_theta) ---
        # Drives spinning velocity around the primary Z-axis of symmetry
        v_theta_x = -self.spin_vtheta * (self.Y / self.d_xy)
        v_theta_y =  self.spin_vtheta * (self.X / self.d_xy)
        
        # --- Component B: Poloidal Pumping (v_r) ---
        # Drives inside-out fluid lanes, looping through the central throat
        # Cross-sectional angle phi relative to the core pipe center loop line
        phi = np.arctan2(self.Z, self.d_xy - self.R_major)
        
        # Directing poloidal currents: Inward at equator, up the throat, outward at poles
        v_r_axial_z =  self.pump_vr * np.cos(phi)
        v_r_radial_x = -self.pump_vr * np.sin(phi) * (self.X / self.d_xy)
        v_r_radial_y = -self.pump_vr * np.sin(phi) * (self.Y / self.d_xy)
        
        # Superimpose both profiles directly inside the locked core mask
        self.vx[self.core_mask] = v_theta_x[self.core_mask] + v_r_radial_x[self.core_mask]
        self.vy[self.core_mask] = v_theta_y[self.core_mask] + v_r_radial_y[self.core_mask]
        self.vz[self.core_mask] = v_r_axial_z[self.core_mask]

    def propagate_vector_fabric(self):
        """ Executes cell-by-cell 3D momentum relaxation (The Spatial Laplacian cascade) """
        for step in range(self.iterations):
            # Enforce core driving inputs at the start of every propagation increment
            self.apply_horn_torus_boundaries()
            
            # Solve local neighbor blending across all three velocity axes simultaneously
            for field in [self.vx, self.vy, self.vz]:
                # 3D stencil reads the 6 orthogonal nearest neighbors
                left  = np.roll(field,  1, axis=0); right = np.roll(field, -1, axis=0)
                down  = np.roll(field,  1, axis=1); up    = np.roll(field, -1, axis=1)
                back  = np.roll(field,  1, axis=2); front = np.roll(field, -1, axis=2)
                
                neighbors_avg = (left + right + down + up + back + front) / 6.0
                
                # Update open substrate cells; core cells stand their ground as the anchor source
                field[~self.core_mask] = (1.0 - self.alpha) * field[~self.core_mask] + self.alpha * neighbors_avg[~self.core_mask]
        
        # 3. Post-processing: Map pristine Bernoulli Pressure P = 1.0 - v^2
        v_sq = self.vx**2 + self.vy**2 + self.vz**2
        self.P = 1.0 - 0.015 * v_sq  # Scaled visually for high-contrast topology rendering

    def render_3d_field_dashboard(self):
        """ Cuts 2D slices through the 3D volume to display the macro-envelope distributions """
        # Extract midplane slices
        mid_y = self.ny // 2
        mid_z = self.nz // 2
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 7), facecolor='#111116')
        
        # PANEL 1: Vertical Poloidal Plane Slice (X-Z cutting face, y=0)
        axs[0].set_facecolor('#111116')
        P_xz_slice = self.P[:, mid_y, :].T
        im1 = axs[0].imshow(P_xz_slice, origin='lower', cmap='twilight_shifted',
                            extent=[-self.nx//2, self.nx//2, -self.nz//2, self.nz//2])
        
        # Overlay 2D projected streamflow lines on the X-Z slice
        vx_xz = self.vx[:, mid_y, :].T
        vz_xz = self.vz[:, mid_y, :].T
        grid_x, grid_z = np.meshgrid(np.arange(self.nx) - self.nx//2, np.arange(self.nz) - self.nz//2)
        axs[0].streamplot(grid_x, grid_z, vx_xz, vz_xz, color='cyan', density=1.3, linewidth=0.8, arrowsize=0.8)
        
        # Outline the physical Horn Torus core boundary cross section
        core_xz = self.core_mask[:, mid_y, :].T
        axs[0].contour(grid_x, grid_z, core_xz, levels=[0.5], colors='white', linestyles='--', linewidths=1.5)
        
        axs[0].set_title("Vertical X-Z Slice: Nozzle Pump & Return Currents\n" +
                         r"Pristine Low-Pressure Vacuum Trench ($P = 1.0 - v^2$)", color='white', pad=15, fontweight='bold')
        axs[0].tick_params(colors='white')
        
        # PANEL 2: Horizontal Toroidal Plane Slice (X-Y cutting face, z=0)
        axs[1].set_facecolor('#111116')
        P_xy_slice = self.P[:, :, mid_z].T
        im2 = axs[1].imshow(P_xy_slice, origin='lower', cmap='twilight_shifted',
                            extent=[-self.nx//2, self.nx//2, -self.ny//2, self.ny//2])
        
        # Overlay 2D projected whirlpool currents on the X-Y slice
        vx_xy = self.vx[:, :, mid_z].T
        vy_xy = self.vy[:, :, mid_z].T
        grid_x2, grid_y2 = np.meshgrid(np.arange(self.nx) - self.nx//2, np.arange(self.ny) - self.ny//2)
        axs[1].streamplot(grid_x2, grid_y2, vx_xy, vy_xy, color='lime', density=1.3, linewidth=0.8, arrowsize=0.8)
        
        axs[1].set_title("Horizontal X-Y Slice: Macro Tangential Whirlpool\n" +
                         r"Emergent Isotropic Spherical Shroud Metrics", color='white', pad=15, fontweight='bold')
        axs[1].tick_params(colors='white')
        
        # Colorbar layout anchoring
        cbar = fig.colorbar(im1, ax=axs, location='bottom', pad=0.1, shrink=0.7)
        cbar.set_label("Substrate Pressure Landscape Field Score", color='white', labelpad=10)
        cbar.ax.xaxis.set_tick_params(color='white', labelcolor='white')
        
        plt.show()

if __name__ == "__main__":
    print("Initializing 3D First-Principles Substrate Grid...")
    sim = SubstrateFluidGrid3D()
    print("Simulating local cell-by-cell 3D momentum vector propagation cascade...")
    sim.propagate_vector_fabric()
    print("Rendering 3D Horn Torus Volume Slices...")
    sim.render_3d_field_dashboard()