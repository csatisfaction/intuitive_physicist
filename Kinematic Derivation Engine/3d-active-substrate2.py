import numpy as np
import matplotlib.pyplot as plt

class DynamicSubstrateEngine3D:
    def __init__(self, nx=44, ny=44, nz=44, dx=0.25):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        
        # 1. Initialize the living 3D substrate fields
        self.rho = np.ones((nx, ny, nz), dtype=float)       # Mass density ledger
        self.vx  = np.zeros((nx, ny, nz), dtype=float)      # Velocity X
        self.vy  = np.zeros((nx, ny, nz), dtype=float)      # Velocity Y
        self.vz  = np.zeros((nx, ny, nz), dtype=float)      # Velocity Z
        self.P   = np.zeros((nx, ny, nz), dtype=float)      # Pristine Pressure
        
        # Bedrock Substrate Constraints
        self.a_max = 80.0    # Fixed structural torque ceiling
        self.dt = 0.004       # Simulation time step

        # Torus Core Geometry (Set to absolute Horn Torus Pinch: R == r)
        self.R_major = 5.5      
        self.r_minor = 5.5      
        self.spin_vtheta = 7.0  
        self.pump_vr = 5.5      # Powerful poloidal drive to test center-choke

        # Center grid coordinates around (0,0,0)
        self.X, self.Y, self.Z = np.meshgrid(
            np.arange(nx) - nx//2, 
            np.arange(ny) - ny//2, 
            np.arange(nz) - nz//2, 
            indexing='ij'
        )
        
        self.d_xy = np.sqrt(self.X**2 + self.Y**2) + 1e-9
        self.r_dist = np.sqrt((self.d_xy - self.R_major)**2 + self.Z**2)
        self.core_mask = self.r_dist <= self.r_minor

    def apply_core_vortex_drive(self):
        """ Imposes the internal helical circulation profile inside the core torus """
        v_theta_x = -self.spin_vtheta * (self.Y / self.d_xy)
        v_theta_y =  self.spin_vtheta * (self.X / self.d_xy)
        
        phi = np.arctan2(self.Z, self.d_xy - self.R_major)
        v_r_axial_z =  self.pump_vr * np.cos(phi)
        v_r_radial_x = -self.pump_vr * np.sin(phi) * (self.X / self.d_xy)
        v_r_radial_y = -self.pump_vr * np.sin(phi) * (self.Y / self.d_xy)
        
        self.vx[self.core_mask] = v_theta_x[self.core_mask] + v_r_radial_x[self.core_mask]
        self.vy[self.core_mask] = v_theta_y[self.core_mask] + v_r_radial_y[self.core_mask]
        self.vz[self.core_mask] = v_r_axial_z[self.core_mask]
        self.rho[self.core_mask] = 1.5 # Lock core baseline density

    def step_kinematic_physics(self):
        """ Solves the full 3D mass continuity and a_max vector constraints """
        old_vx, old_vy, old_vz = self.vx.copy(), self.vy.copy(), self.vz.copy()
        
        # Stencil for 6-nearest neighbor advection/diffusion mapping
        def get_neighbors_avg(field):
            left  = np.roll(field,  1, axis=0); right = np.roll(field, -1, axis=0)
            down  = np.roll(field,  1, axis=1); up    = np.roll(field, -1, axis=1)
            back  = np.roll(field,  1, axis=2); front = np.roll(field, -1, axis=2)
            return (left + right + down + up + back + front) / 6.0
            
        target_vx = get_neighbors_avg(old_vx)
        target_vy = get_neighbors_avg(old_vy)
        target_vz = get_neighbors_avg(old_vz)
        
        # Enforce the rigid a_max structural ceiling per cell
        dvx = target_vx - old_vx
        dvy = target_vy - old_vy
        dvz = target_vz - old_vz
        dv_mag = np.sqrt(dvx**2 + dvy**2 + dvz**2) + 1e-9
        
        max_dv = self.a_max * self.dt
        clamp_mask = dv_mag > max_dv
        
        dvx[clamp_mask] = (dvx[clamp_mask] / dv_mag[clamp_mask]) * max_dv
        dvy[clamp_mask] = (dvy[clamp_mask] / dv_mag[clamp_mask]) * max_dv
        dvz[clamp_mask] = (dvz[clamp_mask] / dv_mag[clamp_mask]) * max_dv
        
        # Update open substrate velocity matrix
        self.vx[~self.core_mask] = old_vx[~self.core_mask] + dvx[~self.core_mask]
        self.vy[~self.core_mask] = old_vy[~self.core_mask] + dvy[~self.core_mask]
        self.vz[~self.core_mask] = old_vz[~self.core_mask] + dvz[~self.core_mask]
        
        # Solve 3D Mass Continuity: dρ/dt = -∇·(ρv)
        flux_x = self.rho * self.vx
        flux_y = self.rho * self.vy
        flux_z = self.rho * self.vz
        
        divergence = (np.roll(flux_x, -1, axis=0) - np.roll(flux_x, 1, axis=0)) / (2 * self.dx) + \
                     (np.roll(flux_y, -1, axis=1) - np.roll(flux_y, 1, axis=1)) / (2 * self.dx) + \
                     (np.roll(flux_z, -1, axis=2) - np.roll(flux_z, 1, axis=2)) / (2 * self.dx)
                     
        self.rho[~self.core_mask] -= divergence[~self.core_mask] * self.dt
        self.rho = np.clip(self.rho, 0.8, 50.0) # Prevent coordinate drainage limits

        # Map Pristine 3D Alignment Pressure: P = ρ * v^2 * (1 - α)
        v_sq = self.vx**2 + self.vy**2 + self.vz**2
        norm_cell = np.sqrt(v_sq) + 1e-9
        norm_avg = np.sqrt(target_vx**2 + target_vy**2 + target_vz**2) + 1e-9
        
        # 3D Vector dot product yields true local directional alignment alpha
        alpha = (self.vx * target_vx + self.vy * target_vy + self.vz * target_vz) / (norm_cell * norm_avg)
        alpha = np.clip(alpha, -1.0, 1.0)
        
        self.P = self.rho * v_sq * (1.0 - alpha)

    def run_engine(self, steps=180):
        for i in range(steps):
            self.apply_core_vortex_drive()
            self.step_kinematic_physics()

    def render_rebound_dashboard(self):
        """ Cuts a vertical cross section to analyze the throat choke horizon """
        mid_y = self.ny // 2
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 7), facecolor='#111116')
        
        # PANEL 1: True Pristine Pressure Field P = ρv²(1-α)
        axs[0].set_facecolor('#111116')
        P_slice = self.P[:, mid_y, :].T
        im1 = axs[0].imshow(P_slice, origin='lower', cmap='magma',
                            extent=[-self.nx//2, self.nx//2, -self.nz//2, self.nz//2])
        fig.colorbar(im1, ax=axs[0], label=r'True Kinematic Pressure Explosion')
        
        # Overlay Streamlines showing the violent collision at the origin
        vx_slice = self.vx[:, mid_y, :].T
        vz_slice = self.vz[:, mid_y, :].T
        grid_x, grid_z = np.meshgrid(np.arange(self.nx) - self.nx//2, np.arange(self.nz) - self.nz//2)
        axs[0].streamplot(grid_x, grid_z, vx_slice, vz_slice, color='white', density=1.2, linewidth=0.7)
        
        # Outline core coordinates
        core_slice = self.core_mask[:, mid_y, :].T
        axs[0].contour(grid_x, grid_z, core_slice, levels=[0.5], colors='cyan', linestyles='--')
        
        axs[0].set_title("Vertical X-Z Slice: 3D Choked Flow Horizon\n" + 
                         r"True Kinematic Pressure Explosion ($P = \rho v^2(1-\alpha)$)", 
                         color='white', pad=15, fontweight='bold')
        axs[0].tick_params(colors='white')
        
        # PANEL 2: Substrate Mass Density Ledger (ρ)
        axs[1].set_facecolor('#111116')
        rho_slice = self.rho[:, mid_y, :].T
        im2 = axs[1].imshow(rho_slice, origin='lower', cmap='inferno',
                            extent=[-self.nx//2, self.nx//2, -self.nz//2, self.nz//2])
        fig.colorbar(im2, ax=axs[1], label=r'Mass Density Accumulation ($\rho$)')
        
        axs[1].set_title("Vertical X-Z Slice: Substrate Density Backlog\n" +
                         "The Structural Rebound Wall at the Zero-Hole Point", color='white', pad=15, fontweight='bold')
        axs[1].tick_params(colors='white')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Launching Volumetric 3D Kinematic Substrate Engine...")
    engine = DynamicSubstrateEngine3D()
    print("Executing dynamic time-step loops (Continuity + a_max constraints)...")
    engine.run_engine()
    print("Rendering Choke and Rebound Horizon Slices...")
    engine.render_rebound_dashboard()