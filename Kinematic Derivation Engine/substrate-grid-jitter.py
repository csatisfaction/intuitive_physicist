import numpy as np
import matplotlib.pyplot as plt

class SubstrateFluidGrid2D:
    def __init__(self, nx=80, ny=80, dx=0.2):
        self.nx, self.ny = nx, ny
        self.dx = dx
        
        # 1. Initialize the background matrix fields
        self.rho = np.ones((nx, ny), dtype=float)           
        self.vx = np.random.uniform(-0.4, 0.4, (nx, ny))    # Dynamic startup noise
        self.vy = np.random.uniform(-0.4, 0.4, (nx, ny))    
        self.P = np.zeros((nx, ny), dtype=float)            
        
        # Define Bedrock Substrate Constraints
        self.a_max = 65.0    
        self.dt = 0.005       

        # Macro-Node Entities
        self.core_radius = 3.5
        self.core_A_pos = np.array([24.0, 40.0])            
        self.core_B_pos = np.array([56.0, 40.0])            
        self.core_A_vel = np.array([0.0, 0.0])
        self.core_B_vel = np.array([0.0, 0.0])
        self.spin_strength = 5.5                             

    def apply_vortex_boundaries(self):
        """ Forces embedded grid cells to maintain the circular vortex entrainment profile """
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        
        # Create a combined mask for all core positions to shield them from noise overwrite
        self.all_cores_mask = np.zeros((self.nx, self.ny), dtype=bool)
        
        for center, spin in [(self.core_A_pos, self.spin_strength), (self.core_B_pos, self.spin_strength)]:
            rx = X - center[0]
            ry = Y - center[1]
            r = np.sqrt(rx**2 + ry**2) + 1e-9
            
            mask = r <= (self.core_radius / self.dx)
            self.all_cores_mask |= mask
            
            self.vx[mask] = -spin * (ry[mask] / r[mask])
            self.vy[mask] = spin * (rx[mask] / r[mask])
            self.rho[mask] = 1.0 

    def relax_substrate_mechanics(self):
        """ Updates the vacuum lattice using local conservation laws and a persistent zero-point jitter """
        old_vx, old_vy = self.vx.copy(), self.vy.copy()
        
        # Step 1: Local Momentum Advection & Neighbor Diffusion
        left_vx  = np.roll(old_vx,  1, axis=0); right_vx = np.roll(old_vx, -1, axis=0)
        down_vx  = np.roll(old_vx,  1, axis=1); up_vx    = np.roll(old_vx, -1, axis=1)
        left_vy  = np.roll(old_vy,  1, axis=0); right_vy = np.roll(old_vy, -1, axis=0)
        down_vy  = np.roll(old_vy,  1, axis=1); up_vy    = np.roll(old_vy, -1, axis=1)
        
        target_vx = (left_vx + right_vx + down_vx + up_vx) * 0.25
        target_vy = (left_vy + right_vy + down_vy + up_vy) * 0.25
        
        # Step 2: Enforce the rigid a_max structural ceiling
        dvx = target_vx - old_vx
        dvy = target_vy - old_vy
        dv_mag = np.sqrt(dvx**2 + dvy**2) + 1e-9
        
        max_dv = self.a_max * self.dt
        clamp_mask = dv_mag > max_dv
        
        dvx[clamp_mask] = (dvx[clamp_mask] / dv_mag[clamp_mask]) * max_dv
        dvy[clamp_mask] = (dvy[clamp_mask] / dv_mag[clamp_mask]) * max_dv
        
        # Update velocities with diffused components
        self.vx = old_vx + dvx
        self.vy = old_vy + dvy
        
        # PERSISTENT FIX: Inject zero-point sub-scale jitter to open vacuum cells every frame
        # Prevents numerical low-pass diffusion from freezing out the background pressure field
        noise_mask = ~self.all_cores_mask
        self.vx[noise_mask] += np.random.uniform(-0.18, 0.18, size=self.vx[noise_mask].shape)
        self.vy[noise_mask] += np.random.uniform(-0.18, 0.18, size=self.vy[noise_mask].shape)
        
        # Step 3: Mass Continuity 
        flux_x = self.rho * self.vx
        flux_y = self.rho * self.vy
        
        divergence = (np.roll(flux_x, -1, axis=0) - np.roll(flux_x, 1, axis=0)) / (2 * self.dx) + \
                     (np.roll(flux_y, -1, axis=1) - np.roll(flux_y, 1, axis=1)) / (2 * self.dx)
                     
        self.rho -= divergence * self.dt
        self.rho = np.clip(self.rho, 1.0, 45.0) 

        # Step 4: Map Pristine Kinematic Pressure: P = ρ * v^2 * (1 - α)
        v_sq = self.vx**2 + self.vy**2
        avg_vx_local = (left_vx + right_vx + down_vx + up_vx) * 0.25
        avg_vy_local = (left_vy + right_vy + down_vy + up_vy) * 0.25
        norm_avg = np.sqrt(avg_vx_local**2 + avg_vy_local**2) + 1e-9
        norm_cell = np.sqrt(v_sq) + 1e-9
        
        alpha = (self.vx * avg_vx_local + self.vy * avg_vy_local) / (norm_cell * norm_avg)
        alpha = np.clip(alpha, -1.0, 1.0)
        
        self.P = self.rho * v_sq * (1.0 - alpha)

    def integrate_pressure_gradients(self):
        """ Evaluates the net force pushing the macro-nodes by integrating the grid's pressure field """
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        
        forces = []
        for center in [self.core_A_pos, self.core_B_pos]:
            rx = X - center[0]
            ry = Y - center[1]
            dist = np.sqrt(rx**2 + ry**2) + 1e-9
            
            boundary_shell = (dist > (self.core_radius / self.dx)) & (dist <= (self.core_radius / self.dx) + 3.5)
            
            u_x = rx / dist
            u_y = ry / dist
            
            net_Fx = -np.sum(self.P[boundary_shell] * u_x[boundary_shell]) * self.dx
            net_Fy = -np.sum(self.P[boundary_shell] * u_y[boundary_shell]) * self.dx
            
            forces.append(np.array([net_Fx, net_Fy]))
            
        return forces

    def run_simulation(self, iterations=750):
        trajectory_A, trajectory_B = [], []
        
        for i in range(iterations):
            self.apply_vortex_boundaries()
            self.relax_substrate_mechanics()
            
            forces = self.integrate_pressure_gradients()
            
            # Scaled up translation constants to amplify grid feedback response
            self.core_A_vel += (forces[0] * 0.45) * self.dt
            self.core_B_vel += (forces[1] * 0.45) * self.dt
            
            self.core_A_pos += self.core_A_vel * self.dt
            self.core_B_pos += self.core_B_vel * self.dt
            
            self.core_A_pos = np.clip(self.core_A_pos, 5, self.nx-5)
            self.core_B_pos = np.clip(self.core_B_pos, 5, self.nx-5)
            
            trajectory_A.append(self.core_A_pos.copy())
            trajectory_B.append(self.core_B_pos.copy())
            
        return np.array(trajectory_A), np.array(trajectory_B)

if __name__ == "__main__":
    print("Launching Extended Bottom-Up Substrate Grid Test...")
    grid = SubstrateFluidGrid2D()
    track_A, track_B = grid.run_simulation()
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), facecolor='#111116')
    
    # Subplot 1: Emergent Pressure Gradient Field
    axs[0].set_facecolor('#111116')
    im = axs[0].imshow(grid.P.T, origin='lower', cmap='twilight_shifted', extent=[0, grid.nx, 0, grid.ny])
    cbar = fig.colorbar(im, ax=axs[0], label=r'Kinematic Pressure: $\rho v^2(1-\alpha)$')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    axs[0].plot(track_A[:, 0], track_A[:, 1], color='#00ffff', linewidth=3.5, label='Macro Core A')
    axs[0].plot(track_B[:, 0], track_B[:, 1], color='#ff00ff', linewidth=3.5, label='Macro Core B')
    axs[0].scatter([track_A[0,0], track_B[0,0]], [track_A[0,1], track_B[0,1]], color='white', s=90, label='Start')
    
    axs[0].set_title("Emergent Substrate Pressure Field\n" + 
                     r"Macro-Nodes Driven Solely by Grid Gradient ($\vec{F} = -\nabla P$)", 
                     color='white', pad=15, fontweight='bold')
    axs[0].tick_params(colors='white'); axs[0].legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    
    # Subplot 2: Substrate Density Distribution Ledger
    axs[1].set_facecolor('#111116')
    im2 = axs[1].imshow(grid.rho.T, origin='lower', cmap='inferno', extent=[0, grid.nx, 0, grid.ny])
    cbar2 = fig.colorbar(im2, ax=axs[1], label=r'Substrate Mass Density Ledger ($\rho$)')
    cbar2.ax.yaxis.label.set_color('white'); cbar2.ax.tick_params(labelcolor='white')
    
    axs[1].set_title("Substrate Mass Density Distribution\nChoked Flow Backlog Creating the Local Matter Floor", color='white', pad=15, fontweight='bold')
    axs[1].tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()