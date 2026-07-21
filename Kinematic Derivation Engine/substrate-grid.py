import numpy as np
import matplotlib.pyplot as plt

class SubstrateFluidGrid2D:
    def __init__(self, nx=60, ny=60, dx=0.2):
        self.nx, self.ny = nx, ny
        self.dx = dx
        
        # 1. Initialize the pristine Level N-1 background matrix fields
        self.rho = np.ones((nx, ny), dtype=float)           # Uniform baseline vacuum density
        self.vx = np.zeros((nx, ny), dtype=float)           # Baseline velocity X
        self.vy = np.zeros((nx, ny), dtype=float)           # Baseline velocity Y
        self.P = np.zeros((nx, ny), dtype=float)            # Emergent pressure field
        
        # Define Bedrock Substrate Constraints
        self.a_max = 60.0    # Absolute structural acceleration tolerance of the grid cells
        self.dt = 0.005       # Tight simulation time step

        # Macro-Node Entities (The two massive spinning objects)
        self.core_radius = 2.2
        self.core_A_pos = np.array([22.0, 30.0])            # Initial pixel index coordinate A
        self.core_B_pos = np.array([38.0, 30.0])            # Initial pixel index coordinate B
        self.core_A_vel = np.array([0.0, 0.0])
        self.core_B_vel = np.array([0.0, 0.0])
        self.spin_strength = 6.5                             # Intrinsic rotational velocity drive

    def apply_vortex_boundaries(self):
        """ Forces embedded grid cells to maintain the circular vortex entrainment profile """
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        
        for center, spin in [(self.core_A_pos, self.spin_strength), (self.core_B_pos, self.spin_strength)]:
            rx = X - center[0]
            ry = Y - center[1]
            r = np.sqrt(rx**2 + ry**2) + 1e-9
            
            # Identify grid cells within the core boundary mask
            mask = r <= (self.core_radius / self.dx)
            
            # Force rigid rotational velocity vectors (Tangential spin drive)
            self.vx[mask] = -spin * (ry[mask] / r[mask])
            self.vy[mask] = spin * (rx[mask] / r[mask])
            self.rho[mask] = 1.0 # Keep core internal density packed stable

    def relax_substrate_mechanics(self):
        """ Updates the vacuum lattice using strictly local conservation laws and the a_max ceiling """
        # Copy states to calculate pure spatial updates
        old_vx, old_vy = self.vx.copy(), self.vy.copy()
        
        # Step 1: Local Momentum Advection & Neighbor Diffusion
        # Cells average out their vectors with their 4 immediate neighbors
        left_vx  = np.roll(old_vx,  1, axis=0); right_vx = np.roll(old_vx, -1, axis=0)
        down_vx  = np.roll(old_vx,  1, axis=1); up_vx    = np.roll(old_vx, -1, axis=1)
        left_vy  = np.roll(old_vy,  1, axis=0); right_vy = np.roll(old_vy, -1, axis=0)
        down_vy  = np.roll(old_vy,  1, axis=1); up_vy    = np.roll(old_vy, -1, axis=1)
        
        target_vx = (left_vx + right_vx + down_vx + up_vx) * 0.25
        target_vy = (left_vy + right_vy + down_vy + up_vy) * 0.25
        
        # Step 2: Enforce the rigid a_max structural ceiling
        # Calculate the requested delta vector
        dvx = target_vx - old_vx
        dvy = target_vy - old_vy
        dv_mag = np.sqrt(dvx**2 + dvy**2) + 1e-9
        
        # Truncate any acceleration request that breaks the substrate's torque limit
        max_dv = self.a_max * self.dt
        clamp_mask = dv_mag > max_dv
        
        dvx[clamp_mask] = (dvx[clamp_mask] / dv_mag[clamp_mask]) * max_dv
        dvy[clamp_mask] = (dvy[clamp_mask] / dv_mag[clamp_mask]) * max_dv
        
        self.vx = old_vx + dvx
        self.vy = old_vy + dvy
        
        # Step 3: Mass Continuity - Solve local density updates via velocity divergence
        # Flux = ρ * v
        flux_x = self.rho * self.vx
        flux_y = self.rho * self.vy
        
        divergence = (np.roll(flux_x, -1, axis=0) - np.roll(flux_x, 1, axis=0)) / (2 * self.dx) + \
                     (np.roll(flux_y, -1, axis=1) - np.roll(flux_y, 1, axis=1)) / (2 * self.dx)
                     
        self.rho -= divergence * self.dt
        self.rho = np.clip(self.rho, 1.0, 45.0) # Density floor pins at absolute vacuum vacuum baseline

        # Step 4: Map Pristine Kinematic Pressure: P = ρ * v^2 * (1 - α)
        v_sq = self.vx**2 + self.vy**2
        avg_vx_local = (left_vx + right_vx + down_vx + up_vx) * 0.25
        avg_vy_local = (left_vy + right_vy + down_vy + up_vy) * 0.25
        norm_avg = np.sqrt(avg_vx_local**2 + avg_vy_local**2) + 1e-9
        norm_cell = np.sqrt(v_sq) + 1e-9
        
        # Compute pure local vector alignment angle alpha
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
            
            # Identify the boundary shell region interfacing with the background substrate
            boundary_shell = (dist > (self.core_radius / self.dx)) & (dist <= (self.core_radius / self.dx) + 1.5)
            
            # The net vector push is the sum of the surrounding cell pressures directed inward
            # F = -∇P translates to summing (-Pressure * Normal Direction Vector)
            u_x = rx / dist
            u_y = ry / dist
            
            net_Fx = -np.sum(self.P[boundary_shell] * u_x[boundary_shell]) * self.dx
            net_Fy = -np.sum(self.P[boundary_shell] * u_y[boundary_shell]) * self.dx
            
            forces.append(np.array([net_Fx, net_Fy]))
            
        return forces

    def run_simulation(self, iterations=400):
        trajectory_A, trajectory_B = [], []
        
        for i in range(iterations):
            self.apply_vortex_boundaries()
            self.relax_substrate_mechanics()
            
            # Extract pure emergent forces acting on the macro-nodes
            forces = self.integrate_pressure_gradients()
            
            # Update coordinate kinematics using nothing but the grid pressure push
            self.core_A_vel += (forces[0] * 0.08) * self.dt
            self.core_B_vel += (forces[1] * 0.08) * self.dt
            
            self.core_A_pos += self.core_A_vel * self.dt
            self.core_B_pos += self.core_B_vel * self.dt
            
            # Keep cores bound safely inside the simulated container window
            self.core_A_pos = np.clip(self.core_A_pos, 5, self.nx-5)
            self.core_B_pos = np.clip(self.core_B_pos, 5, self.nx-5)
            
            trajectory_A.append(self.core_A_pos.copy())
            trajectory_B.append(self.core_B_pos.copy())
            
        return np.array(trajectory_A), np.array(trajectory_B)

# --- RUN AND RENDER THE EMERGENCE ---
if __name__ == "__main__":
    print("Launching Bottom-Up Substrate Grid Test...")
    grid = SubstrateFluidGrid2D()
    track_A, track_B = grid.run_simulation()
    
    # Render the dynamic state of space
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), facecolor='#111116')
    
    # Plot 1: The Emergent Kinematic Pressure Landscape
    axs[0].set_facecolor('#111116')
    im = axs[0].imshow(grid.P.T, origin='lower', cmap='twilight_shifted', extent=[0, grid.nx, 0, grid.ny])
    cbar = fig.colorbar(im, ax=axs[0], label='Kinematic Pressure: $\\rho v^2(1-\\alpha)$')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    # Overlay trajectories showing the inward launch
    axs[0].plot(track_A[:, 0], track_A[:, 1], color='#00ffff', linewidth=3, label='Macro Core A')
    axs[0].plot(track_B[:, 0], track_B[:, 1], color='#ff00ff', linewidth=3, label='Macro Core B')
    axs[0].scatter([track_A[0,0], track_B[0,0]], [track_A[0,1], track_B[0,1]], color='white', s=80, label='Start')
    # axs[0].set_title("Emergent Substrate Pressure Field\nMacro-Nodes Driven Solely by Grid Gradient ($\vec{F} = -\\nabla P$)", color='white', pad=15, fontweight='bold')
    axs[0].set_title("Emergent Substrate Pressure Field\n" + 
                     r"Macro-Nodes Driven Solely by Grid Gradient ($\vec{F} = -\nabla P$)", 
                     color='white', pad=15, fontweight='bold')
    axs[0].tick_params(colors='white'); axs[0].legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    
    # Plot 2: The Substrate Mass Density Backdrop
    axs[1].set_facecolor('#111116')
    im2 = axs[1].imshow(grid.rho.T, origin='lower', cmap='inferno', extent=[0, grid.nx, 0, grid.ny])
    cbar2 = fig.colorbar(im2, ax=axs[1], label='Substrate Mass Density Ledger ($\\rho$)')
    cbar2.ax.yaxis.label.set_color('white'); cbar2.ax.tick_params(labelcolor='white')
    
    axs[1].set_title("Substrate Mass Density Distribution\nChoked Flow Backlog Creating the Local Matter Floor", color='white', pad=15, fontweight='bold')
    axs[1].tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()