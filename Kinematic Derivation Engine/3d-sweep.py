import numpy as np
import matplotlib.pyplot as plt

class GeometricSweepEngine3D:
    def __init__(self, nx=40, ny=40, nz=40, dx=0.25):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        
        # Bedrock Kinematic Constants
        self.a_max = 75.0
        self.dt = 0.004
        
        # Fixed Torus Profile
        self.r_minor = 5.5
        self.spin_vtheta = 6.0
        self.pump_vr = 5.0

        # Center grid around (0,0,0)
        self.X, self.Y, self.Z = np.meshgrid(
            np.arange(nx) - nx//2, 
            np.arange(ny) - ny//2, 
            np.arange(nz) - nz//2, 
            indexing='ij'
        )
        self.d_xy = np.sqrt(self.X**2 + self.Y**2) + 1e-9

    def execute_single_geometry(self, R_major):
        """ Runs a rapid fluid relaxation cycle for a specific major radius configuration """
        # Re-initialize fresh substrate fields to guarantee clean sampling
        rho = np.ones((self.nx, self.ny, self.nz), dtype=float)
        vx  = np.zeros((self.nx, self.ny, self.nz), dtype=float)
        vy  = np.zeros((self.nx, self.ny, self.nz), dtype=float)
        vz  = np.zeros((self.nx, self.ny, self.nz), dtype=float)
        P   = np.zeros((self.nx, self.ny, self.nz), dtype=float)

        # Generate the specific torus mask for this sweep increment
        r_dist = np.sqrt((self.d_xy - R_major)**2 + self.Z**2)
        core_mask = (r_dist <= self.r_minor) & (r_dist >= self.r_minor - 1.5)


        # Apply Helical Internal Vortex Drive
        v_theta_x = -self.spin_vtheta * (self.Y / self.d_xy)
        v_theta_y =  self.spin_vtheta * (self.X / self.d_xy)
        
        phi = np.arctan2(self.Z, self.d_xy - R_major)
        v_r_axial_z =  self.pump_vr * np.cos(phi)
        v_r_radial_x = -self.pump_vr * np.sin(phi) * (self.X / self.d_xy)
        v_r_radial_y = -self.pump_vr * np.sin(phi) * (self.Y / self.d_xy)
        
        vx[core_mask] = v_theta_x[core_mask] + v_r_radial_x[core_mask]
        vy[core_mask] = v_theta_y[core_mask] + v_r_radial_y[core_mask]
        vz[core_mask] = v_r_axial_z[core_mask]
        rho[core_mask] = 1.5

        # Execute 25 high-speed momentum propagation steps
        for step in range(25):
            # Re-enforce anchors
            vx[core_mask] = v_theta_x[core_mask] + v_r_radial_x[core_mask]
            vy[core_mask] = v_theta_y[core_mask] + v_r_radial_y[core_mask]
            vz[core_mask] = v_r_axial_z[core_mask]

            # 3D Neighborhood stencil relaxation
            left_vx = np.roll(vx, 1, axis=0); right_vx = np.roll(vx, -1, axis=0)
            down_vx = np.roll(vx, 1, axis=1); up_vx    = np.roll(vx, -1, axis=1)
            back_vx = np.roll(vx, 1, axis=2); front_vx = np.roll(vx, -1, axis=2)
            target_vx = (left_vx + right_vx + down_vx + up_vx + back_vx + front_vx) / 6.0

            left_vz = np.roll(vz, 1, axis=0); right_vz = np.roll(vz, -1, axis=0)
            down_vz = np.roll(vz, 1, axis=1); up_vz    = np.roll(vz, -1, axis=1)
            back_vz = np.roll(vz, 1, axis=2); front_vz = np.roll(vz, -1, axis=2)
            target_vz = (left_vz + right_vz + down_vz + up_vz + back_vz + front_vz) / 6.0

            # Acceleration delta calculation
            dvx = target_vx - vx
            dvz = target_vz - vz
            dv_mag = np.sqrt(dvx**2 + dvz**2) + 1e-9

            # Enforce strict a_max constraint ceiling
            max_dv = self.a_max * self.dt
            clamp_mask = dv_mag > max_dv
            dvx[clamp_mask] = (dvx[clamp_mask] / dv_mag[clamp_mask]) * max_dv
            dvz[clamp_mask] = (dvz[clamp_mask] / dv_mag[clamp_mask]) * max_dv

            vx[~core_mask] += dvx[~core_mask]
            vz[~core_mask] += dvz[~core_mask]

            # Solve 3D Mass Continuity divergence logs
            divergence = (np.roll(rho*vx, -1, axis=0) - np.roll(rho*vx, 1, axis=0)) / (2*self.dx) + \
                         (np.roll(rho*vz, -1, axis=2) - np.roll(rho*vz, 1, axis=2)) / (2*self.dx)
            rho[~core_mask] -= divergence[~core_mask] * self.dt
            rho = np.clip(rho, 0.5, 100.0)

        # Compute ultimate Pristine Kinematic Pressure: P = ρv²(1-α)
        v_sq = vx**2 + vy**2 + vz**2
        norm_cell = np.sqrt(v_sq) + 1e-9
        norm_avg = np.sqrt(target_vx**2 + vy**2 + target_vz**2) + 1e-9
        alpha = (vx * target_vx + vy * vy + vz * target_vz) / (norm_cell * norm_avg)
        alpha = np.clip(alpha, -1.0, 1.0)
        
        P = rho * v_sq * (1.0 - alpha)

        # Sample the exact coordinate origin node behavior (throat center core)
        center_idx = self.nx // 2
        return np.max(P[center_idx-1:center_idx+2, center_idx-1:center_idx+2, center_idx-1:center_idx+2]), \
               np.max(rho[center_idx-1:center_idx+2, center_idx-1:center_idx+2, center_idx-1:center_idx+2])

    def run_geometric_sweep(self, points=50):
        # Sweep R_major from 7.2 (Wide Open Ring) down to 4.2 (Severely Choked Core)
        R_vals = np.linspace(7.2, 4.2, points)
        gaps = R_vals - self.r_minor
        
        pressure_log = []
        density_log = []

        print(f"Beginning 50-Frame Geometric Stability Sweep across r_minor={self.r_minor}...")
        for step, R_maj in enumerate(R_vals):
            current_gap = R_maj - self.r_minor
            p_spark, rho_wall = self.execute_single_geometry(R_maj)
            pressure_log.append(p_spark)
            density_log.append(rho_wall)
            
            if step % 10 == 0:
                print(f" -> Frame {step:02d}/50 | Throat Gap: {current_gap:+.3f} | Core Pressure Spark: {p_spark:.2f}")

        return gaps, np.array(pressure_log), np.array(density_log)

# --- EXECUTE CRITICAL PLOT ENGINE ---
if __name__ == "__main__":
    sweep = GeometricSweepEngine3D()
    gap_axis, p_sparks, rho_walls = sweep.run_geometric_sweep()

    # Render Stability Curves
    fig, ax1 = plt.subplots(figsize=(12, 6.5), facecolor='#111116')
    ax1.set_facecolor('#111116')
    
    # Left Axis: Kinematic Pressure Spark Detonation Curve
    color = '#ff3366'
    ax1.plot(gap_axis, p_sparks, color=color, linewidth=3.5, label='Peak Throat Pressure ($P$)')
    ax1.set_xlabel('Geometric Throat Gap width ($R_{major} - r_{minor}$)', color='white', fontsize=12, labelpad=10)
    ax1.set_ylabel('Kinematic Pressure Spark Amplitude', color=color, fontsize=12, labelpad=10)
    ax1.tick_params(axis='y', labelcolor=color, colors='white')
    ax1.grid(color='#222233', linestyle=':', linewidth=1)

    # Right Axis: Substrate Density Backlog Wall Curve
    ax2 = ax1.twinx()
    color2 = '#00ffff'
    ax2.plot(gap_axis, rho_walls, color=color2, linewidth=3, linestyle='--', label=r'Throat Density Backlog ($\rho$)')
    ax2.set_ylabel(r'Substrate Mass Density Ledger ($\rho$)', color=color2, fontsize=12, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=color2, colors='white')

    # Add vertical landmark thresholds confirming the First-Principles boundaries
    ax1.axvline(x=0.0, color='white', linestyle='-.', alpha=0.7, linewidth=1.5)
    ax1.text(0.05, np.max(p_sparks)*0.75, 'Horn Torus Interface\n(Gap = 0)', color='white', alpha=0.8, fontweight='bold')

    # Assemble shared legend data packs cleanly avoiding string formatting errors
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, facecolor='#1b1b22', edgecolor='white', labelcolor='white', loc='upper right')

    plt.title("The Absolute Boundaries of Self-Correction\n" + 
              r"Tracking Core Pressure Detonations ($P = \rho v^2(1-\alpha)$) Under Geometric Compaction", 
              color='white', pad=20, fontsize=14, fontweight='bold')
    
    # Invert X-axis so the reading flow reads left-to-right from wide-open to crushed-choke
    ax1.set_xlim(max(gap_axis), min(gap_axis))
    ax1.tick_params(axis='x', colors='white')
    
    plt.tight_layout()
    plt.show()