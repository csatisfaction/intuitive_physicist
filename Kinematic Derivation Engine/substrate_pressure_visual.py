import numpy as np
import matplotlib.pyplot as plt

class TIRPressureEngine:
    def __init__(self, center, major_radius, minor_radius, vorticity_vec, pulse_amp=0.4, freq=15.0, phase=0.0, c=1.0):
        """
        First-Principles Substrate Evaluator computing the spatial Pressure Floor (P).
        """
        self.center = np.array(center, dtype=np.float64)
        self.R = float(major_radius)
        self.r_m = float(minor_radius)
        self.omega = np.array(vorticity_vec, dtype=np.float64)
        self.c = float(c)
        
        # Wave Primitives
        self.pulse_amp = float(pulse_amp)
        self.freq = float(freq)
        self.phase = float(phase)
        
        self.omega_mag = np.linalg.norm(self.omega)
        self.circulation = self.omega_mag * np.pi * (self.r_m ** 2)
        
        # Build pristine orthonormal plane basis for the toroid ring
        arbitrary_vec = np.array([0.0, 0.0, 1.0]) if abs(self.omega[2]/self.omega_mag) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.u_basis = np.cross(self.omega/self.omega_mag, arbitrary_vec)
        self.u_basis /= np.linalg.norm(self.u_basis)
        self.v_basis = np.cross(self.omega/self.omega_mag, self.u_basis)

    def evaluate_pressure_grid_xz(self, x_range, z_range, t=0.0, num_segments=48):
        """
        Evaluates the superimposed velocity fields across a dense X-Z mesh grid 
        and converts them directly into the localized Substrate Pressure Floor (P).
        """
        X, Z = np.meshgrid(x_range, z_range)
        Y = np.zeros_like(X) # Fixed plane slice at Y=0
        
        grid_shape = X.shape
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        
        # Initialize flat array for accumulated fluid velocities: shape (M, 3)
        v_total = np.zeros_like(coords)
        
        # Generate discrete nodes along the circular major ring path
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        ring_nodes = self.center + self.R * (cos_t * self.u_basis + sin_t * self.v_basis)
        
        # Process the fields across all grid points using broadcasting
        for i in range(num_segments):
            node_current = ring_nodes[i]
            node_next = ring_nodes[(i + 1) % num_segments]
            
            delta_s = node_next - node_current
            midpoint = (node_current + node_next) * 0.5
            
            # Vectors from segment midpoint to ALL grid points
            r_vecs = coords - midpoint
            r_mags = np.linalg.norm(r_vecs, axis=-1)
            
            # Mask to prevent division-by-zero right on the line filament
            safe_mask = r_mags > 1e-10
            
            if np.any(safe_mask):
                # 1. Steady-state DC Circulation (Biot-Savart regularized by minor radius r_m)
                denominator_dc = (r_mags[safe_mask]**2 + (self.r_m ** 2)) ** 1.5
                cross_prod = np.cross(delta_s, r_vecs[safe_mask])
                v_dc = (self.circulation / (4 * np.pi)) * (cross_prod / denominator_dc[:, np.newaxis])
                
                # 2. Symmetrical AC Phase Wave Emission (Radial Breathing)
                time_of_flight_lag = r_mags[safe_mask] / self.c
                wave_term = np.cos(self.freq * (t - time_of_flight_lag) + self.phase)
                unit_directions = r_vecs[safe_mask] / r_mags[safe_mask, np.newaxis]
                v_ac = self.pulse_amp * wave_term[:, np.newaxis] * unit_directions
                
                v_total[safe_mask] += v_dc + v_ac
                
        # Enforce the unyielding velocity saturation ceiling constraint (v <= c)
        v_mags_final = np.linalg.norm(v_total, axis=-1)
        saturation_mask = v_mags_final > self.c
        if np.any(saturation_mask):
            v_total[saturation_mask] = (v_total[saturation_mask] / v_mags_final[saturation_mask, np.newaxis]) * self.c
            v_mags_final[saturation_mask] = self.c
            
        # NATIVE BERNOULLI MAP: Pressure P = 1.0 - v^2
        # High fluid speed creates a local vacuum drop; saturation ceiling sets the hard floor
        P_flat = 1.0 - (v_mags_final ** 2)
        
        return X, Z, P_flat.reshape(grid_shape)

    def render_pressure_map(self, x_bounds=(-3.0, 3.0), z_bounds=(-3.0, 3.0), res=150):
        x_space = np.linspace(x_bounds[0], x_bounds[1], res)
        z_space = np.linspace(z_bounds[0], z_bounds[1], res)
        
        X, Z, P = self.evaluate_pressure_grid_xz(x_space, z_space, t=0.0)
        
        plt.figure(figsize=(11, 8), facecolor='#111116')
        ax = plt.gca()
        ax.set_facecolor('#111116')
        
        # Color contour representing the spatial pressure landscape
        # 'viridis' or 'coolwarm' maps this beautifully; viridis shows vacuums as deep purple/blue
        contour = plt.contourf(X, Z, P, levels=100, cmap='viridis')
        cbar = plt.colorbar(contour, label='Substrate Pressure Value (P)')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(labelcolor='white')
        
        # Plot the geometric markers of the Horn Torus core sections
        core_left = plt.Circle((-self.R, 0), self.r_m, color='white', fill=False, linewidth=1.5, linestyle=':')
        core_right = plt.Circle((self.R, 0), self.r_m, color='white', fill=False, linewidth=1.5, linestyle=':')
        ax.add_patch(core_left)
        ax.add_patch(core_right)
        
        plt.title('TIR Substrate Pressure Floor Topography P = 1.0 - v^2 (Snapshot t=0)', color='white', fontsize=14, pad=15)
        plt.xlabel('Substrate Axis: X', color='white', fontsize=12)
        plt.ylabel('Spin Orientation Axis: Z', color='white', fontsize=12)
        
        plt.grid(color='#222233', linestyle=':', linewidth=0.5)
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        plt.xlim(x_bounds)
        plt.ylim(z_bounds)
        plt.show()

# Firing up the visualizer
if __name__ == "__main__":
    visualiser = TIRPressureEngine(
        center=[0.0, 0.0, 0.0],
        major_radius=1.0,          # Loop size
        minor_radius=0.15,         # Pipe size
        vorticity_vec=[0.0, 0.0, 35.0], # Spinning around Z
        pulse_amp=0.25,            # Active AC breathing strength
        freq=12.0,                 # Wave frequency
        c=1.0
    )
    visualiser.render_pressure_map()