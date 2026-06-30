import numpy as np
import matplotlib.pyplot as plt

class TIRVisualizerEngine:
    def __init__(self, center, major_radius, minor_radius, vorticity_vec, c=1.0):
        """
        First-Principles Hydrodynamic Field Projection Engine with Vectorized Grid Capabilities.
        """
        self.center = np.array(center, dtype=np.float64)
        self.R = float(major_radius)
        self.r_m = float(minor_radius)
        self.omega = np.array(vorticity_vec, dtype=np.float64)
        self.c = float(c)
        
        self.omega_mag = np.linalg.norm(self.omega)
        if self.omega_mag < 1e-14:
            raise ValueError("Vorticity vector cannot be zero; matter requires trapped kinetic rotation.")
            
        self.circulation = self.omega_mag * np.pi * (self.r_m ** 2)
        self.normal = self.omega / self.omega_mag
        
        # Pristine orthonormal plane basis generation
        arbitrary_vec = np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.u_basis = np.cross(self.normal, arbitrary_vec)
        self.u_basis /= np.linalg.norm(self.u_basis)
        self.v_basis = np.cross(self.normal, self.u_basis)

    def project_grid_xz(self, x_range, z_range, num_segments=64):
        """
        Projects the full 3D velocity field across a dense 2D X-Z slice mesh grid (at Y=0).
        Utilizes highly optimized NumPy broadcasting over all coordinates.
        """
        # Create the 2D coordinate substrate mesh
        X, Z = np.meshgrid(x_range, z_range)
        Y = np.zeros_like(X) # Fixed slice plane at Y = 0
        
        # Flatten mesh grid into a single listing of 3D target coordinates: shape (M, 3)
        grid_shape = X.shape
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        
        # Initialize raw velocity accumulation array for the substrate points
        v_raw = np.zeros_like(coords)
        
        # Discretize the major ring filament path into loop nodes
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        ring_nodes = self.center + self.R * (cos_t * self.u_basis + sin_t * self.v_basis)
        
        # Vectorized line-integral calculation across all loop segments
        for i in range(num_segments):
            node_current = ring_nodes[i]
            node_next = ring_nodes[(i + 1) % num_segments]
            
            # Segment metrics
            delta_s = node_next - node_current
            midpoint = (node_current + node_next) * 0.5
            
            # Relative vectors from this segment midpoint to ALL grid coordinates: shape (M, 3)
            r_vecs = coords - midpoint
            r_mag_sq = np.sum(r_vecs**2, axis=-1)
            
            # Regularized denominator utilizing minor radius (r_m) to eliminate core singularities
            denominator = (r_mag_sq + (self.r_m ** 2)) ** 1.5
            denominator = denominator[:, np.newaxis] # Reshape for broadcasting
            
            # Cross product of segment direction with relative position vectors
            cross_prod = np.cross(delta_s, r_vecs)
            
            # Accumulate vector fields
            v_raw += (self.circulation / (4 * np.pi)) * (cross_prod / denominator)
            
        # Calculate velocity magnitudes
        v_mags = np.linalg.norm(v_raw, axis=-1)
        
        # Apply the unyielding velocity saturation ceiling (c)
        saturation_mask = v_mags > self.c
        if np.any(saturation_mask):
            # Enforce hard upper bound on throughput speed
            v_raw[saturation_mask] = (v_raw[saturation_mask] / v_mags[saturation_mask, np.newaxis]) * self.c
            v_mags[saturation_mask] = self.c
            
        # Reshape flat array data back into original 2D grid configurations
        Vx = v_raw[:, 0].reshape(grid_shape)
        Vz = v_raw[:, 2].reshape(grid_shape)
        Vm = v_mags.reshape(grid_shape)
        
        return X, Z, Vx, Vz, Vm

    def render_slice(self, x_bounds=(-2.5, 2.5), z_bounds=(-2.5, 2.5), res=100):
        """
        Evaluates the mesh grid and draws a beautiful hydrodynamic field map.
        """
        x_space = np.linspace(x_bounds[0], x_bounds[1], res)
        z_space = np.linspace(z_bounds[0], z_bounds[1], res)
        
        # Extract fields from the math engine
        X, Z, Vx, Vz, Vm = self.project_grid_xz(x_space, z_space)
        
        # Set up plot metrics
        plt.figure(figsize=(10, 8), facecolor='#111116')
        ax = plt.gca()
        ax.set_facecolor('#111116')
        
        # 1. Background color-contour representing absolute fluid velocity magnitudes
        contour = plt.contourf(X, Z, Vm, levels=100, cmap='inferno', alpha=0.85)
        cbar = plt.colorbar(contour, label='Induced Fluid Velocity Magnitude (v)')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(labelcolor='white')
        
        # 2. Fluid flow streamlines mapping the actual transport tracks of the space substrate
        # Transparency (alpha=0.6) is passed cleanly via the 4th element of the RGBA tuple
        plt.streamplot(X, Z, Vx, Vz, color=(0.0, 1.0, 1.0, 0.6), linewidth=0.8, density=1.5, arrowsize=1.0)
        
        # 3. Geometric markers illustrating where the physical vortex tube core cross-sections rest
        # Left tube boundary at X = -R, Right tube boundary at X = +R
        core_left = plt.Circle((-self.R, 0), self.r_m, color='cyan', fill=False, linewidth=2.0, linestyle='--')
        core_right = plt.Circle((self.R, 0), self.r_m, color='cyan', fill=False, linewidth=2.0, linestyle='--')
        ax.add_patch(core_left)
        ax.add_patch(core_right)
        
        # Labeling and clean aesthetics
        plt.title('TIR Engine: Hydrodynamic Substrate Field Projection Slice (Y=0)', color='white', fontsize=14, pad=15)
        plt.xlabel('Substrate Space Axis: X', color='white', fontsize=12)
        plt.ylabel('Substrate Space Axis: Z (Spin Orientation Axis)', color='white', fontsize=12)
        
        plt.grid(color='#222233', linestyle=':', linewidth=0.5)
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        plt.xlim(x_bounds)
        plt.ylim(z_bounds)
        
        print(f"=== RE-RENDERING COMPLETED ===")
        print(f"Max velocity tracked on slice mesh grid: {np.max(Vm):.6f} / {self.c}")
        plt.show()

# Run Engine and Plot Results
if __name__ == "__main__":

    # Proton-like: Compact, ultra-dense, hyper-torqued
    engine = TIRVisualizerEngine(
        center=[0.0, 0.0, 0.0],
        major_radius=0.4,       # Tight macro boundary
        minor_radius=0.05,      # Extremely narrow, choked core pipe
        vorticity_vec=[0.0, 0.0, 150.0], # Massive spin rate / high torque
        c=1.0
    )
    # Render with a tighter frame to inspect the dense nucleus
    engine.render_slice(x_bounds=(-1.5, 1.5), z_bounds=(-1.5, 1.5), res=150)
    
