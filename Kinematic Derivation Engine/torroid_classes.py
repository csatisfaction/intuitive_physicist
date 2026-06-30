import numpy as np

class ToroidalCore:
    def __init__(self, center, major_radius, minor_radius, vorticity_vec, c=1.0):
        """
        Initializes a stable 3D Toroidal Soliton Core.
        
        Parameters:
        -----------
        center : array_like (shape: 3,)
            The 3D coordinate vector [x, y, z] of the toroid's center.
        major_radius : float (R)
            The macro-scale radius of the core's loop.
        minor_radius : float (r_minor)
            The cross-sectional radius of the vortex ring pipe (acts as core filter).
        vorticity_vec : array_like (shape: 3,)
            Vector representing the orientation of the spin axis and circulation magnitude.
        c : float
            The unyielding velocity saturation ceiling of the continuous substrate medium.
        """
        self.center = np.array(center, dtype=np.float64)
        self.R = float(major_radius)
        self.r_m = float(minor_radius)
        self.omega = np.array(vorticity_vec, dtype=np.float64)
        self.c = float(c)
        
        # Invariant Circulation Strength: Gamma = ||omega|| * pi * r_minor^2
        self.omega_mag = np.linalg.norm(self.omega)
        if self.omega_mag < 1e-14:
            raise ValueError("Vorticity vector cannot be zero; matter is a trapped kinetic state.")
            
        self.circulation = self.omega_mag * np.pi * (self.r_m ** 2)
        self.normal = self.omega / self.omega_mag
        
        # Build pristine orthonormal plane basis for the toroid ring
        # Choose arbitrary vector not collinear with normal to avoid cross-product collapse
        arbitrary_vec = np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.u_basis = np.cross(self.normal, arbitrary_vec)
        self.u_basis /= np.linalg.norm(self.u_basis)
        self.v_basis = np.cross(self.normal, self.u_basis)

    def project_velocity_field(self, target_point, num_segments=64):
        """
        Projects the induced fluid velocity field vector into any coordinate point
        in the surrounding substrate utilizing localized regularized Biot-Savart equations.
        
        Parameters:
        -----------
        target_point : array_like (shape: 3,)
            The target coordinate in the substrate to evaluate.
        num_segments : int
            Discretization resolution of the major loop filament.
            
        Returns:
        --------
        v_induced : ndarray (shape: 3,)
            The superimposed fluid velocity vector, bounded natively by ceiling (c).
        """
        x = np.array(target_point, dtype=np.float64)
        v_raw = np.zeros(3, dtype=np.float64)
        
        # Generate discrete nodes along the circular vortex filament path
        theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        
        # Vectorized coordinate computation for all nodes on the ring
        cos_t = np.cos(theta)[:, np.newaxis]
        sin_t = np.sin(theta)[:, np.newaxis]
        ring_nodes = self.center + self.R * (cos_t * self.u_basis + sin_t * self.v_basis)
        
        # Loop through segments to accumulate field vectors
        for i in range(num_segments):
            node_current = ring_nodes[i]
            node_next = ring_nodes[(i + 1) % num_segments]
            
            # Segment kinematics
            delta_s = node_next - node_current
            midpoint = (node_current + node_next) * 0.5
            r_vec = x - midpoint
            r_mag_sq = np.dot(r_vec, r_vec)
            
            # Regularized Biot-Savart projection
            # Smoothing factor is precisely the minor radius (r_m), preventing core singularities
            denominator = (r_mag_sq + (self.r_m ** 2)) ** 1.5
            cross_prod = np.cross(delta_s, r_vec)
            
            v_raw += (self.circulation / (4 * np.pi)) * (cross_prod / denominator)
            
        # Enforce the unyielding velocity saturation ceiling (c) of the medium
        v_mag = np.linalg.norm(v_raw)
        if v_mag > self.c:
            return self.c * (v_raw / v_mag)
        
        return v_raw

# Example Sandbox Initialization
if __name__ == "__main__":
    # Create a stable, high-torque particle core centered at origin
    # Spinning around the Z-axis with minor radius mapping core tightness
    particle = ToroidalCore(
        center=[0.0, 0.0, 0.0],
        major_radius=1.0,
        minor_radius=0.1,
        vorticity_vec=[0.0, 0.0, 50.0],
        c=1.0 # Saturated speed limit normalized to 1.0
    )
    
    # Evaluate field at a target point right above the donut hole
    test_point = [0.0, 0.0, 0.5]
    v_field = particle.project_velocity_field(test_point)
    
    print("=== INITIALIZING TIR FIRST-PRINCIPLES ENGINE ===")
    print(f"Toroidal Soliton Circulation Capacity (Gamma): {particle.circulation:.6f}")
    print(f"Projected Substrate Induced Velocity Vector at {test_point}: {v_field}")
    print(f"Velocity Saturation Magnitude: {np.linalg.norm(v_field):.6f} / {particle.c}")