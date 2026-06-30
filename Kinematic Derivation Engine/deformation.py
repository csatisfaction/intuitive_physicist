import numpy as np

class DeformableToroid:
    def __init__(self, center, radius, vorticity_vec, num_nodes=24, minor_radius=0.1, c=1.0):
        """
        A first-principles, flexible particle core defined strictly by an array of 
        independently floating Lagrangian nodes rather than a rigid geometric formula.
        """
        self.N = num_nodes
        self.r_m = float(minor_radius)
        self.c = float(c)
        
        # 1. Establish the pristine starting coordinates of the loop nodes
        center = np.array(center, dtype=np.float64)
        omega = np.array(vorticity_vec, dtype=np.float64)
        omega_mag = np.linalg.norm(omega)
        normal = omega / omega_mag
        
        # Core Circulation Strength
        self.circulation = omega_mag * np.pi * (self.r_m ** 2)
        
        # Orthonormal orientation basis
        arb = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, arb); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        
        theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        
        # The node coordinates are now stored as a raw shape array: shape (N, 3)
        self.nodes = np.zeros((self.N, 3), dtype=np.float64)
        for i in range(self.N):
            self.nodes[i] = center + radius * (np.cos(theta[i]) * u + np.sin(theta[i]) * v)
            
        # Record the target segment length to maintain internal structural skin tension
        self.target_spacing = 2 * radius * np.sin(np.pi / self.N)

    def project_field_at_point(self, target_point):
        """
        Projects the total induced fluid velocity field vector into any arbitrary point.
        Calculates the line integral directly across the actual, deformed node segments.
        """
        x = np.array(target_point, dtype=np.float64)
        v_induced = np.zeros(3, dtype=np.float64)
        
        for i in range(self.N):
            node_current = self.nodes[i]
            node_next = self.nodes[(i + 1) % self.N]
            
            # Segment kinematics derived from the actual floating node locations
            delta_s = node_next - node_current
            midpoint = (node_current + node_next) * 0.5
            r_vec = x - midpoint
            r_mag_sq = np.dot(r_vec, r_vec)
            
            # Biot-Savart field projection smoothed by the minor core pipe radius (r_m)
            denominator = (r_mag_sq + (self.r_m ** 2)) ** 1.5
            cross_prod = np.cross(delta_s, r_vec)
            
            v_induced += (self.circulation / (4 * np.pi)) * (cross_prod / denominator)
            
        return v_induced

    def step_node_kinematics(self, external_field_func, dt, structural_stiffness=2.0):
        """
        Updates every single node independently based entirely on the local substrate velocities.
        """
        new_nodes = np.zeros_like(self.nodes)
        
        for i in range(self.N):
            curr_node = self.nodes[i]
            
            # 1. Read the local external background velocity current at this exact node coordinate
            v_ext = external_field_func(curr_node)
            
            # 2. Calculate Internal Cohesive Skin Tension (Vortex Core Stiffness 'a')
            # Since we are gridless, we use a localized neighbor-spacing restoring function
            # to simulate the centripetal boundary containment of the vortex ring
            prev_node = self.nodes[(i - 1) % self.N]
            next_node = self.nodes[(i + 1) % self.N]
            
            vec_to_prev = prev_node - curr_node
            vec_to_next = next_node - curr_node
            
            dist_prev = np.linalg.norm(vec_to_prev)
            dist_next = np.linalg.norm(vec_to_next)
            
            # Internal structural restoring forces (The elastic skin primitive)
            force_prev = structural_stiffness * (dist_prev - self.target_spacing) * (vec_to_prev / dist_prev)
            force_next = structural_stiffness * (dist_next - self.target_spacing) * (vec_to_next / dist_next)
            v_internal_restoring = force_prev + force_next
            
            # 3. Total Combined Transport Vector acting on this node
            v_total = v_ext + v_internal_restoring
            
            # Strict enforcement of the unyielding substrate saturation velocity limit (v <= c)
            v_mag = np.linalg.norm(v_total)
            if v_mag > self.c:
                v_total = self.c * (v_total / v_mag)
                
            # Advect this specific node point directly along the localized current arrow
            new_nodes[i] = curr_node + v_total * dt
            
        self.nodes = new_nodes

    def get_center_of_mass(self):
        """Calculates the emergent center of mass of the deformed ring loop."""
        return np.mean(self.nodes, axis=0)