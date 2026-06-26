import numpy as np
from scipy.optimize import basinhopping

class TIR_NucleusEngine:
    def __init__(self, num_alphas=3, num_lone_neutrons=0, num_lone_protons=0):
        self.num_alphas = num_alphas
        self.num_neutrons = num_lone_neutrons
        self.num_protons = num_lone_protons
        self.total_entities = num_alphas + num_lone_neutrons + num_lone_protons
        
        # Define scale-invariant hydrodynamic baselines
        self.R_CORE = 1.0       # Normalized core radius of a basic nucleon vortex
        self.V_SAT  = 1.0       # Saturation velocity limit of the fluid substrate (c)
        
        # Map entity types for index tracking
        self.entity_types = (['alpha'] * num_alphas + 
                             ['neutron'] * num_lone_neutrons + 
                             ['proton'] * num_lone_protons)

    def angles_to_unit_vector(self, theta, phi):
        """Remaps Pitch/Yaw angles to a flawless, non-drifting Cartesian unit vector."""
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        w = np.cos(theta)
        return np.array([u, v, w])

    def unpack_state(self, flat_state):
        """Unpacks the flat optimizer array into structured physical properties."""
        positions = np.zeros((self.total_entities, 3))
        spin_axes = np.zeros((self.total_entities, 3))
        
        idx = 0
        for i in range(self.total_entities):
            # Extract Cartesian positions
            positions[i] = flat_state[idx:idx+3]
            # Convert optimized Pitch/Yaw angles back to pristine unit spin vectors
            theta, phi = flat_state[idx+3], flat_state[idx+4]
            spin_axes[i] = self.angles_to_unit_vector(theta, phi)
            idx += 5
            
        return positions, spin_axes

    def calculate_impedance_cost(self, flat_state):
        """
        The Master Relational Cost Function. Tracks boundary layer clashes,
        core cavitation walls, and orientation-selected phase-locking.
        """
        positions, spin_axes = self.unpack_state(flat_state)
        total_system_impedance = 0.0
        
        for i in range(self.total_entities):
            type_i = self.entity_types[i]
            
            for j in range(i + 1, self.total_entities):
                type_j = self.entity_types[j]
                
                # Calculate relational displacement vector (r)
                R_ij = positions[j] - positions[i]
                distance = np.linalg.norm(R_ij) + 1e-9
                contact_dir = R_ij / distance
                
                # 1. DETERMINING THE STRUCTURAL TOUCH BOUNDARY
                # Different entity couplings yield different non-clashing spatial profiles
                if type_i == 'alpha' and type_j == 'alpha':
                    optimal_r = 2.0 * self.R_CORE
                elif type_i == 'proton' and type_j == 'neutron':
                    optimal_r = 1.0 * self.R_CORE  # Tight interlocking mesh slot
                else:
                    optimal_r = 1.5 * self.R_CORE
                
		# 2. HYDRODYNAMIC CAVITATION FLOOR (Asymmetric Distance Potential)
                # Gentle attraction if adjacent; brutal exponential repulsion if overlapping
                if distance < optimal_r:
                    # Exponential cavitation wall prevents physical core overlapping
                    spatial_clash = 100.0 * np.exp(5.0 * (optimal_r - distance))
                elif distance < 2.5 * self.R_CORE:
                    # Near-field fluid suction only operates within the immediate shell
                    spatial_clash = 1.0 * (distance - optimal_r) ** 2
                else:
                    # Distant non-adjacent nodes experience zero relational pull
                    spatial_clash = 0.0

                # 3. KINEMATIC SHEAR CALCULATION (The Vector Gear Mesh)
                # Compute velocity directions projected at the contact boundary layer
                v_i = np.cross(spin_axes[i], contact_dir)
                v_j = np.cross(spin_axes[j], -contact_dir)
                
                # Protons project dominant vertical axial exhaust vectors
                if type_i == 'proton': v_i = spin_axes[i]
                if type_j == 'proton': v_j = spin_axes[j]
                
                # Neutrons project dominant 90-degree cross-bearing streams
                if type_i == 'neutron': v_i = np.cross(spin_axes[i], contact_dir)
                
                # Delta_V measures the explicit fluid shear across the interface perimeter
                delta_v = np.linalg.norm(v_i - v_j)
                
                # 4. PRIMITIVE UNIFICATION: r = v^2 / a
                # Local acceleration report variance spikes violently with boundary shear
                local_acceleration_variance = (delta_v ** 2) + 1e-4
                
                # Total interface impedance combines the structural layout and velocity meshing
                interface_impedance = (spatial_clash * self.V_SAT) / local_acceleration_variance
                total_system_impedance += interface_impedance
                
        return total_system_impedance

    def run_simulation(self):
        """Launches the global stochastic solver to find the lowest resistance slot."""
        # 5 parameters per entity: X, Y, Z coordinates, and Theta, Phi angles
        total_params = self.total_entities * 5
        
        # Initialize elements with randomized coordinates and orientations
        np.random.seed(42)  # Fixed seed for computational reproducibility
        initial_flat_state = np.random.uniform(-2.0, 2.0, total_params)
        
        print(f"Launching TIR Global Engine for Layout: {self.entity_types}")
        print("Scouting substrate coordinates for path of absolute least resistance...")
        
        # Basin-Hopping treats the local optimizer to iterative stochastic kicks
        minimizer_kwargs = {"method": "BFGS", "options": {"gtol": 1e-5}}
        result = basinhopping(
            self.calculate_impedance_cost,
            initial_flat_state,
            niter=200,             # Number of global coordinate jolts
            T=2.0,                # Simulated coordinate noise amplitude
            minimizer_kwargs=minimizer_kwargs
        )
        
        # Extract optimized structural topology
        final_positions, final_spins = self.unpack_state(result.x)
        
        print("\n=== OPTIMIZATION MATRIX COMPLETE ===")
        print(f"Global Substrate Impedance Minimum Achieved: {result.fun:.6f}\n")
        
        # Copy-pasteable Positions Array Block
        print("# Paste your optimized data arrays here")
        print("positions = np.array([")
        for i, pos in enumerate(final_positions):
            comma = "," if i < self.total_entities - 1 else " "
            print(f"    [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]{comma}  # Entity {i} ({self.entity_types[i]})")
        print("])\n")

        # Copy-pasteable Spins Array Block
        print("spins = np.array([")
        for i, spin in enumerate(final_spins):
            comma = "," if i < self.total_entities - 1 else " "
            print(f"    [{spin[0]:6.3f}, {spin[1]:6.3f}, {spin[2]:6.3f}]{comma}  # Vector {i}")
        print("])")
            
        return final_positions, final_spins

# Example Execution: Predictive Run for Carbon-12 (3 Symmetrical Alpha Clusters)
if __name__ == "__main__":
    # To predict Oxygen-16, simply toggle num_alphas=4
    # To predict Helium-3, toggle num_alphas=0, num_lone_neutrons=1, num_lone_protons=2
    solver = TIR_NucleusEngine(num_alphas=6, num_lone_neutrons=0, num_lone_protons=0)
    final_positions, final_spins = solver.run_simulation()
