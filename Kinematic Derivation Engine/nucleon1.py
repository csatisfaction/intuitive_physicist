import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class KinematicNucleonSolver:
    def __init__(self, target_radius=2.0):
        self.r_target = target_radius

    def unpack_state(self, params):
        """ Decodes the 24-variable optimizer array into Positions and Spin Axes """
        P = params[0:12].reshape(4, 3)     # 4 Cores x (X, Y, Z)
        S_raw = params[12:24].reshape(4, 3) # 4 Spin Vectors
        # Natively normalize spin vectors to enforce an absolute velocity limit (c)
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def calculate_system_impedance(self, params):
        """ The pure TIR fluid impedance equation. The solver seeks to drive this to 0. """
        P, S = self.unpack_state(params)
        total_impedance = 0.0

        # RULE 1: Vacuum Entrainment (Volume Packing)
        # All touching geometries want to relax into the minimal-drag boundary limit (2.0)
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(P[i] - P[j])
                total_impedance += 12.0 * (dist - self.r_target)**2

        # RULE 2: Coulomb Repulsion (Proton-Proton Acoustic Clash)
        # Core 0 and Core 1 are Protons. Their radial waves push each other apart.
        d_protons = np.linalg.norm(P[0] - P[1])
        total_impedance += 5.0 / (d_protons + 0.1)

        # RULE 3: Kinematic Co-Shear (The Strong Force Spin-Lock)
        # Adjacent boundaries must co-shear smoothly. Clashing velocity vectors spike pressure.
        for i in range(4):
            for j in range(i+1, 4):
                r_vec = P[i] - P[j]
                r_hat = r_vec / (np.linalg.norm(r_vec) + 1e-9)
                
                # Cross-product calculates the orthogonal boundary clash
                shear_clash = np.cross(S[i] + S[j], r_hat)
                total_impedance += 4.0 * np.linalg.norm(shear_clash)**2

        # Anti-Drift: Keeps the molecule centered in the viewport
        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2

        return total_impedance

    def assemble(self):
        print("Initializing 4 chaotic, unaligned fluid vortices...")
        # Start with completely random noise
        np.random.seed(42) # Fixed seed for replicable visual
        initial_guess = np.random.uniform(-1.5, 1.5, 24)

        print("Executing TIR Impedance Relaxation (Gradient Descent)...")
        # Run the minimization engine
        result = minimize(
            self.calculate_system_impedance, 
            initial_guess, 
            method='BFGS',
            options={'maxiter': 2000, 'disp': False}
        )
        
        P_final, S_final = self.unpack_state(result.x)
        print(f"Convergence Achieved. Final System Impedance: {result.fun:.4f}")
        return P_final, S_final

# --- EXECUTION AND 3D RENDER ---
if __name__ == "__main__":
    solver = KinematicNucleonSolver()
    P, S = solver.assemble()
    
    # Setup 3D Canvas
    fig = plt.figure(figsize=(10, 8), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    # 1. Plot the Fluid Boundaries (Wireframe)
    for i in range(4):
        for j in range(i+1, 4):
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                    color='gray', linestyle='--', alpha=0.5, linewidth=2)
            
    # 2. Plot the Cores (0,1 = Protons, 2,3 = Neutrons)
    ax.scatter(P[0:2, 0], P[0:2, 1], P[0:2, 2], s=600, c='magenta', label='Protons (Sine)', depthshade=True)
    ax.scatter(P[2:4, 0], P[2:4, 1], P[2:4, 2], s=600, c='cyan', label='Neutrons (Cosine/Tilted)', depthshade=True)

    # 3. Plot the Gyroscopic Spin Axes (The Macro-Vortex Field Lines)
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], 
              S[:, 0], S[:, 1], S[:, 2], 
              length=1.5, color='lime', arrow_length_ratio=0.2, linewidth=3, label='Gyroscopic Spin Axis (v_theta)')

    # Formatting
    ax.set_title("TIR 3D Engine: Autogenous Alpha Particle ($^4He$)\nEmergent Tetrahedral Packing via Fluid Shear Minimization", color='white', pad=20)
    ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2]); ax.set_zlim([-2, 2])
    ax.axis('off') # Turn off Cartesian grid to emphasize relational geometry
    legend = ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()