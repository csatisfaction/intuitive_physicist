import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Carbon12Assembler:
    def __init__(self, num_nucleons=12, target_radius=2.0):
        self.N = num_nucleons # 6 Protons, 6 Neutrons
        self.r_target = target_radius

    def unpack_state(self, params):
        P = params[0:self.N*3].reshape(self.N, 3)     
        S_raw = params[self.N*3:self.N*6].reshape(self.N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def calculate_impedance(self, params):
        P, S = self.unpack_state(params)
        total_impedance = 0.0

        # RULE 1: Vacuum Entrainment (Local volume packing)
        # We only apply the spring penalty to the nearest neighbors to simulate short-range strong lock
        for i in range(self.N):
            distances = []
            for j in range(self.N):
                if i != j:
                    distances.append(np.linalg.norm(P[i] - P[j]))
            distances = np.sort(distances)
            # Each nucleon wants to lock with ~3-4 neighbors in a tetrahedral/ring layout
            for d in distances[:4]: 
                total_impedance += 15.0 * (d - self.r_target)**2

        # RULE 2: Coulomb Repulsion (Protons = Even indices 0,2,4,6,8,10)
        protons = [i for i in range(self.N) if i % 2 == 0]
        for i in range(len(protons)):
            for j in range(i+1, len(protons)):
                d_protons = np.linalg.norm(P[protons[i]] - P[protons[j]])
                total_impedance += 6.0 / (d_protons + 0.1)

        # RULE 3: Kinematic Co-Shear (Spin Vector Meshing)
        for i in range(self.N):
            for j in range(i+1, self.N):
                r_vec = P[i] - P[j]
                d = np.linalg.norm(r_vec)
                if d < self.r_target * 1.5: # Only touching boundaries can shear
                    r_hat = r_vec / (d + 1e-9)
                    shear_clash = np.cross(S[i] + S[j], r_hat)
                    total_impedance += 5.0 * np.linalg.norm(shear_clash)**2

        # Anti-Drift and Flattening Bias
        # The background universe exerts Z-axis pressure on massive spinning rings
        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2
        z_variance = np.var(P[:, 2])
        total_impedance += 2.0 * z_variance # Encourages finding the flat ground state

        return total_impedance

    def assemble(self):
        print("Initializing 12 chaotic fluid vortices (6 Protons, 6 Neutrons)...")
        np.random.seed(12) 
        
        # Seed them loosely in an XY ring to help the non-linear optimizer find the primary basin
        initial_guess = np.zeros(self.N * 6)
        for i in range(self.N):
            angle = i * (2 * np.pi / self.N) + np.random.uniform(-0.5, 0.5)
            rad = 3.5 + np.random.uniform(-0.5, 0.5)
            initial_guess[i*3 : i*3+3] = [rad * np.cos(angle), rad * np.sin(angle), np.random.uniform(-0.5, 0.5)]
            # Random spins
            initial_guess[self.N*3 + i*3 : self.N*3 + i*3+3] = np.random.uniform(-1, 1, 3)

        print("Executing Substrate Impedance Relaxation (Solving 72 dimensions)...")
        result = minimize(
            self.calculate_impedance, 
            initial_guess, 
            method='BFGS',
            options={'maxiter': 3000, 'disp': False}
        )
        
        P_final, S_final = self.unpack_state(result.x)
        print(f"Convergence Achieved. Final System Impedance: {result.fun:.4f}")
        return P_final, S_final

# --- EXECUTION AND 3D RENDER ---
if __name__ == "__main__":
    solver = Carbon12Assembler()
    P, S = solver.assemble()
    
    fig = plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    # Draw connections for nearest neighbors (Structural Bonds)
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(P[i] - P[j])
            if dist < 2.5: # Bonding threshold
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                        color='gray', linestyle='-', alpha=0.6, linewidth=2)

    # Separate Protons (Even) and Neutrons (Odd)
    protons = P[[i for i in range(12) if i % 2 == 0]]
    neutrons = P[[i for i in range(12) if i % 2 != 0]]
    
    ax.scatter(protons[:, 0], protons[:, 1], protons[:, 2], s=500, c='magenta', label='Protons', depthshade=True)
    ax.scatter(neutrons[:, 0], neutrons[:, 1], neutrons[:, 2], s=500, c='cyan', label='Neutrons', depthshade=True)

    # Plot Gyroscopic Spin Axes
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], 
              S[:, 0], S[:, 1], S[:, 2], 
              length=1.2, color='lime', arrow_length_ratio=0.3, linewidth=2)

    ax.set_title("TIR Engine: Autogenous Carbon-12 ($^{12}C$) Assembly\n3 Alpha-Tetrahedrons Form a Flat 2D Triangular Gear", color='white', pad=20, fontsize=14)
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4])
    ax.axis('off')
    legend = ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()