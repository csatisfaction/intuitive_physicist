import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class TripleAlphaPipelineSolver:
    def __init__(self, target_radius=2.0):
        self.r_target = target_radius

    def unpack_state(self, params, N):
        P = params[0:N*3].reshape(N, 3)     
        S_raw = params[N*3:N*6].reshape(N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def evaluate_impedance(self, params, N):
        P, S = self.unpack_state(params, N)
        total_impedance = 0.0

        # RULE 1: Volume Packing
        for i in range(N):
            distances = [np.linalg.norm(P[i] - P[j]) for j in range(N) if i != j]
            distances = np.sort(distances)
            for d in distances[:3]: 
                total_impedance += 15.0 * (d - self.r_target)**2

        # RULE 2: Global Macro-Entrainment (Gravity)
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(P[i] - P[j])
                total_impedance -= 8.0 / (dist + 0.1)

        # RULE 3: Emergent Coulomb Repulsion (No hardcoded identity!)
        q = np.abs(S[:, 2]) # Proton-ness is an emergent orientation
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(P[i] - P[j])
                total_impedance += 7.5 * (q[i] * q[j]) / (dist + 0.1)

        # RULE 4: Kinematic Co-Shear (Gear Meshing)
        for i in range(N):
            for j in range(i+1, N):
                r_vec = P[i] - P[j]
                d = np.linalg.norm(r_vec)
                if d < self.r_target * 1.5:
                    r_hat = r_vec / (d + 1e-9)
                    shear_clash = np.cross(S[i] + S[j], r_hat)
                    total_impedance += 6.0 * np.linalg.norm(shear_clash)**2

        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2
        return total_impedance

    def run_pipeline(self):
        # 1. Generate pristine Alpha
        np.random.seed(42)
        initial_4 = np.random.uniform(-1.0, 1.0, 24)
        res_alpha = minimize(lambda x: self.evaluate_impedance(x, 4), initial_4, method='BFGS')
        P_alpha, S_alpha = self.unpack_state(res_alpha.x, 4)
        
        # 2. Bind Beryllium-8 (The Unstable Intermediate)
        print("PHASE B: Testing Beryllium-8 (2 Alphas)...")
        # Start with two Alphas near each other but chaotic
        P_8 = np.vstack([P_alpha, P_alpha + np.random.uniform(-1, 1, (4,3))])
        S_8 = np.vstack([S_alpha, S_alpha])
        seed_8 = np.concatenate([P_8.flatten(), S_8.flatten()])
        res_be = minimize(lambda x: self.evaluate_impedance(x, 8), seed_8, method='BFGS')
        print(f"Beryllium Stress (High impedance expected): {res_be.fun:.2f}")

        # 3. Inject 3rd Alpha (The Carbon Flash)
        print("PHASE C: Injecting 3rd Alpha for Carbon-12 assembly...")
        P_8_opt, S_8_opt = self.unpack_state(res_be.x, 8)
        P_12 = np.vstack([P_8_opt, P_alpha + np.random.uniform(-1, 1, (4,3))])
        S_12 = np.vstack([S_8_opt, S_alpha])
        seed_12 = np.concatenate([P_12.flatten(), S_12.flatten()])
        
        res_c12 = minimize(lambda x: self.evaluate_impedance(x, 12), seed_12, method='BFGS', options={'maxiter': 5000})
        
        return self.unpack_state(res_c12.x, 12)

# --- EXECUTE ---
if __name__ == "__main__":
    solver = TripleAlphaPipelineSolver()
    P, S = solver.run_pipeline()
    
    fig = plt.figure(figsize=(10, 8), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    # Plot final structure
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(P[i] - P[j]) < 2.5:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], color='gray', alpha=0.4)
                
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=400, c=np.abs(S[:, 2]), cmap='coolwarm')
    ax.axis('off')
    plt.show()