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
        # ... [Same impedance logic as before] ...
        # [I will re-include the full logic below to ensure you have the complete updated script]
        
        # Packing
        for i in range(N):
            distances = [np.linalg.norm(P[i] - P[j]) for j in range(N) if i != j]
            distances = np.sort(distances)
            for d in distances[:3]: 
                total_impedance += 15.0 * (d - self.r_target)**2
        # Gravity
        for i in range(N):
            for j in range(i+1, N):
                total_impedance -= 8.0 / (np.linalg.norm(P[i] - P[j]) + 0.1)
        # Charge
        q = np.abs(S[:, 2])
        for i in range(N):
            for j in range(i+1, N):
                total_impedance += 7.5 * (q[i] * q[j]) / (np.linalg.norm(P[i] - P[j]) + 0.1)
        # Shear
        for i in range(N):
            for j in range(i+1, N):
                r_vec = P[i] - P[j]
                d = np.linalg.norm(r_vec)
                if d < self.r_target * 1.5:
                    total_impedance += 6.0 * np.linalg.norm(np.cross(S[i] + S[j], r_vec/d))**2
        
        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2
        return total_impedance

    def run_pipeline(self):
        # 1. Alpha
        np.random.seed(42)
        res_alpha = minimize(lambda x: self.evaluate_impedance(x, 4), np.random.uniform(-1, 1, 24), method='BFGS')
        P_a, S_a = self.unpack_state(res_alpha.x, 4)
        
        # 2. Beryllium (2 Alphas)
        P_be = np.vstack([P_a, P_a + np.random.uniform(-1, 1, (4,3))])
        S_be = np.vstack([S_a, S_a])
        res_be = minimize(lambda x: self.evaluate_impedance(x, 8), np.concatenate([P_be.flatten(), S_be.flatten()]), method='BFGS')
        print(f"Beryllium Stress: {res_be.fun:.4f}")

        # 3. Carbon (3 Alphas) - TIGHTER INITIALIZATION
        print("PHASE C: Injecting 3rd Alpha...")
        P_c12 = np.vstack([P_be, P_a + np.random.uniform(-1.5, 1.5, (4,3))])
        S_c12 = np.vstack([S_be, S_a])
        res_c12 = minimize(lambda x: self.evaluate_impedance(x, 12), np.concatenate([P_c12.flatten(), S_c12.flatten()]), method='BFGS', options={'maxiter': 5000})
        print(f"Carbon-12 Final Stress: {res_c12.fun:.4f}")
        
        return self.unpack_state(res_c12.x, 12)

# --- EXECUTE ---
if __name__ == "__main__":
    solver = TripleAlphaPipelineSolver()
    P, S = solver.run_pipeline()
    # ... [Visualization code] ...