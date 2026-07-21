import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PhaseTransitionNuclearSolver:
    def __init__(self, target_radius=2.0):
        self.r_target = target_radius

    def unpack_state(self, params, N):
        P = params[0:N*3].reshape(N, 3)     
        S_raw = params[N*3:N*6].reshape(N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms
        return P, S

    def evaluate_impedance(self, params, N):
        """ Pristine, identity-free TIR impedance function """
        P, S = self.unpack_state(params, N)
        total_impedance = 0.0

        # 1. Near-Field Packing (Rule 1a)
        for i in range(N):
            distances = [np.linalg.norm(P[i] - P[j]) for j in range(N) if i != j]
            distances = np.sort(distances)
            for d in distances[:3]: 
                total_impedance += 15.0 * (d - self.r_target)**2

        # 2. Universal Macro-Entrainment (Rule 1b - Gravity)
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(P[i] - P[j])
                total_impedance -= 8.0 / (dist + 0.1)

        # 3. Real-Time Vector Charge Repulsion (Rule 2)
        q = np.abs(S[:, 2]) # Active charge is the vertical projection component
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(P[i] - P[j])
                total_impedance += 7.5 * (q[i] * q[j]) / (dist + 0.1)

        # 4. Tangential Co-Shear (Rule 3 - Boundary Gear-Meshing)
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

    def solve_alpha_template(self):
        """ Stage 1: Autogenous single Alpha generation """
        np.random.seed(42)
        initial_4body = np.random.uniform(-1.0, 1.0, 24)
        res = minimize(lambda x: self.evaluate_impedance(x, 4), initial_4body, method='BFGS')
        P_alpha, S_alpha = self.unpack_state(res.x, 4)
        P_alpha -= np.mean(P_alpha, axis=0)
        return P_alpha, S_alpha

    def macro_rigid_impedance(self, macro_params, P_template, S_template):
        """ Calculates system stress while treating the clusters as locked rigid blocks """
        # macro_params: 3 clusters * 6 degrees of freedom (tx, ty, tz, rx, ry, rz) = 18 variables
        P_current = []
        S_current = []
        
        for idx in range(3):
            t = macro_params[idx*6 : idx*6+3]
            rot = macro_params[idx*6+3 : idx*6+6]
            
            # Reconstruct rotation matrix
            c, s = np.cos(rot), np.sin(rot)
            Rx = np.array([[1,0,0], [0,c[0],-s[0]], [0,s[0],c[0]]])
            Ry = np.array([[c[1],0,s[1]], [0,1,0], [-s[1],0,c[1]]])
            Rz = np.array([[c[2],-s[2],0], [s[2],c[2],0], [0,0,1]])
            R = Rz @ Ry @ Rx
            
            for i in range(4):
                P_current.append(R @ P_template[i] + t)
                S_current.append(R @ S_template[i])
                
        # Flatten into standard array to evaluate via core metric
        flat_state = np.concatenate([np.array(P_current).flatten(), np.array(S_current).flatten()])
        return self.evaluate_impedance(flat_state, 12)

    def pipeline_solve(self):
        # 1. Natively condense a template alpha from scratch
        P_alpha, S_alpha = self.solve_alpha_template()
        
        # 2. Scatter the clusters completely blindly in an open 3D bubble
        print("STAGE 2: Blindingly scattering rigid macro-blocks into stellar core...")
        np.random.seed(117)
        initial_macro = np.random.uniform(-2.5, 2.5, 18) 
        
        # Optimize ONLY the rigid macro positions and orientations first (Phase 2 alignment)
        print("STAGE 3: Orienting macro-gears to clear the Coulomb barrier corridor...")
        res_macro = minimize(lambda x: self.macro_rigid_impedance(x, P_alpha, S_alpha), 
                             initial_macro, method='BFGS', options={'maxiter': 1000})
        
        # Reconstruct the aligned coordinates to seed the final unconstrained run
        final_guess_P = []
        final_guess_S = []
        for idx in range(3):
            t = res_macro.x[idx*6 : idx*6+3]
            rot = res_macro.x[idx*6+3 : idx*6+6]
            c, s = np.cos(rot), np.sin(rot)
            R = np.array([[c[2],-s[2],0], [s[2],c[2],0], [0,0,1]]) @ \
                np.array([[c[1],0,s[1]], [0,1,0], [-s[1],0,c[1]]]) @ \
                np.array([[1,0,0], [0,c[0],-s[0]], [0,s[0],c[0]]])
            for i in range(4):
                final_guess_P.append(R @ P_alpha[i] + t)
                final_guess_S.append(R @ S_alpha[i])
        
        individual_seed = np.concatenate([np.array(final_guess_P).flatten(), np.array(final_guess_S).flatten()])
        
        # 3. Global un-patched 12-body relaxation (Baryons fully free to precess internally)
        print("STAGE 4: Unleashing final unconstrained identity-free 12-body relaxation...")
        res_final = minimize(lambda x: self.evaluate_impedance(x, 12), individual_seed, method='BFGS',
                             options={'maxiter': 3000, 'disp': False})
        
        P_final, S_final = self.unpack_state(res_final.x, 12)
        print(f"Manifold Convergence Achieved. Final Stress: {res_final.fun:.4f}")
        return P_final, S_final

# --- PLOT THE REALIZED MANIFOLD ---
if __name__ == "__main__":
    solver = PhaseTransitionNuclearSolver()
    P, S = solver.pipeline_solve()
    
    fig = plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(P[i] - P[j])
            if dist < 2.4:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                        color='gray', linestyle='-', alpha=0.5, linewidth=2)

    final_charges = np.abs(S[:, 2])
    protons = np.where(final_charges >= 0.5)[0]
    neutrons = np.where(final_charges < 0.5)[0]
    
    if len(protons) > 0:
        ax.scatter(P[protons, 0], P[protons, 1], P[protons, 2], s=550, c='magenta', label='Emergent Protons', depthshade=True)
    if len(neutrons) > 0:
        ax.scatter(P[neutrons, 0], P[neutrons, 1], P[neutrons, 2], s=550, c='cyan', label='Emergent Neutrons', depthshade=True)

    ax.quiver(P[:, 0], P[:, 1], P[:, 2], S[:, 0], S[:, 1], S[:, 2], length=1.1, color='lime', linewidth=2.5)
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4]); ax.axis('off')
    ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()