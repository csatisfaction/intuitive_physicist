import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class StellarSqueezeSolver:
    def __init__(self, target_radius=2.0):
        self.r_target = target_radius

    def unpack_state(self, params, N):
        P = params[0:N*3].reshape(N, 3)     
        S_raw = params[N*3:N*6].reshape(N, 3) 
        S_norms = np.linalg.norm(S_raw, axis=1, keepdims=True) + 1e-9
        S = S_raw / S_norms # Enforce the absolute velocity budget limit (c)
        return P, S

    def evaluate_impedance(self, params, N):
        """ Identity-free fluid impedance: forces are purely orientation-dependent """
        P, S = self.unpack_state(params, N)
        total_impedance = 0.0

        # TERM 1: Close-Range Boundary Packing (Rule 1a)
        for i in range(N):
            distances = [np.linalg.norm(P[i] - P[j]) for j in range(N) if i != j]
            distances = np.sort(distances)
            for d in distances[:3]: 
                total_impedance += 15.0 * (d - self.r_target)**2

        # TERM 2: Long-Range Macro-Entrainment (Rule 1b - Gravity Sink)
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(P[i] - P[j])
                total_impedance -= 8.0 / (dist + 0.1)

        # TERM 3: ORIENTATION-DRIVEN COULOMB CLASH (Rule 2)
        # Equatorial charge is determined strictly by the vertical projection of the spin vector.
        q = np.abs(S[:, 2]) # 1.0 if upright, 0.0 if tilted flat into the horizon
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(P[i] - P[j])
                # Clash only ignites if BOTH axes are vertically aligned to project active equators
                total_impedance += 7.5 * (q[i] * q[j]) / (dist + 0.1)

        # TERM 4: Kinematic Co-Shear (Rule 3 - Boundary Gear-Meshing)
        for i in range(N):
            for j in range(i+1, N):
                r_vec = P[i] - P[j]
                d = np.linalg.norm(r_vec)
                if d < self.r_target * 1.5: # Contact threshold
                    r_hat = r_vec / (d + 1e-9)
                    shear_clash = np.cross(S[i] + S[j], r_hat)
                    total_impedance += 6.0 * np.linalg.norm(shear_clash)**2

        # System Anchor
        total_impedance += np.linalg.norm(np.mean(P, axis=0))**2
        return total_impedance

    def solve_autogenous_alpha(self):
        """ Stage 1: 4 identical cores find their native tetrahedral spin-balance """
        print("STAGE 1: Condensing 4 identical fluid loops into native Alpha template...")
        np.random.seed(42)
        initial_4body = np.random.uniform(-1.0, 1.0, 24)
        res = minimize(lambda x: self.evaluate_impedance(x, 4), initial_4body, method='BFGS')
        P_alpha, S_alpha = self.unpack_state(res.x, 4)
        P_alpha -= np.mean(P_alpha, axis=0)
        return P_alpha, S_alpha

    def scatter_squeezed_clones(self, P_alpha, S_alpha):
        """ Stage 2: Imposing the Stellar Furnace (Tight 3D spatial confinement) """
        print("STAGE 2: Simulating Stellar Squeeze (Random orientations, but compressed space)...")
        c12_guess_P = []
        c12_guess_S = []
        
        np.random.seed(112) # Scrambled layout seed
        
        for cluster_idx in range(3):
            # THE STELLAR SQUEEZE: Tight translation window forces perimeters to overlap
            translation = np.random.uniform(-1.1, 1.1, 3)
            
            # 100% Randomized 3D tumbles (Zero angular alignment template)
            angles = np.random.uniform(0, 2 * np.pi, 3)
            c = np.cos(angles)
            s = np.sin(angles)
            
            R_x = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
            R_y = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
            R_z = np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])
            R_3d = R_z @ R_y @ R_x
            
            for i in range(4):
                c12_guess_P.append(R_3d @ P_alpha[i] + translation)
                c12_guess_S.append(R_3d @ S_alpha[i])
                
        return np.concatenate([np.array(c12_guess_P).flatten(), np.array(c12_guess_S).flatten()])

    def solve_carbon12(self):
        P_alpha, S_alpha = self.solve_autogenous_alpha()
        squeezed_seed = self.scatter_squeezed_clones(P_alpha, S_alpha)
        
        print("STAGE 3: Unleashing identity-free 12-body relaxation under compression...")
        res = minimize(lambda x: self.evaluate_impedance(x, 12), squeezed_seed, method='BFGS', 
                       options={'maxiter': 4000, 'disp': False})
        
        P_final, S_final = self.unpack_state(res.x, 12)
        print(f"Squeeze Relaxation Complete. Final Impedance: {res.fun:.4f}")
        return P_final, S_final

# --- PLOT THE RELAXED MATRIX ---
if __name__ == "__main__":
    solver = StellarSqueezeSolver()
    P, S = solver.solve_carbon12()
    
    fig = plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111116')
    
    # Trace relaxed structural mesh lines
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(P[i] - P[j])
            if dist < 2.4:
                ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], 
                        color='gray', linestyle='-', alpha=0.5, linewidth=2)

    # DYNAMIC IDENTIFICATION: Deduced solely from final optimized spatial precession
    final_charges = np.abs(S[:, 2])
    proton_indices = np.where(final_charges >= 0.5)[0]
    neutron_indices = np.where(final_charges < 0.5)[0]
    
    if len(proton_indices) > 0:
        ax.scatter(P[proton_indices, 0], P[proton_indices, 1], P[proton_indices, 2], 
                   s=550, c='magenta', label='Emergent Protons (|Sz| >= 0.5)', depthshade=True)
    if len(neutron_indices) > 0:
        ax.scatter(P[neutron_indices, 0], P[neutron_indices, 1], P[neutron_indices, 2], 
                   s=550, c='cyan', label='Emergent Neutrons (|Sz| < 0.5)', depthshade=True)

    # Plot gyroscopic vector headings
    ax.quiver(P[:, 0], P[:, 1], P[:, 2], S[:, 0], S[:, 1], S[:, 2], length=1.1, color='lime', linewidth=2.5, label='Spin Axis')

    ax.set_title("TIR Furnace Engine: Carbon-12 ($^{12}C$)\nAutogenous Ring Assembly via Precession Under Environmental Squeeze", color='white', pad=20)
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4]); ax.axis('off')
    ax.legend(facecolor='#222233', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()