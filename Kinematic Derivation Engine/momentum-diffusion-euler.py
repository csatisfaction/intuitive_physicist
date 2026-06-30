import numpy as np
import matplotlib.pyplot as plt

class ChargeInteractionFabric:
    def __init__(self, x_bounds=(-3.0, 3.0), y_bounds=(-3.0, 3.0), resolution=60, c=1.0):
        self.res = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # The Active Vector Substrate Plane
        self.VX = np.zeros_like(self.X)
        self.VY = np.zeros_like(self.Y)
        
        # Boundary tracking mask to shield particle tracks from smoothing out
        self.boundary_mask = np.zeros_like(self.X, dtype=bool)

    def embed_proton_core(self, cx=0.0, cy=0.0, radius=0.4, spin_v=0.9, pulse_phase=0.0):
        """
        Embeds a high-torque, tight-radius proton anchor. 
        Pumps fluid with baseline zero-phase parameters.
        """
        dx = self.X - cx
        dy = self.Y - cy
        r = np.sqrt(dx**2 + dy**2)
        
        # Identify grid cells sitting directly on the proton's narrow track
        mask = np.abs(r - radius) < 0.12
        self.boundary_mask[mask] = True
        
        angle = np.arctan2(dy, dx)
        # DC Spin: Counter-Clockwise
        self.VX[mask] = -spin_v * np.sin(angle)[mask]
        self.VY[mask] = spin_v * np.cos(angle)[mask]
        
        # AC Breath: Standard Outward Phase
        self.VX[mask] += 0.2 * np.cos(pulse_phase) * np.cos(angle)[mask]
        self.VY[mask] += 0.2 * np.cos(pulse_phase) * np.sin(angle)[mask]

    def embed_electron_core(self, cx=1.6, cy=0.0, radius=0.9, spin_v=0.7, pulse_phase=0.0):
        """
        Embeds a relaxed, wide-radius open electron disk.
        Driven 180 degrees out-of-phase (pulse_phase + pi) to act as a hydraulic drain.
        """
        dx = self.X - cx
        dy = self.Y - cy
        r = np.sqrt(dx**2 + dy**2)
        
        # Wide, relaxed toroidal track layout
        mask = np.abs(r - radius) < 0.15
        self.boundary_mask[mask] = True
        
        angle = np.arctan2(dy, dx)
        # Inverted Spin to enforce co-shearing vector alignment in the gap
        self.VX[mask] = spin_v * np.sin(angle)[mask] 
        self.VY[mask] = -spin_v * np.cos(angle)[mask]
        
        # Anti-Phase AC Breath: Inward vacuum pull matching the proton's outward push
        self.VX[mask] += 0.2 * np.cos(pulse_phase + np.pi) * np.cos(angle)[mask]
        self.VY[mask] += 0.2 * np.cos(pulse_phase + np.pi) * np.sin(angle)[mask]

    def step_momentum_diffusion(self, iterations=25, alpha=0.28):
        """
        The core relational engine loop. Substrate cells sideswipe neighbors,
        propagating and aligning vector profiles across space.
        """
        for _ in range(iterations):
            next_vx = self.VX.copy()
            next_vy = self.VY.copy()
            
            # 4-Neighbor Laplacian Relaxation matrix stencil
            for i in range(1, self.res - 1):
                for j in range(1, self.res - 1):
                    # Skip cells locked by particle boundaries to protect core drives
                    if self.boundary_mask[i, j]:
                        continue
                        
                    avg_vx = (self.VX[i+1, j] + self.VX[i-1, j] + self.VX[i, j+1] + self.VX[i, j-1]) * 0.25
                    avg_vy = (self.VY[i+1, j] + self.VY[i-1, j] + self.VY[i, j+1] + self.VY[i, j-1]) * 0.25
                    
                    next_vx[i, j] = (1.0 - alpha) * self.VX[i, j] + alpha * avg_vx
                    next_vy[i, j] = (1.0 - alpha) * self.VY[i, j] + alpha * avg_vy
            
            # Tight enforcement of the medium saturation ceiling (v <= c)
            v_mags = np.sqrt(next_vx**2 + next_vy**2)
            clamp = v_mags > self.c
            if np.any(clamp):
                next_vx[clamp] = (next_vx[clamp] / v_mags[clamp]) * self.c
                next_vy[clamp] = (next_vy[clamp] / v_mags[clamp]) * self.c
                
            self.VX, self.VY = next_vx, next_vy

    def compute_pressure_field(self):
        v_sq = self.VX**2 + self.VY**2
        return 1.0 - v_sq

# --- SIMULATION CONFIGURATION ---
if __name__ == "__main__":
    # Instantiate the active space canvas grid
    space = ChargeInteractionFabric(resolution=60)
    
    # Force embed the Proton Anchor and the Wide-Open Electron Ring
    space.embed_proton_core(cx=-0.6, cy=0.0, radius=0.45, spin_v=0.95, pulse_phase=0.0)
    space.embed_electron_core(cx=1.2, cy=0.0, radius=0.85, spin_v=0.65, pulse_phase=0.0)
    
    print("Propagating field states through neighbor-to-neighbor entrainment relaxation...")
    space.step_momentum_diffusion(iterations=35, alpha=0.3)
    
    # Calculate emergent Bernoulli profiles
    P = space.compute_pressure_field()
    
    # --- RENDER PRINCIPLE LOG LAYOUT ---
    plt.figure(figsize=(13, 9), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    # Plot the emergent pressure topography
    contour = plt.contourf(space.X, space.Y, P, levels=70, cmap='inferno')
    cbar = plt.colorbar(contour, label='Substrate Hydrostatic Pressure (P)')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    # Overlay the vectorized fluid flow directional arrows
    mask = (space.VX**2 + space.VY**2) > 0.008
    plt.quiver(space.X[mask], space.Y[mask], space.VX[mask], space.VY[mask],
               color='white', alpha=0.55, scale=24, width=0.0025)
    
    # Annotate the specific geometric structures to verify your paper's invariants
    plt.text(-0.6, 0.6, "Proton Core\n(Tight Anchor)", color='magenta', fontsize=10, ha='center', fontweight='bold')
    plt.text(1.2, 1.1, "Electron Disk\n(Wide Open Drain)", color='cyan', fontsize=10, ha='center', fontweight='bold')
    plt.text(0.3, -0.2, "Emergent\nVacuum Bridge", color='lime', fontsize=10, ha='center', fontweight='bold')
    
    plt.title("TIR Substrate Engine: First-Principles Opposite-Charge Entrainment Lock-In", color='white', fontsize=12, pad=15)
    plt.xlabel("X Substrate coordinate", color='white')
    plt.ylabel("Y Substrate coordinate", color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()