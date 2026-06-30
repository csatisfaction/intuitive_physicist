import numpy as np
import matplotlib.pyplot as plt

class BreathingAtomFabric:
    def __init__(self, x_bounds=(-2.5, 2.5), y_bounds=(-2.5, 2.5), resolution=85, c=1.0):
        self.res = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Base DC current layers
        dx, dy = self.X, self.Y
        r = np.sqrt(dx**2 + dy**2) + 1e-9
        angle = np.arctan2(dy, dx)
        
        self.VX_dc = np.zeros_like(self.X)
        self.VY_dc = np.zeros_like(self.Y)
        
        p_radius, e_radius = 0.4, 1.4
        p_mask = r <= p_radius
        e_mask = r >= e_radius
        gap_mask = (r > p_radius) & (r < e_radius)
        
        # 1. Inner core (Proton CCW Drive)
        self.VX_dc[p_mask] = -0.95 * np.sin(angle)[p_mask]
        self.VY_dc[p_mask] = 0.95 * np.cos(angle)[p_mask]
        
        # 2. Outer shield (Electron CCW Drive)
        self.VX_dc[e_mask] = -0.65 * np.sin(angle)[e_mask]
        self.VY_dc[e_mask] = 0.65 * np.cos(angle)[e_mask]
        
        # 3. Clean, direct slipstream interpolation inside the orbital gap
        r_gap = r[gap_mask]
        weight = (r_gap - p_radius) / (e_radius - p_radius)
        v_gap_mag = 0.95 * (1.0 - weight) + 0.65 * weight
        
        self.VX_dc[gap_mask] = -v_gap_mag * np.sin(angle)[gap_mask]
        self.VY_dc[gap_mask] = v_gap_mag * np.cos(angle)[gap_mask]

    def evaluate_breathing_pressure(self, t, freq=16.0, pulse_amp=0.35):
        """
        Calculates the instantaneous substrate pressure landscape by superimposing
        the 180-degree out-of-phase phase-retarded radial waves.
        """
        dx, dy = self.X, self.Y
        r = np.sqrt(dx**2 + dy**2) + 1e-9
        angle = np.arctan2(dy, dx)
        
        p_radius, e_radius = 0.4, 1.4
        
        # 1. Wave Front originating from the Central Proton Core (Phase = 0.0)
        dist_from_p = np.abs(r - p_radius)
        lag_p = dist_from_p / self.c
        wave_p = np.cos(freq * (t - lag_p))
        
        # 2. Wave Front originating from the Enclosing Electron Ring (Phase = PI)
        dist_from_e = np.abs(r - e_radius)
        lag_e = dist_from_e / self.c
        wave_e = np.cos(freq * (t - lag_e) + np.pi) # Absolute 180-degree inversion
        
        # Attenuate waves with a clean 1/sqrt(r) geometric dilution profile
        amp_p = pulse_amp / (np.sqrt(r) + 0.2)
        amp_e = pulse_amp / (np.sqrt(np.abs(e_radius - r)) + 0.2)
        
        # Total time-dependent radial velocity field
        v_rad_total = (amp_p * wave_p) + (amp_e * wave_e)
        
        # Reconstruct components for total kinematic snapshot
        vx_total = self.VX_dc + v_rad_total * np.cos(angle)
        vy_total = self.VY_dc + v_rad_total * np.sin(angle)
        
        # Absolute velocity speed capacity ceiling cap (v <= c)
        v_mag = np.sqrt(vx_total**2 + vy_total**2)
        v_mag_clamped = np.minimum(v_mag, self.c)
        
        # Return first-principles Bernoulli hydrostatic pressure field
        return 1.0 - (v_mag_clamped ** 2)

# --- EXECUTION ENGINE RUN ---
if __name__ == "__main__":
    space = BreathingAtomFabric(resolution=85)
    
    # Freeze the timeline at a stable snapshot window to evaluate the interference layout
    target_time = 0.18
    P = space.evaluate_breathing_pressure(t=target_time)
    
    # --- RENDER PROFILE ---
    plt.figure(figsize=(12, 10), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    # High-resolution contour profile mapping the concentric standing wave rings
    contour = plt.contourf(space.X, space.Y, P, levels=90, cmap='inferno')
    close_bar = plt.colorbar(contour, label='Substrate Kinematic Energy / Pressure (P)')
    close_bar.ax.yaxis.label.set_color('white')
    close_bar.ax.tick_params(labelcolor='white')
    
    # Trace specific geometric orbital lines where nodes lock in
    circle1 = plt.Circle((0, 0), 0.4, color='magenta', fill=False, linewidth=2, linestyle='--', alpha=0.8)
    circle2 = plt.Circle((0, 0), 1.4, color='cyan', fill=False, linewidth=2, linestyle='--', alpha=0.8)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    plt.text(0.0, 0.2, "Proton\nCore", color='magenta', fontsize=9, ha='center', fontweight='bold')
    plt.text(0.0, 1.55, "Encapsulating Electron Shield", color='cyan', fontsize=9, ha='center', fontweight='bold')
    plt.text(0.0, 0.85, "Quantized\nStanding Wave Gutter", color='lime', fontsize=9, ha='center', fontweight='bold')
    
    plt.title("TIR Substrate Engine: Concentric Standing Wave Nodes (180° Anti-Phase Breathing)", color='white', fontsize=12, pad=15)
    plt.xlabel("X Spatial Radius", color='white')
    plt.ylabel("Y Spatial Radius", color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    
    # Typo corrected here: removed the invalid font parameter argument
    plt.grid(color='#222233', linestyle=':', linewidth=0.4, alpha=0.5)
    plt.show()