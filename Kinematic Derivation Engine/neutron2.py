import numpy as np
import matplotlib.pyplot as plt

class AxialNucleonFabric:
    def __init__(self, x_bounds=(-3.0, 3.0), y_bounds=(-2.5, 2.5), resolution=110, c=1.0):
        self.res_x = resolution + 20
        self.res_y = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res_x)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Accumulator for the permanent macroscopic pressure footprint
        self.pressure_accumulator = np.zeros_like(self.X)
        self.time_steps = 0

    def calculate_3d_projected_field(self, t, freq=16.0, pulse_amp=0.45):
        """ 
        Calculates the instantaneous substrate pressure by projecting the 
        3D toroidal radial waves onto the 2D (XY) orbital plane.
        """
        r_core = 0.35
        
        # --- PROTON ENGINE (Vertical Axis) ---
        # Spin axis aligns with Z. The 2D plane slices right through its equator.
        # Its radial wave propagates fully into the XY plane.
        dist_p = np.sqrt((self.X - (-1.0))**2 + self.Y**2)
        lag_p = np.abs(dist_p - r_core) / self.c
        
        # Pure isotropic 2D propagation
        wave_p = np.sin(freq * (t - lag_p))
        amp_p = pulse_amp / (dist_p + 0.1)
        v_rad_p = amp_p * wave_p
        
        # --- NEUTRON ENGINE (Tilted 90 Degrees) ---
        # Spin axis lies IN the XY plane (e.g., along the Y axis).
        # We are looking at a vertical cross-section of the donut.
        dist_n = np.sqrt((self.X - 1.0)**2 + self.Y**2)
        lag_n = np.abs(dist_n - r_core) / self.c
        
        # The radial wave is highly anisotropic in this slice.
        # It pulses maximally along the X-axis (equator of the tilted donut) 
        # and minimally along the Y-axis (the "hole" of the donut).
        angle_n = np.arctan2(self.Y, self.X - 1.0)
        
        # Geometric Projection Factor: The wave is strongest at the equator (cos(theta) = 1)
        # and vanishes at the poles (cos(theta) = 0)
        projection_mask = np.abs(np.cos(angle_n))
        
        wave_n = np.sin(freq * (t - lag_n))
        amp_n = pulse_amp / (dist_n + 0.1)
        v_rad_n = amp_n * wave_n * projection_mask
        
        # --- AGGREGATE FIELD ---
        mask_p = dist_p >= r_core
        mask_n = dist_n >= r_core
        
        v_rad_total = np.zeros_like(self.X)
        v_rad_total[mask_p] += v_rad_p[mask_p]
        v_rad_total[mask_n] += v_rad_n[mask_n]
        
        v_rad_total = np.clip(v_rad_total, -self.c, self.c)
        return 1.0 - (v_rad_total**2)

    def integrate_footprint(self, cycles=5, steps_per_cycle=30, freq=16.0):
        period = (2 * np.pi) / freq
        dt = (cycles * period) / (cycles * steps_per_cycle)
        
        for step in range(cycles * steps_per_cycle):
            t = step * dt
            self.pressure_accumulator += self.calculate_3d_projected_field(t, freq)
            self.time_steps += 1
            
        return self.pressure_accumulator / self.time_steps

# --- EXECUTE ---
if __name__ == "__main__":
    space = AxialNucleonFabric(resolution=110)
    P_avg = space.integrate_footprint()
    
    # --- RENDER ---
    plt.figure(figsize=(13, 8), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    contour = plt.contourf(space.X, space.Y, P_avg, levels=85, cmap='magma_r')
    cbar = plt.colorbar(contour, label='Time-Averaged Electrostatic Pressure Deficit')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    circle_p = plt.Circle((-1.0, 0), 0.35, color='magenta', fill=False, linewidth=2, linestyle='-')
    circle_n = plt.Circle((1.0, 0), 0.35, color='cyan', fill=False, linewidth=2, linestyle='-')
    ax.add_patch(circle_p)
    ax.add_patch(circle_n)
    
    plt.text(-1.0, 0.5, "Proton Engine\n(Vertical Axis / Uniform Charge)", color='magenta', fontsize=10, ha='center', fontweight='bold')
    plt.text(1.0, 0.5, "Neutron Engine\n(90° Tilted Axis / Anisotropic)", color='cyan', fontsize=10, ha='center', fontweight='bold')
    
    plt.title("TIR Engine: Deriving Neutron Neutrality via 90° Axial Geometric Projection", color='white', fontsize=13, pad=15)
    plt.xlabel("X Substrate Plane", color='white'); plt.ylabel("Y Substrate Plane", color='white')
    ax.tick_params(colors='white'); ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()