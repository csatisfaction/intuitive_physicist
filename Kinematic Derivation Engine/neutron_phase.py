import numpy as np
import matplotlib.pyplot as plt

class PhaseNeutralityFabric:
    def __init__(self, x_bounds=(-3.0, 3.0), y_bounds=(-2.5, 2.5), resolution=100, c=1.0):
        self.res_x = resolution + 20
        self.res_y = resolution
        self.c = float(c)
        self.x = np.linspace(x_bounds[0], x_bounds[1], self.res_x)
        self.y = np.linspace(y_bounds[0], y_bounds[1], self.res_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Accumulator for time-averaged pressure (the macroscopic electrostatic footprint)
        self.pressure_accumulator = np.zeros_like(self.X)
        self.time_steps = 0

    def calculate_instantaneous_field(self, t, freq=16.0, pulse_amp=0.45):
        """ Calculates the radial wave propagation for the given time step """
        r_p = 0.35 # Proton Radius
        r_n = 0.35 # Neutron Radius
        
        # Distance maps
        dist_p = np.sqrt((self.X - (-1.0))**2 + self.Y**2)
        dist_n = np.sqrt((self.X - 1.0)**2 + self.Y**2)
        
        # Delay propagation based on distance and medium latency (c)
        lag_p = np.abs(dist_p - r_p) / self.c
        lag_n = np.abs(dist_n - r_n) / self.c
        
        # --- THE PHASE ARCHITECTURE ---
        # 1. Proton Engine (Pure Sine Wave - The Base Reference)
        wave_p = np.sin(freq * (t - lag_p))
        
        # 2. Neutron Engine (Cosine Wave - The 90-degree Orthogonal Shift)
        # It pulses at the exact same frequency, but phase-shifted by pi/2
        wave_n = np.cos(freq * (t - lag_n))
        
        # Geometric 1/r wave attenuation
        amp_p = pulse_amp / (dist_p + 0.1)
        amp_n = pulse_amp / (dist_n + 0.1)
        
        # Mask out the physical core interiors (keep waves outside the boundary)
        mask_p = dist_p >= r_p
        mask_n = dist_n >= r_n
        
        v_rad_total = np.zeros_like(self.X)
        v_rad_total[mask_p] += (amp_p * wave_p)[mask_p]
        v_rad_total[mask_n] += (amp_n * wave_n)[mask_n]
        
        # Absolute velocity ceiling
        v_rad_total = np.clip(v_rad_total, -self.c, self.c)
        
        # Instantaneous Bernoulli Pressure Deficit from wave motion
        P_inst = 1.0 - (v_rad_total**2)
        return P_inst

    def integrate_macro_footprint(self, cycles=4, steps_per_cycle=30, freq=16.0):
        """ 
        Integrates the instantaneous pressure over time to reveal 
        the permanent, macroscopic electrostatic field footprint.
        """
        period = (2 * np.pi) / freq
        total_time = cycles * period
        dt = total_time / (cycles * steps_per_cycle)
        
        for step in range(cycles * steps_per_cycle):
            t = step * dt
            P_inst = self.calculate_instantaneous_field(t, freq)
            self.pressure_accumulator += P_inst
            self.time_steps += 1
            
        return self.pressure_accumulator / self.time_steps

# --- EXECUTE ---
if __name__ == "__main__":
    space = PhaseNeutralityFabric(resolution=100)
    
    print("Integrating high-frequency radial waves to map macroscopic charge footprint...")
    P_avg = space.integrate_macro_footprint(cycles=5, steps_per_cycle=40)
    
    # --- RENDER ---
    plt.figure(figsize=(13, 8), facecolor='#111116')
    ax = plt.gca()
    ax.set_facecolor('#111116')
    
    # We plot the variance from baseline pressure (1.0). Darker = Deeper Electrostatic Footprint
    contour = plt.contourf(space.X, space.Y, P_avg, levels=85, cmap='magma_r')
    cbar = plt.colorbar(contour, label='Time-Averaged Electrostatic Pressure Deficit')
    cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(labelcolor='white')
    
    # Annotate the cores
    circle_p = plt.Circle((-1.0, 0), 0.35, color='magenta', fill=False, linewidth=2, linestyle='-')
    circle_n = plt.Circle((1.0, 0), 0.35, color='cyan', fill=False, linewidth=2, linestyle='-')
    ax.add_patch(circle_p)
    ax.add_patch(circle_n)
    
    plt.text(-1.0, 0.5, "Proton Engine\n(Sine Wave)", color='magenta', fontsize=10, ha='center', fontweight='bold')
    plt.text(1.0, 0.5, "Neutron Engine\n(Cosine Wave)", color='cyan', fontsize=10, ha='center', fontweight='bold')
    
    plt.title("TIR Substrate Engine: Deriving Neutron Charge Neutrality via 90° Phase Shift", color='white', fontsize=13, pad=15)
    plt.xlabel("X Substrate Plane", color='white'); plt.ylabel("Y Substrate Plane", color='white')
    ax.tick_params(colors='white'); ax.set_aspect('equal')
    plt.grid(color='#222233', linestyle=':', linewidth=0.5)
    plt.show()