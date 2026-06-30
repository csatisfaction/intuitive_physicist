import math
import matplotlib.pyplot as plt

# ==========================================
# Level 2: The 3D Resolution Engine
# Branching: Magnetic Wobble vs Electric Charge
# ==========================================

def run_resolution_engine(initial_phase_error_deg, v_tan=0.8, vacuum_yield_strength=0.4, iterations=600):
    """
    Simulates a particle attempting to resolve its phase error via 3D tilt.
    If the required tilt causes fluid shear exceeding the vacuum yield strength,
    the particle cavitates and is forced to pump v_rad (Electric Charge).
    """
    dt = 0.05
    phase_error = math.radians(initial_phase_error_deg)
    
    # State Variables
    phi = 0.0          # Current 3D axis tilt angle (radians)
    v_rad = 0.0        # Radial pumping velocity
    
    # Tracking for graphs
    history_phi = []
    history_shear = []
    history_vrad = []
    
    for i in range(iterations):
        # 1. The Engine attempts to tilt the axis to absorb the phase error
        # It applies a smooth restoring torque toward the target phase
        torque = (phase_error - phi) * 0.1
        phi += torque
        
        # 2. Calculate the resulting Fluid Shear at the boundary
        # Shear = v_tan * sin(phi)
        shear = v_tan * math.sin(phi)
        
        # 3. The Hydrodynamic Branching Logic (The Critical Wall)
        if abs(shear) <= vacuum_yield_strength:
            # PATH A: LAMINAR FLOW (MAGNETISM)
            # The tilt is supported by the vacuum. No cavitation occurs.
            v_rad = 0.0 
        else:
            # PATH B: CAVITATION (ELECTRIC CHARGE)
            # The shear exceeds the cohesive limit of space. The gear teeth tear.
            
            # The particle is mathematically locked at the critical angle
            phi_critical = math.asin(vacuum_yield_strength / v_tan)
            phi = phi_critical if phase_error > 0 else -phi_critical
            shear = vacuum_yield_strength
            
            # The unresolved leftover phase error creates a permanent pressure wall
            unresolved_error = phase_error - phi
            
            # The pressure forces the particle to actuate the v_rad release valve.
            # It becomes a permanent, oscillating hydraulic pump.
            v_rad = unresolved_error * math.sin(i * 0.2) # Active breathing
            
        # Record state
        history_phi.append(math.degrees(phi))
        history_shear.append(shear)
        history_vrad.append(v_rad)
        
    return history_phi, history_shear, history_vrad

# --- Run the Two Test Cases ---
# Case 1: Neutron (Small mismatch, resolves via pure tilt)
phi1, shear1, vrad1 = run_resolution_engine(initial_phase_error_deg=15.0)

# Case 2: Electron (180 degree flip, forced to pump)
phi2, shear2, vrad2 = run_resolution_engine(initial_phase_error_deg=180.0)

# --- Visualization Dashboard ---
time_steps = range(600)
fig, axs = plt.subplots(3, 2, figsize=(12, 9))
fig.canvas.manager.set_window_title('3D Resolution Engine: Magnetism vs Charge')

# --- Plotting Case 1 (Left Column) ---
axs[0, 0].set_title('Case 1: Minor Mismatch (Neutron/Magnetism)', fontweight='bold', color='navy')
axs[0, 0].plot(time_steps, phi1, color='blue', linewidth=2)
axs[0, 0].set_ylabel('Axis Tilt Angle (Degrees)')
axs[0, 0].grid(True, linestyle='--', alpha=0.6)

axs[1, 0].plot(time_steps, shear1, color='orange', linewidth=2)
axs[1, 0].axhline(0.4, color='red', linestyle=':', label='Vacuum Yield Limit') # 0.4 is our set yield
axs[1, 0].set_ylabel('Fluid Shear')
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

axs[2, 0].plot(time_steps, vrad1, color='purple', linewidth=2)
axs[2, 0].set_ylabel('Radial Pump ($v_{rad}$)')
axs[2, 0].set_xlabel('Time (Simulation Steps)')
axs[2, 0].grid(True, linestyle='--', alpha=0.6)

# --- Plotting Case 2 (Right Column) ---
axs[0, 1].set_title('Case 2: 180° Anti-Phase (Electron/Charge)', fontweight='bold', color='darkred')
axs[0, 1].plot(time_steps, phi2, color='blue', linewidth=2)
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

axs[1, 1].plot(time_steps, shear2, color='orange', linewidth=2)
axs[1, 1].axhline(0.4, color='red', linestyle=':')
axs[1, 1].legend(['Actual Shear', 'Vacuum Yield Limit'])
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

axs[2, 1].plot(time_steps, vrad2, color='purple', linewidth=2)
axs[2, 1].set_xlabel('Time (Simulation Steps)')
axs[2, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()