import math
import matplotlib.pyplot as plt

# ==========================================
# Level 1: Soliton Core Ground-State Settling
# Pure Kinematic Propagation Model
# ==========================================

# --- 1. Particle State Initialization ---
v_tan = 0.8         # Fixed rotational baseline (Angular Momentum)
v_rad = 0.6         # High initial excess energy (Will bleed out)
a = 1.0             # Constant baseline vacuum stiffness
theta_rad = 0.0     # Internal phase tracker

# --- Engine Parameters ---
dt = 0.01           
radiation_rate = 0.5 # How fast unmatched energy propagates into the medium
iterations = 3000   

history_r = []
history_v_rad = []
history_eta = []
time_steps = []

# --- 2. The Pure Execution Loop ---
print("Initiating pure kinematic wave-bleed...")

for i in range(iterations):
    # Step 1: Passive Geometric Readout
    r = (v_tan**2 + v_rad**2) / a
    
    # Step 2: Advance the wave phase
    delta_theta = (v_tan / (2 * math.pi * r)) * dt
    theta_rad += delta_theta
    
    # Step 3: Phase Coherence (eta)
    eta = math.cos(theta_rad)
    
    # Step 4: Propagate Error into the Medium (Evanescent Bleed)
    # If out of phase, destructive interference bleeds the radial velocity
    splatter = 1.0 - eta
    v_rad -= (radiation_rate * splatter * v_rad) * dt
    
    if v_rad < 0: v_rad = 0 # Prevent negative absolute magnitude
        
    # Record state
    time_steps.append(i * dt)
    history_r.append(r)
    history_v_rad.append(v_rad)
    history_eta.append(eta)

print(f"Simulation complete. Final locked radius: {history_r[-1]:.4f}")
print(f"Residual internal energy (v_rad): {history_v_rad[-1]:.4f}")

# --- 3. Visualization ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.canvas.manager.set_window_title('Pure Primitive Equation: Wave-Bleed Settling')

# Plot Radius (r)
ax1.plot(time_steps, history_r, color='blue', linewidth=2)
ax1.set_ylabel('Radius (r)', color='blue', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title('Radius Contracting via Energy Propagation', fontweight='bold')

# Plot Radial Velocity (v_rad)
ax2.plot(time_steps, history_v_rad, color='purple', linewidth=2)
ax2.set_ylabel('Energy Budget ($v_{rad}$)', color='purple', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)

# Plot Phase Coherence (eta)
ax3.plot(time_steps, history_eta, color='green', linewidth=2)
ax3.set_ylabel('Coherence ($\eta$)', color='green', fontweight='bold')
ax3.set_xlabel('Time (Simulation Steps)', fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.axhline(1.0, color='black', linestyle=':', alpha=0.8)

plt.tight_layout()
plt.show()