import math
import matplotlib.pyplot as plt

# ==========================================
# Level 1: Soliton Core Ground-State Settling
# Theory of Kinematically Defined Wave Fields
# ==========================================

# --- 1. Particle State Initialization ---
v_tan = 0.8         # Fixed tangential circulation budget
v_rad = 0.2         # Radial breathing pulse speed
r = 5.0             # Deliberately unmatched starting radius
a_0 = 1.0           # Baseline vacuum acceleration/stiffness
a = a_0             # Current localized stiffness
theta_rad = 0.0     # Internal phase tracker of the radial wave

# --- Engine Parameters ---
dt = 0.05           # Simulation time-step
kappa = 3.0         # Restoring force multiplier (hydraulic spike intensity)
gamma = 1.5         # Damping multiplier (substrate relaxation/viscosity)
iterations = 1500   # Total loops to run

# --- Data Tracking arrays (for the graph) ---
history_r = []
history_a = []
history_eta = []
time_steps = []

# --- 2. The Execution Loop ---
print("Initiating simulation... watching variables seek equilibrium.")

for i in range(iterations):
    # Step 1: Advance the wave phase based on current geometry
    # The wave's speed through the loop depends on circulation and perimeter
    delta_theta = (v_tan / (2 * math.pi * r)) * dt
    theta_rad += delta_theta
    
    # Step 2: Calculate Phase Coherence Coefficient (eta)
    # 1.0 = Perfect harmony, -1.0 = Destructive clash
    eta = math.cos(theta_rad)
    
    # Step 3: Update Substrate Stiffness (a) via Gradient Flow
    # Phase mismatch causes 'a' to spike. It naturally relaxes back toward a_0.
    stiffness_error = 1.0 - eta
    da_dt = (kappa * stiffness_error) - gamma * (a - a_0)
    a += da_dt * dt
    
    # Prevent 'a' from dropping below absolute zero (mathematical safeguard)
    if a < 0.001: 
        a = 0.001
        
    # Step 4: Passive Geometric Readout (r)
    # The radius is completely slaved to the velocity and localized pressure
    r = (v_tan**2 + v_rad**2) / a
    
    # Record the current state for the graph
    time_steps.append(i * dt)
    history_r.append(r)
    history_a.append(a)
    history_eta.append(eta)

print(f"Simulation complete. Final settled radius: {r:.4f}")

# --- 3. Visualization ---
# This builds the visual dashboard to plot your 3 variables
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.canvas.manager.set_window_title('Level 1: Ground State Radius Settling')

# Plot Radius (r)
ax1.plot(time_steps, history_r, color='blue', linewidth=2)
ax1.set_ylabel('Radius (r)', color='blue', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title('Soliton Core Discovering its Harmonic Ground-State', fontweight='bold')

# Plot Acceleration/Stiffness (a)
ax2.plot(time_steps, history_a, color='red', linewidth=2)
ax2.set_ylabel('Stiffness (a)', color='red', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)

# Plot Phase Coherence (eta)
ax3.plot(time_steps, history_eta, color='green', linewidth=2)
ax3.set_ylabel('Coherence ($\eta$)', color='green', fontweight='bold')
ax3.set_xlabel('Time (Simulation Steps)', fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.axhline(1.0, color='black', linestyle=':', alpha=0.8) # The perfect harmony line

plt.tight_layout()
plt.show()