# PANEL 2: Precessional Cone Spherical Mapping
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('#111116')
    ax2.xaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.yaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax2.zaxis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    
    # Draw reference wireframe sphere to help contextualize orientation tilt
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax2.plot_wireframe(xs, ys, zs, color='#222233', linewidth=0.5, alpha=0.4)
    
    # Plot spin orientation vector evolution over time
    ax2.plot(normA[:,0], normA[:,1], normA[:,2], color='magenta', linewidth=1.5, label='Axis A Track')
    ax2.plot(normB[:,0], normB[:,1], normB[:,2], color='cyan', linewidth=2.5, label='Axis B Precession Track')
    
    # Start and end markers for orientation axis
    ax2.scatter(normB[0,0], normB[0,1], normB[0,2], color='yellow', s=50, marker='*', label='Axis B Setup Start')
    ax2.scatter(normB[-1,0], normB[-1,1], normB[-1,2], color='cyan', s=30, marker='o')
    
    ax2.set_title("Spin Orientation Vector Track (Gyroscopic Precession)", color='white', fontsize=12, pad=10)
    ax2.set_xlabel("Normal Nx", color='white')
    ax2.set_ylabel("Normal Ny", color='white')
    ax2.set_zlabel("Normal Nz", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1c1c24', edgecolor='none', labelcolor='white')
    
    # Fixed coordinate lock for axis bounding representation
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])