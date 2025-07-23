import numpy as np
import matplotlib.pyplot as plt

from spelke_net.utils.flow import flow_uv_to_colors

def make_radial_color_wheel_pcolormesh(radius=20, resolution=512, angle_label_offset=0.05):
    """
    Generate a smooth radial flow color wheel using pcolormesh in polar coordinates,
    with pixel radius ticks and angular labels offset from the disc.
    """
    # Define polar grid
    r = np.linspace(0, 1, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, r_grid = np.meshgrid(theta, r)

    # Convert to flow vector field (u, v)
    u = r_grid * np.cos(theta_grid)
    v = r_grid * np.sin(theta_grid)
    flow_rgb = flow_uv_to_colors(u, -v) / 255.0  # Flip v if needed

    # Plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.pcolormesh(theta, r, flow_rgb, shading='gouraud')

    # Radial ticks (in pixels)
    tick_locs = np.linspace(0, 1, 5)
    tick_labels = np.round(tick_locs * radius).astype(int)
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(labelsize=24)

    # Move angular (theta) labels outward by small offset
    for label in ax.get_xticklabels():
        x, y = label.get_position()
        label.set_position((x, y - angle_label_offset))

    # Title and save
    ax.set_title(f"Flow color coding (radius = {radius} pixels)", fontsize=24, pad=30)
    plt.tight_layout()
    plt.savefig('./flow_color_wheel.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run it
make_radial_color_wheel_pcolormesh(radius=20)