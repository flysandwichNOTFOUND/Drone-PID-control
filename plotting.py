"""Simulation and PID-validation graphs, including the terrain preview.

All rendering settings live here. Flight limits and ground height are supplied
by the simulation data, keeping this module independent of main.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from matplotlib.patches import Patch

# Terrain preview and wider scale for the 3D graph (metres).
TERRAIN_PLOT_SPAN = 16.0


def use_backend(name):
    """Select a rendering backend, for example Agg for unattended tuning."""

    plt.switch_backend(name)


def close_all():

    plt.close("all")


def mark_flight_stages(axes, stage_start_times):
    """Mark actual mission stage starts consistently across time-series panels."""

    colors = {"climbing": "#567a35", "flying": "#206593",
              "landing": "#aa6632", "landed": "#656565"}
    events = [(stage, time) for stage, time in stage_start_times.items() if stage in colors]
    for ax in axes:
        right_edge = ax.get_xlim()[1]
        for index, (stage, start_time) in enumerate(events):
            if stage not in colors:
                continue
            color = colors[stage]
            ax.axvline(start_time, color = color, linestyle = ":", linewidth = 1.2, alpha = 0.8,
                       gid = f"stage-{stage}")
            end_time = events[index + 1][1] if index + 1 < len(events) else right_edge
            label_time = (start_time + end_time) / 2
            ax.annotate(stage.capitalize(), (label_time, 1.0),
                        xycoords = ax.get_xaxis_transform(), xytext = (0, 4),
                        textcoords = "offset points", color = color, fontsize = 8,
                        ha = "center", va = "bottom", annotation_clip = False)
        ax.set_title(ax.get_title(), pad = 26)


def plot_data(simulation_data, *, show = True):

    time_history = np.asarray(simulation_data["time"], dtype = float)
    height_history = np.asarray(simulation_data["height"], dtype = float)
    velocity_history = np.asarray(simulation_data["velocity"], dtype = float)
    position_history = np.asarray(simulation_data["position"], dtype = float)
    motor_speed_history = np.asarray(simulation_data["motor_speed"], dtype = float)
    orientation_history = np.asarray(simulation_data["orientation"], dtype = float)
    target_orientation_history = np.asarray(simulation_data["target_orientation"], dtype = float)
    target_position_history = np.asarray(simulation_data.get("commanded_position_history", simulation_data["target_position_history"]), dtype = float)

    hover_speed = simulation_data["hover_speed"]

    if target_orientation_history.ndim == 1:
        target_orientation_history = np.tile(target_orientation_history, (len(time_history), 1))

    if orientation_history.ndim != 2:
        raise ValueError("orientation_history must contain one [roll, pitch, yaw] array for every timestep.")

    figure = plt.figure(figsize = (12, 12))

    # Plot 1: Altitude
    plt.subplot(3, 2, 1)
    plt.plot(time_history, height_history, label = "Actual height")
    plt.step(time_history, target_position_history[:, 2], color = "red", linestyle = "--", where = "post", label = "Target height")
    if "ground_height_history" in simulation_data:
        plt.plot(time_history, simulation_data["ground_height_history"], color = "olivedrab", label = "Ground elevation")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Drone Altitude")
    plt.grid(True)
    plt.legend()

    # Plot 2: Vertical velocity
    plt.subplot(3, 2, 2)
    plt.plot(time_history, velocity_history, color = "orange")
    plt.axhline(y = 0, color = "black", linestyle = "--")
    plt.xlabel("Time (s)")
    plt.ylabel("Vertical velocity (m/s)")
    plt.title("Vertical Velocity")
    plt.grid(True)

    # Plot 3: Four motor speeds
    plt.subplot(3, 2, 3)

    for motor_index in range(4):
        plt.plot(time_history, motor_speed_history[:, motor_index], label = f"Motor {motor_index + 1}")

    plt.axhline(y = hover_speed, color = "red", linestyle = "--", label = "Hover speed")
    plt.axhline(y = simulation_data["max_motor_speed"], color = "purple", linestyle = "--", linewidth = 1.2, label = "Maximum motor speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Motor speed (rad/s)")
    plt.title("Motor Speeds")
    plt.grid(True)
    plt.legend(fontsize = 8)

    # Plot 4: Roll, pitch and yaw
    plt.subplot(3, 2, 4)
    axis_names = ["Roll", "Pitch", "Yaw"]
    axis_colors = ["tab:blue", "tab:orange", "tab:green"]

    for axis_index in range(3):
        plt.plot(time_history, orientation_history[:, axis_index], color = axis_colors[axis_index], label = f"Actual {axis_names[axis_index]}")
        plt.plot(time_history, target_orientation_history[:, axis_index], color = axis_colors[axis_index], linestyle = "--", label = f"Target {axis_names[axis_index]}")

    plt.axhline(y = 0, color = "black", linewidth = 0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation (degrees)")
    plt.title("Orientation Tracking")
    plt.grid(True)
    plt.legend(fontsize = 7)

    # Plot 5: Horizontal position
    plt.subplot(3, 2, 5)
    plt.plot(time_history, position_history[:, 0], color = "tab:blue", label = "X position")
    plt.plot(time_history, position_history[:, 1], color = "tab:orange", label = "Y position")
    plt.step(time_history, target_position_history[:, 0], color = "tab:blue", linestyle = "--", where = "post", label = "Target X")
    plt.step(time_history, target_position_history[:, 1], color = "tab:orange", linestyle = "--", where = "post", label = "Target Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Horizontal position (m)")
    plt.title("Horizontal Position and Drift")
    plt.grid(True)
    plt.legend(fontsize = 8)

    # Plot 6: Distance to the active waypoint
    plt.subplot(3, 2, 6)
    plt.plot(time_history, simulation_data["distance_to_target"], label = "Distance to active waypoint")
    plt.axhline(y = simulation_data["arrival_tolerance"], color = "red", linestyle = "--", label = "Arrival tolerance")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Waypoint Tracking")
    plt.grid(True)
    plt.legend(fontsize = 8)

    mark_flight_stages(figure.axes, simulation_data.get("stage_start_times", {}))
    figure.subplots_adjust(hspace = 0.55)
    plt.tight_layout(h_pad = 2.0)
    plt.savefig("altitude_control_results.png", dpi = 300)
    if show:
        plt.show()
    return figure

def plot_3d_trajectory(simulation_data, *, show = True):

    ground_height = simulation_data["ground_height"]
    terrain = simulation_data["terrain"]
    positions = np.asarray(simulation_data["position"], dtype = float)
    waypoints = np.asarray(simulation_data["waypoints"], dtype = float)

    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111, projection = "3d", computed_zorder = False)

    all_points = np.vstack((positions, waypoints))
    map_points = np.vstack((all_points[:, :2], getattr(terrain, "center", all_points[0, :2])))
    if hasattr(terrain, "plot_bounds"):
        xmin, xmax, ymin, ymax = terrain.plot_bounds
        map_points = np.vstack((map_points, [[xmin, ymin], [xmax, ymax]]))
    lower = map_points.min(axis = 0)
    upper = map_points.max(axis = 0)
    center = (lower + upper) / 2.0
    span = max(TERRAIN_PLOT_SPAN, float(np.max(upper - lower)) *
               (1.0 if hasattr(terrain, "plot_bounds") else 1.3))
    x_limits = (center[0] - span / 2, center[0] + span / 2)
    y_limits = (center[1] - span / 2, center[1] + span / 2)
    terrain_x, terrain_y = np.meshgrid(
        np.linspace(*x_limits, 220), np.linspace(*y_limits, 220))
    terrain_z = terrain.height(terrain_x, terrain_y)
    terrain_colors = LinearSegmentedColormap.from_list(
        "landscape", ["#315646", "#6b8c58", "#aaa178", "#a9957b", "#c4b497"])
    colors = LightSource(azdeg = 315, altdeg = 45).shade(
        terrain_z, cmap = terrain_colors, vert_exag = 1.0, blend_mode = "soft",
        dx = span / 219, dy = span / 219,
        vmin = float(np.min(terrain_z)),
        vmax = max(float(np.max(terrain_z)), float(np.min(terrain_z)) + 0.01))
    ax.plot_surface(terrain_x, terrain_y, terrain_z, facecolors = colors,
                    linewidth = 0, antialiased = False, shade = False,
                    rcount = 220, ccount = 220, zorder = 1)

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color = "tab:blue", label = "Actual trajectory")
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color = "tab:orange", linestyle = "--", marker = "o", label = "Planned route")
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color = "green", s = 80, zorder = 4, label = "First recorded position")
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2], color = "red", marker = "*", s = 150, zorder = 4, label = "Endpoint")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color = "purple", marker = "x", s = 80, zorder = 5, label = "Final position")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("Drone Trajectory over Terrain")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    z_lower = min(float(np.min(terrain_z)), float(all_points[:, 2].min()))
    z_upper = max(ground_height + 6.0, float(all_points[:, 2].max()) + 1.0,
                  float(np.max(terrain_z)) + 1.0)
    ax.set_zlim(z_lower, z_upper)
    # Match the axes' physical spans so terrain and flight retain their proportions.
    ax.set_box_aspect((span, span, z_upper - z_lower))
    ax.view_init(elev = 28, azim = -60)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [Patch(facecolor = "olivedrab", alpha = 0.85)], labels + ["Terrain"],
              loc = "upper left", fontsize = 8)
    plt.tight_layout()
    plt.savefig("drone_3d_trajectory.png", dpi = 300)
    if show:
        plt.show()
    return fig

def plot_validation(history, *, target_height, hover_speed, max_motor_speed, show = True):

    keys = ("time", "position", "velocity", "motor_speeds",
            "orientation", "target_orientation", "target_position")
    data = {key: np.asarray(history[key], dtype = float) for key in keys}

    time = data["time"]
    position = data["position"]
    velocity = data["velocity"]
    motor_speeds = data["motor_speeds"]

    orientation_degrees = np.degrees(
        data["orientation"]
    )

    target_orientation_degrees = np.degrees(
        data["target_orientation"]
    )

    target_position = data["target_position"]

    figure, axes = plt.subplots(
        3,
        2,
        figsize = (12, 12),
    )

    axes[0, 0].plot(
        time,
        position[:, 2],
        label = "Actual height",
    )
    axes[0, 0].axhline(
        target_height,
        color = "red",
        linestyle = "--",
        label = "Target height",
    )
    axes[0, 0].set_title("Drone Altitude")
    axes[0, 0].set_ylabel("Height (m)")
    axes[0, 0].legend()

    axes[0, 1].plot(
        time,
        velocity[:, 2],
        color = "tab:orange",
    )
    axes[0, 1].axhline(
        0.0,
        color = "black",
        linestyle = "--",
    )
    axes[0, 1].set_title("Vertical Velocity")
    axes[0, 1].set_ylabel("Vertical velocity (m/s)")

    for motor_index in range(4):
        axes[1, 0].plot(
            time,
            motor_speeds[:, motor_index],
            label = f"Motor {motor_index + 1}",
        )

    axes[1, 0].axhline(
        hover_speed,
        color = "red",
        linestyle = "--",
        label = "Hover speed",
    )
    axes[1, 0].axhline(
        max_motor_speed,
        color = "purple",
        linestyle = "--",
        label = "Maximum motor speed",
    )
    axes[1, 0].set_title("Motor Speeds")
    axes[1, 0].set_ylabel("Motor speed (rad/s)")
    axes[1, 0].legend(fontsize = 8)

    labels = ["Roll", "Pitch", "Yaw"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for axis_index, label in enumerate(labels):
        axes[1, 1].plot(
            time,
            orientation_degrees[:, axis_index],
            color = colors[axis_index],
            label = f"Actual {label}",
        )
        axes[1, 1].plot(
            time,
            target_orientation_degrees[:, axis_index],
            color = colors[axis_index],
            linestyle = "--",
            label = f"Target {label}",
        )

    axes[1, 1].set_title("Orientation Tracking")
    axes[1, 1].set_ylabel("Orientation (degrees)")
    axes[1, 1].legend(fontsize = 7)

    axes[2, 0].plot(
        time,
        position[:, 0],
        color = "tab:blue",
        label = "X position",
    )
    axes[2, 0].plot(
        time,
        position[:, 1],
        color = "tab:orange",
        label = "Y position",
    )
    axes[2, 0].plot(
        time,
        target_position[:, 0],
        color = "tab:blue",
        linestyle = "--",
        label = "Target X",
    )
    axes[2, 0].plot(
        time,
        target_position[:, 1],
        color = "tab:orange",
        linestyle = "--",
        label = "Target Y",
    )
    axes[2, 0].set_title("Horizontal Position")
    axes[2, 0].set_ylabel("Position (m)")
    axes[2, 0].legend(fontsize = 8)

    axes[2, 1].axis("off")

    for axis in axes.flat:
        if axis.axison:
            axis.set_xlabel("Time (s)")
            axis.grid(True)

    figure.tight_layout()
    figure.savefig(
        "auto_tune_results.png",
        dpi = 200,
    )
    if show:
        plt.show()
    return figure
