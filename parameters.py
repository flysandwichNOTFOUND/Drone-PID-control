"""
Shared simulation parameters.

mission targets, and mode selections are configured in main.py.
"""
import numpy as np

# ============================================================
# PID PARAMETERS
# ============================================================

# PID GAINS
KP = 211.092059550
KI = 0.223363309
KD = 93.863981577

# Orientation and position gains: [Kp, Ki, Kd]
ROLL_GAINS = np.array([50.679433130, 0.026414318, 14.574450033])
PITCH_GAINS = np.array([50.571938943, 0.060850500, 14.352389900])
YAW_GAINS = np.array([37.650316822, 0.048992121, 34.053486350])
X_POSITION_GAINS = np.array([0.301072977, 0.000751421, 0.297699224])
Y_POSITION_GAINS = np.array([0.303979658, 0.004076894, 0.309451213])

# ============================================================
# DRONE PARAMETERS
# ============================================================

TOTAL_MASS = 1.0                  # kg

ARM_LENGTH = 0.20                  # m

MAX_THRUST_PER_MOTOR = 10.0        # N

MOMENT_OF_INERTIA = np.array([
    0.005,                          # roll inertia
    0.005,                          # pitch inertia
    0.009                           # yaw inertia
], dtype = float)

THRUST_COEFFICIENT = 1.0e-5

TORQUE_COEFFICIENT = 2.0e-7

GRAVITY = 9.81


# ============================================================
# CONTROLLER LIMITS
# ============================================================

MAX_ORIENTATION_CORRECTION = 20.0

MAX_TILT_DEGREES = 15.0

MAX_TILT_RATE_DEGREES = 10.0       # roll/pitch command change per second

TILT_SMOOTHING_TIME = 0.03         # seconds; zero disables smoothing


# ============================================================
# ROUTE PLANNING
# ============================================================

GROUND_HEIGHT = 0.0

MINIMUM_CLEARANCE = 2.0

WAYPOINT_SPACING = 1.0

ASTAR_GRID_RESOLUTION = 1.0

ASTAR_SEARCH_MARGIN = 5.0

ASTAR_CLIMB_WEIGHT = 2.0

ASTAR_MAXIMUM_ALTITUDE = None

ASTAR_CLEARANCE_MARGIN = 0.25


# ============================================================
# FLIGHT REFERENCE AND ARRIVAL
# ============================================================

MAX_FLIGHT_SPEED = 1.2

MAX_FLIGHT_ACCELERATION = 0.8

MAX_FLIGHT_JERK = 1.0             # m/s cubed; controls acceleration transitions

MAX_CLIMB_RATE = 0.8

MAX_DESCENT_RATE = 0.4

PATH_LOOKAHEAD_DISTANCE = 1.0

ARRIVAL_TOLERANCE = 0.2

SPEED_TOLERANCE = 0.15


# ============================================================
# LANDING
# ============================================================

INCLUDE_LANDING = True

LANDING_DESCENT_SPEED = 0.4

LANDING_FLARE_SPEED = 0.12

LANDING_FLARE_HEIGHT = 0.5

MAX_LANDING_SLOPE_DEGREES = 20.0


# ============================================================
# TERRAIN
# ============================================================

TERRAIN_SEED = 17

HILL_CENTER = np.array([5.0, 4.0])

HILL_HEIGHT = 2

HILL_WIDTH = 1.6

TERRAIN_SLOPE = np.array([0.1, 0.0])


# ============================================================
# WIND
# ============================================================

WIND_VELOCITY = np.array([0.0, 0.0, 0.0], dtype = float)

WIND_FORCE_COEFFICIENT = 0.2

GUST_START = 10.0

GUST_DURATION = 2.0

GUST_PERIOD = 6.0


# ============================================================
# SIMULATION
# ============================================================

DT = 0.005                          # seconds per step

SIMULATION_TIME = 40.0            


# ============================================================
# DERIVED MOTOR SPEEDS
# ============================================================

MAX_MOTOR_SPEED = np.sqrt(
    MAX_THRUST_PER_MOTOR
    / THRUST_COEFFICIENT
)

HOVER_SPEED = np.sqrt(
    TOTAL_MASS * GRAVITY
    / (4 * THRUST_COEFFICIENT)
)
