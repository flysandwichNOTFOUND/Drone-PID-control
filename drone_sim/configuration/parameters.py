import numpy as np

# ============================================================
# PID PARAMETERS
# ============================================================

# PID GAINS
KP = 235.322440352
KI = 0.159827328
KD = 106.244649639

# Orientation and position gains: [Kp, Ki, Kd]
ROLL_GAINS = np.array([48.937711680, 0.031942350, 16.399264339])
PITCH_GAINS = np.array([67.920116448, 0.080796429, 9.644687487])
YAW_GAINS = np.array([37.143020405, 0.043782265, 31.138992396])
X_POSITION_GAINS = np.array([0.437555050, 0.000649502, 0.446548836])
Y_POSITION_GAINS = np.array([0.455969487, 0.004478701, 0.432557432])

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

LANDING_FLARE_HEIGHT = 0.8

LANDING_APPROACH_GAIN = 0.5       # 1/s
LANDING_TOUCHDOWN_SPEED = 0.02    # m/s

MAX_LANDING_SLOPE_DEGREES = 20.0

LANDING_SEARCH_RADIUS = 3.0       # m

LANDING_SEARCH_RESOLUTION = 0.25  # m

LANDING_MAX_ACCELERATION = 0.3    # m/s²

LANDING_MAX_JERK = 0.5            # m/s³

# ============================================================
# TERRAIN
# ============================================================

TERRAIN_SEED = 17
TERRAIN_BOUNDS = (-100.0, 100.0, -100.0, 100.0) 

#noise configuration
LANDSCAPE_HEIGHT = 18.0          #scales noise variation, not an absolute maximum
LANDSCAPE_FEATURE_SIZE = 60.0    #width of the largest noise cells
LANDSCAPE_OCTAVES = 4            # noise layers
LANDSCAPE_PERSISTENCE = 0.3      # amplitude multiplier 
LANDSCAPE_LACUNARITY = 2.0       # frequency multiplier
LANDSCAPE_RESOLUTION = 1.0       #cached height-map spacing

HILL_CENTER = np.array([5.0, 4.0])

HILL_HEIGHT = 10

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

SIMULATION_TIME = 40.0           # fixed-duration PID trials; missions run until arrival by default
ARRIVAL_WAIT_TIME = 1.5          # simulated seconds to wait at the destination before stopping

# Gym backend initialization (collision geometry is an approximation).
GYM_BODY_RADIUS = 0.08            # metres; central body, excluding rotor arms
GYM_BODY_HEIGHT = 0.04            # metres
GYM_PHYSICS_SUBSTEPS = 1          # DT=0.005 -> control/physics both 200 Hz
GYM_TERRAIN_BOUNDS = TERRAIN_BOUNDS
GYM_TERRAIN_RESOLUTION = 0.5      # mesh spacing in metres


# ============================================================
# DERIVED MOTOR SPEEDS
# ============================================================

MAX_MOTOR_SPEED = np.sqrt( MAX_THRUST_PER_MOTOR / THRUST_COEFFICIENT )

HOVER_SPEED = np.sqrt( TOTAL_MASS * GRAVITY / (4 * THRUST_COEFFICIENT) )

# ============================================================
#MISSION TARGET AND MODE SELECTION
# ============================================================

