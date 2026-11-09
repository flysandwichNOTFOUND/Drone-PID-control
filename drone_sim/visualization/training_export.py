from datetime import datetime, timezone
import json
from uuid import uuid4

import numpy as np

from drone_sim.configuration import parameters
from drone_sim.configuration.project_paths import OUTPUT_DIRECTORY
from drone_sim.physics.paired_training import FEATURE_COLUMNS, FEATURE_UNITS, TARGET_COLUMNS, TARGET_UNITS
from drone_sim.physics.randomization import config_with, flatten


def export_comparison(comparison, initial_state, gains = None, target_yaw = None):
    """Save one immutable episode file containing both backend transitions.

    Row k holds state before the command, that step's motor command and force,
    and state after the command. All angles use radians. Zero-physics rows
    after landing/failure are flagged so training can exclude them.
    """

    from drone_sim.main import current_gains, TARGET_YAW

    state_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'p', 'q', 'r']
    initial = np.concatenate((initial_state.position, initial_state.velocity, initial_state.orientation, initial_state.angular_velocity))
    arrays = {}
    outcomes = {}

    for backend in ('simulation', 'gym_pybullet'):
        history = comparison[backend]
        time = np.asarray(history['time'], dtype = np.float64)
        after = np.column_stack((history['position'], history['velocity_vector'], np.radians(history['orientation']), history['angular_velocity']))
        before = np.vstack((initial, after[:-1]))
        arrays[f'{backend}_time_after'] = time
        arrays[f'{backend}_time_before'] = np.r_[0.0, time[:-1]]

        fields = {
            'state_before': before, 'state_after': after,
            'motor_command': history['motor_speed'], 'rotor_speed': history['rotor_speed'],
            'external_force': history['external_force'],
            'target_position': history['commanded_position_history'],
            'target_velocity': history['target_velocity_history'],
            'target_orientation': np.radians(history['target_orientation']),
            'terrain_height': history['ground_height_history'],
            'contact_speed': history['contact_speed'],
            'contact_force': history['contact_force'],
            'ground_contact': np.asarray(history['ground_contact'], dtype = bool),
            'physics_advanced': np.asarray(history['physics_advanced'], dtype = bool),
            'flight_stage': np.asarray(history['flight_stage'], dtype = str),
            'landing_phase': np.asarray(history['landing_phase'], dtype = str),
            'waypoints': history['waypoints'],
            'requested_destination': history['requested_destination'],
            'selected_destination': history['selected_destination']
        }

        for name, values in fields.items():
            array = np.asarray(values)
            if array.dtype.kind in 'iuf':
                array = array.astype(np.float64)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f'{backend}/{name} contains nonfinite training data.')
            arrays[f'{backend}_{name}'] = array

        outcomes[backend] = {key: bool(history[key]) for key in ('mission_complete', 'landed', 'landing_failed')}
        outcomes[backend].update({key: history[key] for key in ('stop_reason', 'arrival_time', 'arrival_wait_time',
                                                                'touchdown_speed', 'peak_landing_contact_force', 'landing_position_error')})
        outcomes[backend]['duration'] = float(time[-1])

    # Compare only matching timestamps; keep each complete episode unpadded.
    common_time, simulation_indices, gym_indices = np.intersect1d(arrays['simulation_time_after'], arrays['gym_pybullet_time_after'], return_indices = True)
    arrays['time_after'] = common_time
    arrays['time_before'] = arrays['simulation_time_before'][simulation_indices]
    arrays['state_difference'] = arrays['gym_pybullet_state_after'][gym_indices] - arrays['simulation_state_after'][simulation_indices]
    arrays['final_position_difference'] = comparison['final_position_difference']
    arrays['final_position_distance'] = comparison['final_position_distance']

    paired = comparison['paired_training']
    for name, values in paired.items():
        array = np.asarray(values)
        if array.dtype.kind not in 'biuf' or not np.all(np.isfinite(array)):
            raise ValueError(f'Invalid paired regression data: {name}')
        arrays[name] = array
    arrays['X_columns'] = np.asarray(FEATURE_COLUMNS)
    arrays['X_units'] = np.asarray(FEATURE_UNITS)
    arrays['y_columns'] = np.asarray(TARGET_COLUMNS)
    arrays['y_units'] = np.asarray(TARGET_UNITS)

    # The "true" PyBullet drone for this episode, flattened for use as model
    # inputs (same value on every row of this episode).
    pybullet_parameters = dict(comparison.get('pybullet_parameters') or {})
    randomized_names = sorted(pybullet_parameters)
    reference = config_with(pybullet_parameters)
    names, values = flatten({name: getattr(reference, name) for name in parameters.PYBULLET_PARAMETER_RANGES})
    arrays['pybullet_parameter_columns'] = np.asarray(names)
    arrays['pybullet_parameter_values'] = np.asarray(values, dtype = np.float64)

    def json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported metadata value: {type(value).__name__}')

    history = comparison['simulation']
    terrain = history['terrain']
    if hasattr(terrain, '_heights'):
        arrays['terrain_elevation_grid'] = terrain.base_height + terrain._heights
        arrays['terrain_bounds'] = np.asarray(terrain.plot_bounds, dtype = np.float64)

    metadata = {
        'schema_version': 5, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'time_axes': 'Use <backend>_time_before/after for each complete episode. time_before/after cover only shared timestamps for state_difference; final differences compare endpoints at their respective completion times.',
        'comparison_mode': comparison['mode'], 'state_columns': state_columns,
        'state_units': ['m'] * 3 + ['m/s'] * 3 + ['rad'] * 3 + ['rad/s'] * 3,
        'frames': {'position': 'world', 'velocity': 'world', 'angular_velocity': 'body', 'external_force': 'world'},
        'motor_order': ['+X', '+Y', '-X', '-Y'], 'motor_command_units': 'rad/s',
        'target_yaw_rad': float(np.radians(TARGET_YAW) if target_yaw is None else target_yaw),
        'gains': current_gains() if gains is None else gains,
        'terrain_class': type(history['terrain']).__name__,
        'terrain_base_height': float(terrain.base_height),
        'modes': {key: history[key] for key in ('terrain_mode', 'wind_mode', 'route_planning_mode', 'landing_mode')},
        'parameters': {key: value for key, value in vars(parameters).items() if key.isupper()},
        'pybullet_parameters': {'randomized': randomized_names, 'seed': comparison.get('randomization_seed'),
                                'overrides': pybullet_parameters,
                                'note': 'Custom model uses parameters; PyBullet uses parameters with these overrides.'},
        'outcomes': outcomes,
        'regression': {
            'mode': 'matched_state_one_step', 'reference': 'gym_pybullet',
            'X_columns': FEATURE_COLUMNS, 'X_units': FEATURE_UNITS,
            'y_columns': TARGET_COLUMNS, 'y_units': TARGET_UNITS,
            'frame': 'linear residuals in world frame; angular residuals in body frame',
            'target_definition': '[(v_pybullet_next - v_custom_next) / dt, (w_pybullet_next - w_custom_next) / dt]',
            'reference_physics': 'PyBullet adds motor lag, per-rotor ground effect over the collision surface, rotor drag and body drag (drone_sim/physics/aerodynamics.py); the custom model has none of these.',
            'inputs': 'X starts from the PyBullet state; command = clipped motor command; rotor = actual lagged rotor speeds at the start of the step; clearance = body bottom to ground. pybullet_parameter_values can be appended for a parameter-conditioned model.',
            'acceleration_definition': 'Control-step average delta-velocity/dt; includes discretization and physics-substep effects.',
            'model_use': 'Add the predicted linear (world) and angular (body) acceleration residuals to the custom model before integration, using the same dt and model settings.',
            'collection': 'Before each active Gym mission step, copy its state. Advance an uncorrected custom model from that copy with identical clipped motor commands, world force and dt. PyBullet evolves continuously; only the custom prediction is reset each step.',
            'contact_filter': 'X/y exclude any step where either model touches the ground (contact flag, impact speed or normal force, including PyBullet substeps) and any step whose body bottom is within TRAINING_MIN_CLEARANCE of the ground. Disarmed/frozen mission steps are not collected.',
            'collision_geometry': 'Both models collide the same cylinder (COLLISION_BODY_*) with the same ground: an exact plane for flat terrain, otherwise the shared COLLISION_MESH_* triangle mesh.',
            'excluded_near_ground_rows': int(np.count_nonzero(arrays['paired_near_ground'])),
            'training_rows': int(len(arrays['X'])), 'paired_rows': int(len(arrays['paired_valid_training'])),
            'excluded_contact_rows': int(np.count_nonzero(~arrays['paired_valid_training'])),
            'row_mapping': 'training_row_indices maps X/y and clean acceleration arrays to full paired_* diagnostics.',
            'split_guidance': 'Split by episode (and by randomized drone), not random neighboring rows, to avoid trajectory leakage.'
        },
        'notes': 'simulation_* and gym_pybullet_* are independent evaluation trajectories. state_difference is not a regression target. X/y come only from matched-state steps. Euler differences are raw, without angle wrapping.'
    }
    arrays['metadata_json'] = np.array(json.dumps(metadata, default = json_value))
    directory = OUTPUT_DIRECTORY / 'training'
    directory.mkdir(parents = True, exist_ok = True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = directory / f'comparison_{stamp}_{uuid4().hex[:8]}.npz'
    np.savez_compressed(path, **arrays)
    return path
