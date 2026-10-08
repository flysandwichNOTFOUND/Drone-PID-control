"""Regression tests for controller behavior and full missions.

Run with: python -m unittest -v
"""
import unittest
from unittest.mock import patch
import numpy as np
import main
import auto_tune_pid as tuner
from attitude import integrate_orientation
from controllerV1 import OrientationPIDController, PositionPIDController
from testing import DroneState
from navigation import Navigation
from terrain import FlatTerrain, HillTerrain, SlopedTerrain, LandscapeTerrain
from landing import LandingController
from routeplanner import RoutePlanner


class ControllerTests(unittest.TestCase):
    def test_tilt_commands_limit_step_and_reversal_rates(self):

        rate = np.radians(10)
        controller = PositionPIDController([1, 0, 0], [1, 0, 0], 0.5,
                                           tilt_smoothing_time = 0.15, max_tilt_rate = rate)
        state = DroneState()
        previous = np.zeros(2)
        for target in [[2, -2]] * 150 + [[-2, 2]] * 150:
            command = controller.calculate_target_orientation(target, 0.7, state, 0.01)
            self.assertTrue(np.all(np.abs(command[:2] - previous) <= rate * 0.01 + 1e-12))
            self.assertTrue(np.all(np.abs(command[:2]) <= 0.5))
            self.assertEqual(command[2], 0.7)
            previous = command[:2].copy()
        self.assertTrue(np.all(previous < 0))

    def test_tilt_smoothing_matches_time_constant_across_time_steps(self):

        for dt in [0.01, 0.025]:
            controller = PositionPIDController([1, 0, 0], [1, 0, 0], 0.5,
                                               tilt_smoothing_time = 0.15)
            for _ in range(round(0.3 / dt)):
                command = controller.calculate_target_orientation([0.2, -0.2], 0, DroneState(), dt)
            np.testing.assert_allclose(command[:2], 0.2 * (1 - np.exp(-2)), atol = 1e-12)

    def test_position_integrates_once(self):

        controller = PositionPIDController([0, 1, 0], [0, 1, 0], 0.5)
        controller.calculate_target_orientation([0.1, -0.2], 0, DroneState(), 0.01)
        self.assertAlmostEqual(controller.x_integral, 0.001)
        self.assertAlmostEqual(controller.y_integral, -0.002)

    def test_position_command_follows_world_direction_at_multiple_headings(self):

        for heading in [0, np.pi / 2, -np.pi / 2, np.pi, 0.6]:
            for target in [[0.1, 0], [0, 0.1]]:
                state = DroneState()
                state.orientation[2] = heading
                controller = PositionPIDController([1, 0, 0], [1, 0, 0], 0.5)
                state.orientation = controller.calculate_target_orientation(target, heading, state, 0.01)
                state.motor_speeds.fill(main.HOVER_SPEED)
                force = main.create_drone().net_force(state)[:2]
                self.assertGreater(np.dot(force, target), 0)
                # Cross-axis acceleration is negligible for small tilt commands.
                self.assertLess(abs(force[1 if target[0] else 0]), 0.005)

    def test_yaw_uses_shortest_turn(self):

        state = DroneState()
        state.orientation[2] = np.radians(179)
        controller = OrientationPIDController([1, 0, 0], [1, 0, 0], [1, 0, 0], 20, 1000)
        correction = controller.calculate_corrections([0, 0, np.radians(-179)], state, 0.01)
        self.assertAlmostEqual(correction[2], np.radians(2))

    def test_integral_does_not_wind_up_at_correction_limit(self):

        controller = OrientationPIDController([20, 1, 0], [1, 0, 0], [1, 0, 0], 1, 1000)
        for _ in range(100):
            controller.calculate_corrections([1, 0, 0], DroneState(), 0.01)
        self.assertEqual(controller.integral_error[0], 0)
        controller.calculate_corrections([0.01, 0, 0], DroneState(), 0.01)
        self.assertGreater(controller.integral_error[0], 0)

    def test_integral_does_not_wind_up_at_motor_limit(self):

        controller = OrientationPIDController([1, 1, 0], [1, 0, 0], [1, 0, 0], 20, 1000)
        for _ in range(100):
            speeds = controller.calculate_motor_speed(np.full(4, 1000.0), [0.1, 0, 0], DroneState(), 0.01)
        self.assertEqual(controller.integral_error[0], 0)
        self.assertTrue(np.all((speeds >= 0) & (speeds <= 1000)))

    def test_input_validation(self):

        for dt in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                PositionPIDController([1, 0, 0], [1, 0, 0], 0.5).calculate_target_orientation(
                    [0, 0], 0, DroneState(), dt)
        with self.assertRaises(ValueError):
            Navigation([], 0, 0.2, 0.15)
        with self.assertRaises(ValueError):
            Navigation([[0, 0, 1]], -1, 0.2, 0.15)


class DynamicsTests(unittest.TestCase):
    def test_body_yaw_rotation_changes_pitch_when_rolled(self):

        # At roll=90 degrees, a positive body yaw rotation lowers world pitch.
        result = integrate_orientation([np.pi / 2, 0, 0], [0, 0, 1], 0.1)
        np.testing.assert_allclose(result, [np.pi / 2, -0.1, 0], atol = 1e-12)

    def test_ground_contact_and_takeoff(self):

        state = DroneState()
        state.position[2] = 0.01
        state.velocity[2] = -2
        drone = main.create_drone(FlatTerrain(main.GROUND_HEIGHT))
        drone.update_state(state, 0.01)
        self.assertEqual(state.position[2], main.GROUND_HEIGHT)
        self.assertEqual(state.velocity[2], 0)
        state.motor_speeds.fill(main.MAX_MOTOR_SPEED)
        drone.update_state(state, 0.01)
        self.assertGreater(state.position[2], main.GROUND_HEIGHT)


class MissionTests(unittest.TestCase):
    def test_mission_at_zero_and_ninety_degree_yaw(self):

        for heading in [0, np.pi / 2]:
            state, data = main.run_simulation(verbose = False, target_yaw = heading, include_landing = False)
            self.assertTrue(data['mission_complete'])
            self.assertLess(np.linalg.norm(state.position - data['waypoints'][-1]), main.ARRIVAL_TOLERANCE)
            self.assertLess(np.linalg.norm(state.velocity), main.SPEED_TOLERANCE)
            self.assertTrue(np.all(np.isfinite(data['position'])))
            targets = np.asarray(data['target_position_history'])
            np.testing.assert_allclose(targets, data['waypoints'][data['waypoint_index']])

    def test_landing_disarms_only_at_ground(self):

        state, data = main.run_simulation(include_landing = True, duration = 70, verbose = False,
                                          terrain = FlatTerrain(main.GROUND_HEIGHT))
        self.assertTrue(data['landed'])
        self.assertTrue(data['mission_complete'])
        self.assertEqual(state.position[2], main.GROUND_HEIGHT)
        self.assertGreaterEqual(min(data['height']), main.GROUND_HEIGHT)
        np.testing.assert_array_equal(state.motor_speeds, np.zeros(4))
        speeds = np.asarray(data['motor_speed'])
        disarmed = np.all(speeds == 0, axis = 1)
        self.assertTrue(np.all(np.asarray(data['height'])[disarmed] == main.GROUND_HEIGHT))

    def test_tuner_uses_main_settings_and_environment(self):

        for key, value in main.current_gains().items():
            np.testing.assert_array_equal(tuner.initial_gain_dictionary()[key], value)
        self.assertEqual(tuner.MAX_TILT_DEGREES, main.MAX_TILT_DEGREES)
        with patch.object(main, 'control_step', wraps = main.control_step) as step:
            history = tuner.run_simulation(main.current_gains(), 0.02, target_position = [1, 0])
        self.assertIsNotNone(history)
        self.assertEqual(step.call_count, 2)
        environment = step.call_args.args[3]
        self.assertEqual(environment.wind_mode, main.WIND_MODE)
        self.assertEqual(environment.wind_force_coefficient, main.WIND_FORCE_COEFFICIENT)

    def test_tuner_rejects_worse_recommendations(self):

        candidate = main.current_gains()
        candidate['scores'] = {}
        with patch.object(tuner, 'run_simulation', return_value = {}), \
             patch.object(tuner, 'combined_validation_score', side_effect = [1.0, 2.0]), \
             patch.object(tuner, 'validate_route', side_effect = [
                 {'score': 1.0, 'mission_complete': True},
                 {'score': 0.5, 'mission_complete': True}]):
            selected, _, _, validation = tuner.validate_recommendations(candidate)
        self.assertFalse(validation['candidate_accepted'])
        np.testing.assert_array_equal(selected['height'], main.current_gains()['height'])

    def test_retuning_accepts_baseline_outside_previous_search_bounds(self):

        initial = np.array([200.0, 0.0, 50.0])
        bounds = np.array([[20.0, 100.0], [0.0, 2.0], [15.0, 80.0]])
        with patch.object(tuner, 'TRIALS_PER_CONTROLLER', 3):
            selected, score = tuner.search_gains(
                'Regression', initial, bounds,
                lambda value: np.sum((value - initial)**2), np.random.default_rng(1))
        np.testing.assert_array_equal(selected, initial)
        self.assertEqual(score, 0.0)


class TerrainLandingTests(unittest.TestCase):
    def test_landscape_gradient_matches_elevation(self):

        terrain = LandscapeTerrain()
        for x, y in [(0, 0), (8, 7), (-4, 8), (3, 5)]:
            epsilon = 1e-5
            finite_difference = np.array([
                (terrain.height(x + epsilon, y) - terrain.height(x - epsilon, y)) / (2 * epsilon),
                (terrain.height(x, y + epsilon) - terrain.height(x, y - epsilon)) / (2 * epsilon)])
            np.testing.assert_allclose(terrain.gradient(x, y), finite_difference, atol = 1e-9)

    def test_landscape_route_height_covers_between_sample_peaks(self):

        terrain = LandscapeTerrain()
        start, goal = np.array([-9, -5]), np.array([16, 14])
        points = start + np.linspace(0, 1, 10000)[:, None] * (goal - start)
        dense_max = np.max(terrain.height(points[:, 0], points[:, 1]))
        self.assertGreaterEqual(terrain.maximum_height_along(start, goal), dense_max)
        np.testing.assert_array_equal(terrain.height(points[:, 0], points[:, 1]),
                                      LandscapeTerrain(seed = 17).height(points[:, 0], points[:, 1]))

    def test_default_landscape_landing(self):

        state, data = main.run_simulation(terrain = LandscapeTerrain(), include_landing = True, verbose = False)
        self.assertTrue(data['landed'])
        self.assertFalse(data['landing_failed'])
        self.assertGreaterEqual(min(data['clearance']), -1e-10)
        self.assertAlmostEqual(state.position[2], float(data['terrain'].height(*state.position[:2])))

    def test_landing_matches_local_surface_on_different_terrain(self):

        cases = [(FlatTerrain(0), [1.5, -0.5, 2]),
                 (FlatTerrain(2), [1.5, -0.5, 4]),
                 (HillTerrain(), [5, 4, 2]),
                 (SlopedTerrain(1, [0.1, 0.05]), [2, 1, 3])]
        for terrain, goal in cases:
            with self.subTest(terrain = type(terrain).__name__, goal = goal):
                state, data = main.run_simulation(terrain = terrain, goal_position = goal,
                                                  include_landing = True, duration = 70, verbose = False)
                self.assertTrue(data['landed'])
                self.assertTrue(data['mission_complete'])
                self.assertFalse(data['landing_failed'])
                self.assertAlmostEqual(state.position[2], float(terrain.height(*state.position[:2])))
                self.assertGreaterEqual(min(data['clearance']), -1e-10)
                self.assertLess(data['touchdown_speed'], 0.2)
                self.assertLess(np.linalg.norm(state.position[:2] - goal[:2]), 0.08)
                np.testing.assert_array_equal(state.motor_speeds, np.zeros(4))
                self.assertIn('flare', data['landing_phase'])
                self.assertEqual(data['landing_phase'][-1], 'landed')

    def test_route_climbs_above_intervening_hill_before_crossing(self):

        terrain = HillTerrain()
        start = [0, 0, float(terrain.height(0, 0))]
        planner = RoutePlanner(start, [10, 8, 2], 0, 2, 1, terrain = terrain)
        waypoints = np.asarray(planner.generate_route(include_landing = True))
        np.testing.assert_allclose(planner.cruise_altitude, 5.0)
        crossing = waypoints[(waypoints[:, 0] > 0) & (waypoints[:, 0] < 10)]
        self.assertTrue(np.all(crossing[:, 2] >= terrain.height(crossing[:, 0], crossing[:, 1]) + 2))
        self.assertAlmostEqual(waypoints[-1, 2], float(terrain.height(10, 8)))

    def test_steep_landing_site_is_rejected(self):

        with self.assertRaisesRegex(ValueError, 'too steep'):
            main.run_simulation(terrain = HillTerrain(), goal_position = [6.6, 4, 2],
                                include_landing = True, verbose = False)
        with self.assertRaisesRegex(ValueError, 'too steep'):
            LandingController(SlopedTerrain(0, [0.5, 0]), [0, 0])

    def test_descent_pauses_until_horizontally_aligned(self):

        controller = LandingController(FlatTerrain(), [1, 0])
        state = DroneState()
        state.position[2] = 2
        target, velocity = controller.target(state, 0.01)
        self.assertEqual(controller.phase, 'align')
        self.assertEqual(target[2], 2)
        self.assertEqual(velocity, 0)
        state.position[0] = 1
        target, velocity = controller.target(state, 0.01)
        self.assertEqual(controller.phase, 'descend')
        self.assertLess(target[2], 2)
        self.assertLess(velocity, 0)
        state.position[0] = 1.3
        target, velocity = controller.target(state, 0.01)
        self.assertEqual(controller.phase, 'align')
        self.assertEqual(velocity, 0)

    def test_hard_contact_is_not_reported_as_successful_landing(self):

        controller = LandingController(FlatTerrain(), [0, 0])
        state = DroneState()
        state.ground_contact = True
        state.contact_speed = 1.0
        self.assertFalse(controller.check_touchdown(state))
        self.assertTrue(controller.failed)
        self.assertEqual(controller.phase, 'hard_landing')


class PathFollowerTests(unittest.TestCase):
    def test_continuous_reference_obeys_limits_and_crosses_waypoints(self):

        from navigation import PathFollower

        follower = PathFollower([[0, 0, 2], [1, 0, 2], [2, 0, 2], [4, 0, 2]],
                                [0, 0, 0], 0.02, 0.05)
        state = DroneState()
        previous_velocity = np.zeros(3)
        previous_position = state.position.copy()
        crossing_speeds = []
        dt = 0.01
        for step in range(2500):
            target = follower.update(state, dt)
            velocity = follower.target_velocity.copy()
            self.assertLessEqual(np.linalg.norm(velocity), 1.2 + 1e-9)
            self.assertLessEqual(np.linalg.norm(velocity - previous_velocity) / dt, 0.8 + 1e-8)
            self.assertLessEqual(abs(velocity[2]), 0.8 + 1e-9)
            self.assertLessEqual(np.linalg.norm(target - previous_position), 1.2 * dt + 1e-9)
            for x in [1, 2]:
                if previous_position[0] < x <= target[0]:
                    crossing_speeds.append(np.linalg.norm(velocity))
            state.position = target.copy()
            state.velocity = velocity.copy()
            previous_position = target.copy()
            previous_velocity = velocity.copy()
            if follower.mission_complete:
                break
        self.assertTrue(follower.mission_complete)
        self.assertEqual(len(crossing_speeds), 2)
        self.assertTrue(all(speed > 0.8 for speed in crossing_speeds))
        np.testing.assert_allclose(state.position, [4, 0, 2])
        np.testing.assert_allclose(state.velocity, 0)

    def test_stalled_drone_keeps_reference_nearby(self):

        from navigation import PathFollower

        follower = PathFollower([[10, 0, 0]], [0, 0, 0], 0.02, 0.05)
        state = DroneState()
        for step in range(2000):
            follower.update(state, 0.01)
        self.assertFalse(follower.mission_complete)
        # Jerk-limited slowing takes more distance than an abrupt clock stop.
        self.assertLess(follower.target_position[0], 7.0)
        self.assertLess(np.linalg.norm(follower.target_velocity), 0.02)
        for step in range(4000):
            follower.update(state, 0.01)
        self.assertAlmostEqual(np.linalg.norm(follower.target_velocity), 0.0)

    def test_s_curve_acceleration_and_jerk_limits(self):

        from navigation import PathFollower

        for goal, dt, stalled in [([0.03, 0, 0], 0.01, False),
                                  ([10, 0, 0], 0.02, False),
                                  ([2, 1, 3], 0.01, False),
                                  ([0, 0, -2], 0.01, False),
                                  ([10, 0, 0], 0.01, True)]:
            with self.subTest(goal = goal, dt = dt, stalled = stalled):
                follower = PathFollower([goal], [0, 0, 0], 0.02, 0.05)
                state = DroneState()
                previous_acceleration = np.zeros(3)
                for step in range(round(40 / dt)):
                    target = follower.update(state, dt)
                    acceleration = follower.target_acceleration.copy()
                    self.assertLessEqual(np.linalg.norm(acceleration), 0.8 + 1e-9)
                    self.assertLessEqual(np.linalg.norm(acceleration - previous_acceleration) / dt, 1.0 + 1e-8)
                    self.assertLessEqual(np.linalg.norm(follower.target_velocity), 1.2 + 1e-9)
                    self.assertLessEqual(follower.target_velocity[2], 0.8 + 1e-9)
                    self.assertGreaterEqual(follower.target_velocity[2], -0.4 - 1e-9)
                    previous_acceleration = acceleration
                    if not stalled:
                        state.position = target.copy()
                        state.velocity = follower.target_velocity.copy()
                    if follower.mission_complete:
                        break
                self.assertEqual(follower.mission_complete, not stalled)
                if not stalled:
                    np.testing.assert_allclose(state.position, goal, atol = 1e-10)
                    np.testing.assert_allclose(acceleration, 0)

    def test_duplicate_stationary_waypoints(self):

        from navigation import PathFollower

        follower = PathFollower([[0, 0, 0], [0, 0, 0]], [0, 0, 0], 0.02, 0.05)
        follower.update(DroneState(), 0.01)
        self.assertTrue(follower.mission_complete)
        follower.reset()
        self.assertFalse(follower.mission_complete)


if __name__ == '__main__':
    unittest.main()

