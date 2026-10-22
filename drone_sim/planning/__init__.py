"""Terrain-safe takeoff, cruise and landing routes.

pathfinding.py: AStarPathfinder searches safe cruise paths; NoPathError signals failure.
routeplanner.py: RoutePlanner generates spaced takeoff/cruise/landing waypoints.
A* uses one safe altitude per XY node; terrain must remain static.
"""
