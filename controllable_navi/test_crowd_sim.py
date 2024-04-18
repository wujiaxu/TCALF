from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
print(base)
# we need to add base repo to be able to import url_benchmark
# we need to add url_benchmarl to be able to reload legacy checkpoints
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
for fp in [base, base / "controllable_navi"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
import dm_env
from controllable_navi.crowd_sim.crowd_sim import build_crowdworld_task
import numpy as np
from url_benchmark.video import VideoRecorder
from url_benchmark import dmc

recorder = VideoRecorder(base / "controllable_navi", camera_id=0, use_wandb=False)
recorder.enabled = True

crowd_sim = dmc.EnvWrapper(build_crowdworld_task("PointGoalNavi","train"))

n_episodes = 5
for i in range(n_episodes):
    step_ex = crowd_sim.reset()
    recorder.init(crowd_sim)
    while not step_ex.step_type == dm_env.StepType.LAST:
        step_ex = crowd_sim.step(np.array([0.7,-0.5]))
        # print(step_ex.observation.dtype) make sure the obs is float32 matched to torch model
        recorder.record(crowd_sim)
    recorder.save("debug_test_demo_{}.mp4".format(i))

    

# import matplotlib.pyplot as plt
# import shapely
# from shapely.geometry import Polygon, LineString, MultiPolygon,Point,LinearRing

# # Define vertices of the shapes (triangles, rectangles, hexagons)
# triangle_vertices = [(-1, 0), (-2, 0), (-1, 2)]
# rectangle_vertices = [(1, 0), (3, 0), (3, 2), (1, 2)]
# hexagon_vertices = [(2, 0), (3, 1), (2, 2), (1, 2), (0, 1), (1, 0)]
# circle_center = (1, 1)
# circle_radius = 1

# # Create polygons for each shape
# triangle = Polygon(triangle_vertices)
# rectangle = Polygon(rectangle_vertices)
# hexagon = Polygon(hexagon_vertices)
# circle = Point(circle_center).buffer(circle_radius)

# _map_size = 8
# map_boundary = LinearRing( ((-_map_size-1, _map_size+1), 
#                                     (_map_size+1, _map_size+1),
#                                     (_map_size+1, -_map_size-1),
#                                     (-_map_size-1,-_map_size-1)) )
# print(map_boundary.coords.xy)
# print(circle.intersects(map_boundary))
# print(circle.intersects(hexagon))
# x, y = circle.exterior.xy
# plt.plot(x, y)
# # print(triangle.contains(rectangle))
# # # Get boundary polygons
# boundary_polygon = triangle.union(rectangle).union(hexagon)
# if isinstance(boundary_polygon,MultiPolygon):
#     for i, polygon in enumerate(boundary_polygon.geoms):
#         x, y = polygon.exterior.xy
#         plt.plot(x, y,label='Boundary Polygon'+str(i))
#         plt.xlabel('X')
#         plt.ylabel('Y')
#     plt.legend()
# else:
#     # Plot the boundary polygon
#     x, y = boundary_polygon.exterior.xy
#     plt.plot(x, y)
#     plt.xlabel('X')
#     plt.ylabel('Y')
# plt.title('Boundary Polygon')
# plt.grid(True)
# plt.show()