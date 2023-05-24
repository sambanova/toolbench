"""
Copyright 2021 Google LLC. SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

prompt_tabletop_ui = """
# Python 2D robot control script
import numpy as np
from env_utils import get_obj_pos, detect_obj, get_obj_names
from plan_utils import parse_position, parse_obj_name
from ctrl_utils import put_first_on_second, stack_objects_in_order, point_gripper_to

objects = ['cyan block', 'yellow block', 'brown block', 'green bowl']
# put the yellow one the green thing.
put_first_on_second('yellow block', 'green bowl')
objects = ['pink block', 'graNy block', 'orange block']
# move the pinkish colored block on the bottom side.
bottom_side_pos = parse_position('the bottom side')
put_first_on_second('pink block', bottom_side_pos)
objects = ['orange bowl', 'red block', 'purple bowl', 'blue block', 'blue bowl', 'orange block', 'purple block', 'red bowl']
# put the blocks into bowls with matching colors.
# put_first_on_second('orange block', 'orange bowl')
# put_first_on_second('red block', 'red bowl')
# put_first_on_second('purple block', 'purple bowl')
# put_first_on_second('blue block', 'blue bowl')
objects = ['orange bowl', 'red block',  'purple block', 'orange block', 'purple bowl', 'red bowl']
# point gripper to any three points in a horizontal line in the middle
n_points = 3
place_positions = parse_position(f'a horizontal line in the middle with {n_points} points')
for line_position in place_positions:
    point_gripper_to(line_position)
objects = ['orange bowl', 'red block',  'purple block', 'orange block', 'purple bowl', 'red bowl']
# point gripper to k points in a diagonal line, where k is the number of bowls
bowl_names = parse_obj_name('the bowls', f'objects = {get_obj_names()}')
k = len(bowl_names)
place_positions = parse_position(f'a diagonal line with {k} points')
for line_position in place_positions:
    point_gripper_to(line_position)
objects = ['purple block', 'cyan bowl', 'blue block', 'cyan block', 'purple bowl', 'blue bowl']
# move the blue block in between the cyan block and purple bowl.
target_pos = parse_position('a point between the cyan block and purple bowl')
put_first_on_second('blue block', target_pos)
objects = ['purple block', 'cyan bowl', 'blue block', 'cyan block', 'purple bowl', 'blue bowl']
# move the block closest to the purple bowl to the cyan bowl.
closest_block_name = parse_obj_name('the block closest to the purple bowl', f'objects = {get_obj_names()}')
put_first_on_second(closest_block_name, 'cyan bowl')
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl']
# point gripper to the corner closest to the yellow block.
closest_corner_pos = parse_position('the corner closest to the yellow block')
point_gripper_to(closest_corner_pos)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the left most block to the green bowl.
left_block_name = parse_obj_name('left most block', f'objects = {get_obj_names()}')
put_first_on_second(left_block_name, 'green bowl')
objects = ['brown bowl', 'green block', 'brown block', 'blue bowl', 'blue block', 'green bowl']
# move the brown bowl to the closest side.
closest_side_position = parse_position('the side closest to the brown bowl')
put_first_on_second('brown bowl', closest_side_position)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the green block in the bowl closest to the middle.
middle_bowl_name = parse_obj_name('the bowl closest to the middle', f'objects = {get_obj_names()}')
put_first_on_second('green block', middle_bowl_name)
objects = ['brown bowl', 'green block', 'brown block', 'blue bowl', 'blue block', 'green bowl']
# place the blue block in the empty bowl.
empty_bowl_name = parse_obj_name('the empty bowl', f'objects = {get_obj_names()}')
put_first_on_second('blue block', empty_bowl_name)
objects = ['brown bowl', 'blue bowl', 'blue block', 'red block', 'brown block', 'red bowl']
# stack the blocks that are close to the red bowl.
close_block_names = parse_obj_name('blocks that are close to the red bowl', f'objects = {get_obj_names()}')
stack_objects_in_order(object_names=close_block_names)
objects = ['red block', 'red bowl', 'blue bowl', 'blue block']
# stack the blocks on the top most bowl.
bowl_name = parse_obj_name('top most bowl', f'objects = {get_obj_names()}')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
object_names = [bowl_name] + block_names
stack_objects_in_order(object_names=object_names)
objects = ['yellow bowl', 'red block', 'yellow block', 'red bowl', 'green plate', 'orange plate']
# move objects from the green plate to the red bowl.
object_names = parse_obj_name('objects from the green plate', f'objects = {get_obj_names()}')
for object_name in object_names:
    put_first_on_second(object_name, 'red bowl')
objects = ['yellow bowl', 'red block', 'yellow block', 'red bowl', 'green plate', 'orange plate']
# move the red bowl the left of the blocks.
left_pos = parse_position('a point left of the blocks')
put_first_on_second('red bowl', left_pos)
""".strip()

prompt_parse_obj_name = """
import numpy as np
from env_utils import get_obj_pos, get_color
from utils import get_obj_positions_np, get_box_area

objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block', 'purple block']
# the block closest to the purple bowl.
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=get_obj_pos('purple bowl'))
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the left most block.
block_names = ['green block', 'brown block', 'blue block']
block_positions = get_obj_positions_np(block_names)
left_block_idx = np.argsort(block_positions[:, 0])[0]
left_block_name = block_names[left_block_idx]
ret_val = left_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the third bowl from the top.
bowl_names = ['brown bowl', 'green bowl', 'blue bowl']
bowl_positions = get_obj_positions_np(bowl_names)
top_bowl_idx = np.argsort(bowl_positions[:, 1])[-3]
top_bowl_name = bowl_names[top_bowl_idx]
ret_val = top_bowl_name
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the largest fruit.
fruit_names = ['banana', 'apple']
fruit_bbox = [get_bbox(name) for name in fruit_names]
fruit_sizes = [get_box_area(bbox) for bbox in fruit_bbox]
ret_val = fruit_names[np.argmax(fruit_sizes)]
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the blocks.
ret_val = ['brown block', 'blue block']
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# a fruit that's not the apple
fruit_names = ['banana', 'apple']
for fruit_name in fruit_names:
    if fruit_name != 'apple':
        ret_val = fruit_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the object on the green bowl.
for obj_name in objects:
    if obj_name != 'green bowl':
        if np.linalg.norm(get_obj_pos('green bowl') - get_obj_pos(obj_name)) < 0.05:
            ret_val = obj_name
            break
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the brown block.
ret_val = 'brown block'
""".strip()

prompt_parse_position = """
import numpy as np
from shapely.geometry import *
from shapely.affinity import *
from env_utils import denormalize_xy, parse_obj_name, get_obj_names, get_obj_pos

# a 30cm horizontal line in the middle with 3 points.
middle_pos = denormalize_xy([0.5, 0.5]) 
start_pos = middle_pos + [-0.3/2, 0]
end_pos = middle_pos + [0.3/2, 0]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=3)
ret_val = points
# a 20cm vertical line near the right with 4 points.
middle_pos = denormalize_xy([1, 0.5]) 
start_pos = middle_pos + [0, -0.2/2]
end_pos = middle_pos + [0, 0.2/2]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=4)
ret_val = points
# a diagonal line from the top left to the bottom right corner with 5 points.
top_left_corner = denormalize_xy([0, 1])
bottom_right_corner = denormalize_xy([1, 0])
line = make_line(start=top_left_corner, end=bottom_right_corner)
points = interpolate_pts_on_line(line=line, n=5)
ret_val = points
# a triangle with size 10cm with 3 points.
polygon = make_triangle(size=0.1, center=denormalize_xy([0.5, 0.5]))
points = get_points_from_polygon(polygon)
ret_val = points
# the corner closest to the sun colored block.
block_name = parse_obj_name('the sun colored block', f'objects = {get_obj_names()}')
corner_positions = np.array([denormalize_xy(pos) for pos in [[0, 0], [0, 1], [1, 1], [1, 0]]])
closest_corner_pos = get_closest_point(points=corner_positions, point=get_obj_pos(block_name))
ret_val = closest_corner_pos
# the side farthest from the right most bowl.
bowl_name = parse_obj_name('the right most bowl', f'objects = {get_obj_names()}')
side_positions = np.array([denormalize_xy(pos) for pos in [[0.5, 0], [0.5, 1], [1, 0.5], [0, 0.5]]])
farthest_side_pos = get_farthest_point(points=side_positions, point=get_obj_pos(bowl_name))
ret_val = farthest_side_pos
# a point above the third block from the bottom.
block_name = parse_obj_name('the third block from the bottom', f'objects = {get_obj_names()}')
ret_val = get_obj_pos(block_name) + [0.1, 0]
# a point 10cm left of the bowls.
bowl_names = parse_obj_name('the bowls', f'objects = {get_obj_names()}')
bowl_positions = get_all_object_positions_np(obj_names=bowl_names)
left_obj_pos = bowl_positions[np.argmin(bowl_positions[:, 0])] + [-0.1, 0]
ret_val = left_obj_pos
# the bottom side.
bottom_pos = denormalize_xy([0.5, 0])
ret_val = bottom_pos
# the top corners.
top_left_pos = denormalize_xy([0, 1])
top_right_pos = denormalize_xy([1, 1])
ret_val = [top_left_pos, top_right_pos]
""".strip()

prompt_parse_question = """
""".strip()

prompt_transform_shape_pts = """
""".strip()

prompt_fgen = """
import numpy as np
from shapely.geometry import *
from shapely.affinity import *

from env_utils import get_obj_pos, get_bbox, get_obj_names
from ctrl_utils import put_first_on_second

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: pt = get_pt_to_the_left(pt, dist).
def get_pt_to_the_left(pt, dist):
    return pt + [-dist, 0]

# define function: pt = get_pt_to_the_top(pt, dist).
def get_pt_to_the_top(pt, dist):
    return pt + [0, dist]

# define function line = make_line_by_length(length=x).
def make_line_by_length(length):
  line = LineString([[0, 0], [length, 0]])
  return line

# define function: line = make_vertical_line_by_length(length=x).
def make_vertical_line_by_length(length):
  line = make_line_by_length(length)
  vertical_line = rotate(line, 90)
  return vertical_line

# define function: pt = interpolate_line(line, t=0.5).
def interpolate_line(line, t):
  pt = line.interpolate(t, normalized=True)
  return np.array(pt.coords[0])

# example: scale a line by 2.
line = make_line_by_length(1)
new_shape = scale(line, xfact=2, yfact=2)

# example: put object1 on top of object0.
put_first_on_second('object1', 'object0')

# example: get two corners of object bbox.
get_bbox('object').reshape(2, 2)

# example: get the position of the first object.
obj_names = get_obj_names()
pos_2d = get_obj_pos(obj_names[0])
""".strip()
