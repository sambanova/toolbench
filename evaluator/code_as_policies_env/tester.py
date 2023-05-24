import re
import numpy as np
from evaluator.code_as_policies_env.LMP_env import PickPlaceEnv, CORNER_POS
from evaluator.code_as_policies_env.initial_envs import VALID_POSITIONS

INSTRUCTIONS = {
    "SI": {
        "0": "Pick up the {block} and place it on the {object1}",
        "1": "Stack all the blocks",
        "2": "Put all the blocks on the {corner_or_side}",
        "3": "Put the blocks in the {bowl}",
        "4": "Put all the blocks in the bowls with matching colors",
        "5": "Pick up all blocks that are located to the {direction} of the {bowl} and place them on the {corner_or_side}",
        "6": "Pick up the block {distance} to the {bowl} and place it on the {corner_or_side}",
        "7": "Pick up the {nth} block from the {direction} and place it on the {corner_or_side}",  # sample 4 or more blocks
    },
    "UI": {
        "0": "Put all the blocks in different corners",  # no more than 4 blocks
        "1": "Put the blocks in the bowls with mismatched colors",  # same num. of blocks and bowls (not necessary, but if not, what's the best way to evaluate)
        "2": "Stack all the blocks on the {corner_or_side}",
        "3": "Pick up the {block} and place it {magnitude} to the {direction} of the {bowl}",
        "4": "Pick up the {block} and place it in the corner {distance} to the {bowl}",
        "5": "Put all the blocks in a {line} line",
        "6": "Put all the blocks in different sides",
    },
}

ATTRIBUTES = {
    "block": [
        "blue block",
        "red block",
        "green block",
        "orange block",
        "yellow block",
        "pink block",
        "cyan block",
        "brown block",
    ],
    "bowl": [
        "blue bowl",
        "red bowl",
        "green bowl",
        "orange bowl",
        "yellow bowl",
        "pink bowl",
        "cyan bowl",
        "brown bowl",
    ],
    "corner_or_side": [
        "left side",
        "top left corner",
        "top side",
        "top right corner",
        "bottom right corner",
        "bottom side",
        "bottom left corner",
        "right side",
    ],
    "direction": ["top", "left", "bottom", "right"],
    "distance": ["closest", "farthest"],
    "magnitude": ["10 cm", "0.2 m"],
    "nth": ["first", "second", "third", "fourth"],
    "line": ["horizontal", "vertical", "diagonal"],
}

THRESHOLD = 0.05


# test utils
def sample_position_dict(num_blocks, num_bowls):
    return np.random.choice(VALID_POSITIONS[f"({num_blocks}, {num_bowls})"])


def get_all_blocks_or_bowls_pos(env, object_type="block"):
    object_list = env.object_list
    all_block_names = [obj for obj in object_list if object_type in obj]
    all_block_pos = np.array([env.get_obj_pos(obj) for obj in all_block_names])
    return all_block_pos, all_block_names


def get_matches(env):
    object_list = env.object_list
    matches = {}
    all_block_list = [obj for obj in object_list if "block" in obj]
    all_bowl_list = [obj for obj in object_list if "bowl" in obj]
    for block in all_block_list:
        color = block.split(" ")[0]
        if f"{color} bowl" in all_bowl_list:
            matches[block] = f"{color} bowl"
    return matches


def get_corner_positions():
    corners = []
    for name, pos in CORNER_POS.items():
        if "corner" in name:
            corners.append(pos)
    return corners


def get_side_positions():
    sides = []
    for name, pos in CORNER_POS.items():
        if "side" in name:
            sides.append(pos)
    return sides


def get_closest_or_farthest_point(points, point, distance_type):
    if distance_type == "closest":
        point_idx = np.argmin(np.linalg.norm(points - point, axis=1))
    elif distance_type == "farthest":
        point_idx = np.argmax(np.linalg.norm(points - point, axis=1))
    else:
        raise KeyError
    target_point = points[point_idx]
    return target_point, point_idx


def sample_instruction(instruction_template, block_list, bowl_list):
    used_attributes = re.findall("\{(.*?)\}", instruction_template)
    used_attributes.sort()
    vars_dict = {}
    for attribute in used_attributes:
        if attribute == "bowl":
            vars_dict["bowl"] = np.random.choice(bowl_list)
        elif attribute == "block":
            vars_dict["block"] = np.random.choice(block_list)
        elif attribute == "object1":
            obj_set = set(bowl_list + block_list)
            if "bowl" in vars_dict:
                obj_set.remove(vars_dict["bowl"])
            if "block" in vars_dict:
                obj_set.remove(vars_dict["block"])
            vars_dict["object1"] = np.random.choice(list(obj_set))
        else:
            vars_dict[attribute] = np.random.choice(ATTRIBUTES[attribute])
    for attribute, value in vars_dict.items():
        locals()[attribute] = value
    instruction = instruction_template.format(**locals())
    return instruction, vars_dict


class Tester:
    def __init__(self, task):
        task_group = task.split("_")[0]
        task_id = task.split("_")[1]
        self.instruction_template = INSTRUCTIONS[task_group][task_id]
        self.env = PickPlaceEnv(render=True)

        n_blocks_range = [1, 5]
        n_bowls_range = [1, 5]
        self.n_blocks_generator = lambda: np.random.choice(range(*n_blocks_range))
        self.n_bowls_generator = lambda: np.random.choice(range(*n_bowls_range))

    def update_vars_by_initial_state(self, instruction_vars_dict):
        pass

    def sample_instruction_and_reset_env(self):
        block_list = np.random.choice(
            ATTRIBUTES["block"],
            size=self.n_blocks_generator(),
            replace=False,
        ).tolist()
        bowl_list = np.random.choice(
            ATTRIBUTES["bowl"],
            size=self.n_bowls_generator(),
            replace=False,
        ).tolist()
        obj_list = block_list + bowl_list

        self.position_dict = sample_position_dict(len(block_list), len(bowl_list))
        _ = self.env.reset_with_positions(obj_list, self.position_dict)
        instruction, instruction_vars_dict = sample_instruction(
            self.instruction_template, block_list, bowl_list
        )
        self.update_vars_by_initial_state(instruction_vars_dict)
        self.vars_dict = instruction_vars_dict

        configs = {}
        configs["block_list"] = block_list
        configs["bowl_list"] = bowl_list
        configs["position_dict"] = self.position_dict
        configs["instruction"] = instruction
        configs["instruction_vars_dict"] = instruction_vars_dict
        return configs

    def reset_env_from_configs(self, configs):
        obj_list = configs["block_list"] + configs["bowl_list"]
        self.position_dict = configs["position_dict"]
        _ = self.env.reset_with_positions(obj_list, self.position_dict)

        instruction = configs["instruction"]
        self.vars_dict = configs["instruction_vars_dict"]
        return instruction

    def check_final_state(self):
        raise NotImplementedError


# SI_0: Pick up the {block} and place it on the {object1}
class TesterSI_0(Tester):
    def __init__(self):
        super().__init__("SI_0")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: np.random.choice(range(2, 5))
        self.n_bowls_generator = lambda: np.random.choice(range(2, 5))

    def check_final_state(self, threshold=THRESHOLD):
        block = self.vars_dict["block"]
        object1 = self.vars_dict["object1"]
        block_pos = self.env.get_obj_pos(block)
        object1_pos = self.env.get_obj_pos(object1)

        if "bowl" in object1:
            threshold = THRESHOLD
        is_successful = (
            np.linalg.norm(block_pos[:2] - object1_pos[:2]) < threshold
            and block_pos[2] > object1_pos[2]
        )
        final_states = {
            "initial positions": self.position_dict,
            "final block_pos": str(block_pos),
            "final object1_pos": str(object1_pos),
            "distance": str(np.linalg.norm(block_pos[:2] - object1_pos[:2])),
        }
        return is_successful, final_states


# SI_1: Stack all the blocks
class TesterSI_1(Tester):
    def __init__(self):
        super().__init__("SI_1")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: np.random.choice(range(2, 5))

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        all_block_pos_x = all_block_pos[:, 0]
        all_block_pos_y = all_block_pos[:, 1]
        is_successful = (
            max(all_block_pos_x) - min(all_block_pos_x) < threshold
            and max(all_block_pos_y) - min(all_block_pos_y) < threshold
        )
        final_states = {
            "initial positions": self.position_dict,
            "final block positions: ": str(all_block_pos),
            "max_x: ": str(max(all_block_pos_x)),
            "min_x: ": str(min(all_block_pos_x)),
            "delta_x: ": str(max(all_block_pos_x) - min(all_block_pos_x)),
            "max_y: ": str(max(all_block_pos_y)),
            "min_y: ": str(min(all_block_pos_y)),
            "delta_y: ": str(max(all_block_pos_y) - min(all_block_pos_y)),
        }
        return is_successful, final_states


# SI_2: Put all the blocks on the {corner_or_side}
class TesterSI_2(Tester):
    def __init__(self):
        super().__init__("SI_2")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: np.random.choice(range(2, 5))
        self.n_bowls_generator = lambda: np.random.choice(range(1, 4))

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        corner_pos_2d = CORNER_POS[self.vars_dict["corner_or_side"]][:2]
        is_successful = True
        for block_pos in all_block_pos:
            if np.linalg.norm(block_pos[:2] - corner_pos_2d) > threshold:
                is_successful = False
                break
        final_states = {
            "initial positions": self.position_dict,
            "corner_pos_2d": str(corner_pos_2d),
            "final all_block_pos": str(all_block_pos),
            "block to corner distance": str(
                np.linalg.norm(block_pos[:2] - corner_pos_2d)
            ),
        }
        return is_successful, final_states


# SI_3: Put the blocks in the {bowl}
class TesterSI_3(Tester):
    def __init__(self):
        super().__init__("SI_3")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: np.random.choice(range(2, 5))

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        bowl_pos_2d = self.env.get_obj_pos(self.vars_dict["bowl"])[:2]
        is_successful = True
        for block_pos in all_block_pos:
            if np.linalg.norm(block_pos[:2] - bowl_pos_2d) > threshold:
                is_successful = False
                break
        final_states = {
            "initial positions": self.position_dict,
            "final bowl_pos_2d": str(bowl_pos_2d),
            "final all_block_pos": str(all_block_pos),
            "block to bowl distances": str(
                [
                    np.linalg.norm(block_pos[:2] - bowl_pos_2d)
                    for block_pos in all_block_pos
                ]
            ),
        }
        return is_successful, final_states


# SI_4: Put all the blocks in the bowls with matching colors
class TesterSI_4(Tester):
    def __init__(self):
        super().__init__("SI_4")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: 4
        self.n_bowls_generator = lambda: 4

    def check_final_state(self, threshold=THRESHOLD):
        matches = get_matches(self.env)
        is_successful = True
        for block, bowl in matches.items():
            if (
                np.linalg.norm(
                    self.env.get_obj_pos(block)[:2] - self.env.get_obj_pos(bowl)[:2]
                )
                > threshold
            ):
                is_successful = False
                break
        final_states = {
            "initial positions": self.position_dict,
            "n_matches": len(matches),
        }
        for block, bowl in matches.items():
            final_states[f"{block}_{bowl}_distance"] = str(
                np.linalg.norm(
                    self.env.get_obj_pos(block)[:2] - self.env.get_obj_pos(bowl)[:2]
                )
            )
        return is_successful, final_states


# SI_5: Pick up all blocks to the {direction} of the {bowl} and place them on the {corner_or_side}
class TesterSI_5(Tester):
    def __init__(self):
        super().__init__("SI_5")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: 4
        self.n_bowls_generator = lambda: np.random.choice(range(1, 4))

    def update_vars_by_initial_state(self, instruction_vars_dict):
        # need to check initial state and fill in self.vars_dict["target_blocks"]
        all_block_pos, all_block_names = get_all_blocks_or_bowls_pos(self.env)
        all_block_pos_x = all_block_pos[:, 0]
        all_block_pos_y = all_block_pos[:, 1]
        for block, block_pos in zip(all_block_names, all_block_pos):
            instruction_vars_dict[f"{block}_initial_pos"] = block_pos

        bowl_pos = self.env.get_obj_pos(instruction_vars_dict["bowl"])
        bowl_x = bowl_pos[0]
        bowl_y = bowl_pos[1]
        direction = instruction_vars_dict["direction"]

        block_indices = []
        if direction == "top":
            sorted_idx = np.argsort(all_block_pos_y)
            for idx in sorted_idx:
                if all_block_pos_y[idx] > bowl_y:
                    block_indices.append(idx)
        elif direction == "bottom":
            sorted_idx = reversed(np.argsort(all_block_pos_y))
            for idx in sorted_idx:
                if all_block_pos_y[idx] < bowl_y:
                    block_indices.append(idx)
        elif direction == "left":
            sorted_idx = reversed(np.argsort(all_block_pos_x))
            for idx in sorted_idx:
                if all_block_pos_x[idx] < bowl_x:
                    block_indices.append(idx)
        elif direction == "right":
            sorted_idx = np.argsort(all_block_pos_x)
            for idx in sorted_idx:
                if all_block_pos_x[idx] > bowl_x:
                    block_indices.append(idx)
        else:
            raise KeyError
        instruction_vars_dict["target_blocks"] = [
            all_block_names[block_idx] for block_idx in block_indices
        ]

    def check_final_state(self, threshold=THRESHOLD):
        target_blocks = self.vars_dict["target_blocks"]
        corner_pos_2d = CORNER_POS[self.vars_dict["corner_or_side"]][:2]
        is_successful = True
        all_block_pos, all_block_names = get_all_blocks_or_bowls_pos(self.env)
        for i, block in enumerate(all_block_names):
            if block in target_blocks:
                if (
                    np.linalg.norm(self.env.get_obj_pos(block)[:2] - corner_pos_2d)
                    > threshold
                ):
                    is_successful = False
                    break
            else:
                # non-target blocks should not be moved
                if (
                    np.linalg.norm(
                        np.array(self.vars_dict[f"{block}_initial_pos"][:2]).astype(
                            "float32"
                        )
                        - all_block_pos[i][:2]
                    )
                    > threshold
                ):
                    is_successful = False
                    break
        final_states = {
            "initial positions": self.position_dict,
            "corner_pos_2d": corner_pos_2d,
        }
        if target_blocks:
            for target_block in target_blocks:
                final_states[f"{target_block}_pos_2d"] = str(
                    self.env.get_obj_pos(target_block)[:2]
                )
                final_states[f"{target_block}_corner_distance"] = str(
                    np.linalg.norm(
                        self.env.get_obj_pos(target_block)[:2] - corner_pos_2d
                    )
                )
        return is_successful, final_states


# SI_6: Pick up the block {distance} to the {bowl} and place it on the {corner_or_side}
class TesterSI_6(Tester):
    def __init__(self):
        super().__init__("SI_6")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: 4
        self.n_bowls_generator = lambda: np.random.choice(range(2, 4))

    def update_vars_by_initial_state(self, instruction_vars_dict):
        distance = instruction_vars_dict["distance"]
        bowl = instruction_vars_dict["bowl"]
        bowl_pos = self.env.get_obj_pos(bowl)
        all_block_pos, all_block_names = get_all_blocks_or_bowls_pos(self.env)
        _, block_idx = get_closest_or_farthest_point(
            all_block_pos, bowl_pos, distance_type=distance
        )
        instruction_vars_dict.update({"target_block": all_block_names[block_idx]})

    def check_final_state(self, threshold=THRESHOLD):
        target_block = self.vars_dict["target_block"]
        corner_pos_2d = CORNER_POS[self.vars_dict["corner_or_side"]][:2]
        is_successful = (
            np.linalg.norm(self.env.get_obj_pos(target_block)[:2] - corner_pos_2d)
            < threshold
        )
        final_states = {
            "initial positions": self.position_dict,
            "target_block_pos_2d": str(self.env.get_obj_pos(target_block)[:2]),
            "corner_pos_2d": str(corner_pos_2d),
            "distance": str(
                np.linalg.norm(self.env.get_obj_pos(target_block)[:2] - corner_pos_2d)
            ),
        }
        return is_successful, final_states


# SI_7: Pick up the {n_th} block from the {direction} and place it on the {corner_or_side}
class TesterSI_7(Tester):
    def __init__(self):
        super().__init__("SI_7")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: 4
        self.n_bowls_generator = lambda: np.random.choice(range(1, 4))
        self.nth_mapping = {"first": 0, "second": 1, "third": 2, "fourth": 3}

    def update_vars_by_initial_state(self, instruction_vars_dict):
        direction = instruction_vars_dict["direction"]
        nth_idx = self.nth_mapping[instruction_vars_dict["nth"]]
        all_block_pos, all_block_names = get_all_blocks_or_bowls_pos(self.env)
        n_blocks = len(all_block_names)
        if direction == "top":
            block_idx = np.argsort(all_block_pos[:, 1])[n_blocks - 1 - nth_idx]
        elif direction == "bottom":
            block_idx = np.argsort(all_block_pos[:, 1])[nth_idx]
        elif direction == "left":
            block_idx = np.argsort(all_block_pos[:, 0])[nth_idx]
        elif direction == "right":
            block_idx = np.argsort(all_block_pos[:, 0])[n_blocks - 1 - nth_idx]
        else:
            raise KeyError
        instruction_vars_dict.update({"target_block": all_block_names[block_idx]})

    def check_final_state(self, threshold=THRESHOLD):
        target_block = self.vars_dict["target_block"]
        corner_pos_2d = CORNER_POS[self.vars_dict["corner_or_side"]][:2]
        is_successful = (
            np.linalg.norm(self.env.get_obj_pos(target_block)[:2] - corner_pos_2d)
            < threshold
        )
        final_states = {
            "initial positions": self.position_dict,
            "target_block_pos_2d": str(self.env.get_obj_pos(target_block)[:2]),
            "corner_pos_2d": str(corner_pos_2d),
            "distance": str(
                np.linalg.norm(self.env.get_obj_pos(target_block)[:2] - corner_pos_2d)
            ),
        }
        return is_successful, final_states


# UI_0: Put all the blocks in different corners
class TesterUI_0(Tester):
    def __init__(self):
        super().__init__("UI_0")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: 4
        self.n_bowls_generator = lambda: np.random.choice(range(1, 4))

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        all_corner_pos = get_corner_positions()

        is_successful = True
        used_corners = []
        final_states = {
            "initial positions": self.position_dict,
            "final block positions": str(all_block_pos),
            "all_corner positions": str(all_corner_pos),
        }

        for block_pos in all_block_pos:
            closest_corner, corner_idx = get_closest_or_farthest_point(
                all_corner_pos, block_pos, distance_type="closest"
            )
            distance = np.linalg.norm(closest_corner[:2] - block_pos[:2])
            if distance > threshold:
                is_successful = False
                break
            else:
                used_corners.append(corner_idx)
        if len(set(used_corners)) != len(all_block_pos):
            is_successful = False
        return is_successful, final_states


# UI_1: Put the blocks in the bowls with mismatched colors
class TesterUI_1(Tester):
    def __init__(self):
        super().__init__("UI_1")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: 4
        self.n_bowls_generator = lambda: 4

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        all_bowl_pos = get_all_blocks_or_bowls_pos(self.env, object_type="bowl")[0]

        is_successful = True
        used_bowls = []

        final_states = {
            "initial positions": self.position_dict,
        }
        for block_pos in all_block_pos:
            closest_bowl, bowl_idx = get_closest_or_farthest_point(
                all_bowl_pos, block_pos, distance_type="closest"
            )
            distance = np.linalg.norm(closest_bowl[:2] - block_pos[:2])
            final_states[f"distance_{bowl_idx}"] = str(distance)
            if distance > threshold:
                is_successful = False
                break
            else:
                used_bowls.append(bowl_idx)
        if len(set(used_bowls)) != len(used_bowls):
            is_successful = False

        return is_successful, final_states


# UI_2: Stack all the blocks on the {corner_or_side}
class TesterUI_2(Tester):
    def __init__(self):
        super().__init__("UI_2")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: np.random.choice(range(2, 5))
        self.n_bowls_generator = lambda: np.random.choice(range(1, 4))

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        corner_pos_2d = CORNER_POS[self.vars_dict["corner_or_side"]][:2]
        is_successful = True

        final_states = {
            "initial positions": self.position_dict,
            "corner pos": str(corner_pos_2d),
            "final block_pos": str(all_block_pos),
        }
        for block_pos in all_block_pos:
            if np.linalg.norm(block_pos[:2] - corner_pos_2d) > threshold:
                is_successful = False
                break
        return is_successful, final_states


# UI_3: Pick up the {block} and place it {magnitude} to the {direction} of the {bowl}
class TesterUI_3(Tester):
    def __init__(self):
        super().__init__("UI_3")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.magnitude_mappings = {"10 cm": 0.1, "0.2 m": 0.2}

    def check_final_state(self, threshold=THRESHOLD):
        delta = self.magnitude_mappings[self.vars_dict["magnitude"]]
        direction = self.vars_dict["direction"]
        block_pos_2d = self.env.get_obj_pos(self.vars_dict["block"])[:2]
        bowl_x = self.env.get_obj_pos(self.vars_dict["bowl"])[0]
        bowl_y = self.env.get_obj_pos(self.vars_dict["bowl"])[1]
        if direction == "top":
            target_pos = [bowl_x, bowl_y + delta]
        elif direction == "left":
            target_pos = [bowl_x - delta, bowl_y]
        elif direction == "bottom":
            target_pos = [bowl_x, bowl_y - delta]
        elif direction == "right":
            target_pos = [bowl_x + delta, bowl_y]
        else:
            raise KeyError
        is_successful = np.linalg.norm(block_pos_2d - target_pos) < threshold
        final_states = {
            "initial positions": self.position_dict,
            "bowl_pos": str(self.env.get_obj_pos(self.vars_dict["bowl"])),
            "target pos": str(target_pos),
            "block pos": str(block_pos_2d),
            "distance": str(np.linalg.norm(block_pos_2d - target_pos)),
        }
        return is_successful, final_states


# UI_4: Pick up the {block} and place it in the corner {distance} to the {bowl}
class TesterUI_4(Tester):
    def __init__(self):
        super().__init__("UI_4")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_bowls_generator = lambda: np.random.choice(range(1, 4))

    def check_final_state(self, threshold=THRESHOLD):
        block_pos_2d = self.env.get_obj_pos(self.vars_dict["block"])[:2]
        distance = self.vars_dict["distance"]
        bowl_pos = self.env.get_obj_pos(self.vars_dict["bowl"])
        corners_pos = get_corner_positions()
        target_corner_pos, _ = get_closest_or_farthest_point(
            corners_pos, bowl_pos, distance_type=distance
        )

        is_successful = np.linalg.norm(block_pos_2d - target_corner_pos[:2]) < threshold
        final_states = {
            "initial positions": self.position_dict,
            "bowl pos": str(self.env.get_obj_pos(self.vars_dict["bowl"])),
            "target corner pos": str(target_corner_pos),
            "distance": str(np.linalg.norm(block_pos_2d - target_corner_pos[:2])),
        }
        return is_successful, final_states


# UI_5: Put all the blocks in a {line} line
class TesterUI_5(Tester):
    def __init__(self):
        super().__init__("UI_5")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: np.random.choice(range(3, 5))
        self.n_bowls_generator = lambda: 0

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        all_block_pos_x = all_block_pos[:, 0]
        all_block_pos_y = all_block_pos[:, 1]
        line = self.vars_dict["line"]
        if line == "vertical":
            is_successful = max(all_block_pos_x) - min(all_block_pos_x) < threshold
        elif line == "horizontal":
            is_successful = max(all_block_pos_y) - min(all_block_pos_y) < threshold
        elif line == "diagonal":
            is_successful = all(
                (all_block_pos_x - all_block_pos_y - 0.5) < threshold
            ) or all((all_block_pos_x + all_block_pos_y + 0.5) < threshold)
        else:
            raise KeyError
        final_states = {
            "initial positions": self.position_dict,
            "blocks pos": str(all_block_pos),
        }
        return is_successful, final_states


# UI_6: Put all the blocks in different sides
class TesterUI_6(Tester):
    def __init__(self):
        super().__init__("UI_6")
        # overwrite self.n_blocks_generator and self.n_bowls_generator if needed
        self.n_blocks_generator = lambda: 4
        self.n_bowls_generator = lambda: np.random.choice(range(1, 4))

    def check_final_state(self, threshold=THRESHOLD):
        all_block_pos = get_all_blocks_or_bowls_pos(self.env)[0]
        all_side_pos = get_side_positions()

        is_successful = True
        used_sides = []
        final_states = {
            "initial positions": self.position_dict,
            "final block positions": str(all_block_pos),
        }

        for block_pos in all_block_pos:
            closest_side, side_idx = get_closest_or_farthest_point(
                all_side_pos, block_pos, distance_type="closest"
            )
            distance = np.linalg.norm(closest_side[:2] - block_pos[:2])
            final_states[f"distance_{side_idx}"] = str(distance)
            if distance > threshold:
                is_successful = False
                break
            else:
                used_sides.append(side_idx)
        if len(set(used_sides)) != len(all_block_pos):
            is_successful = False
        return is_successful, final_states


TESTERS = {
    "SI_0": TesterSI_0,
    "SI_1": TesterSI_1,
    "SI_2": TesterSI_2,
    "SI_3": TesterSI_3,
    "SI_4": TesterSI_4,
    "SI_5": TesterSI_5,
    "SI_6": TesterSI_6,
    "SI_7": TesterSI_7,
    "UI_0": TesterUI_0,
    "UI_1": TesterUI_1,
    "UI_2": TesterUI_2,
    "UI_3": TesterUI_3,
    "UI_4": TesterUI_4,
    "UI_5": TesterUI_5,
    "UI_6": TesterUI_6,
}
