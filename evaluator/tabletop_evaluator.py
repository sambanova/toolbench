"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
import numpy as np
from typing import List
from io import StringIO
from contextlib import redirect_stdout

import shapely
from shapely.geometry import *
from shapely.affinity import *

from evaluator.code_as_policies_env.LMP_configs import lmp_tabletop_coords, cfg_tabletop
from evaluator.code_as_policies_env.LMP import LMP, LMPFGen
from evaluator.code_as_policies_env.LMP_env import LMPEnvWrapper
from evaluator.code_as_policies_env.tester import Tester

from evaluator import BaseEvaluator


class TabletopEvaluator(BaseEvaluator):
    def __init__(self, generator):
        super().__init__(generator)

    ## This function is copied from [Code as Policies](https://code-as-policies.github.io/).
    def setup(self, env):
        self.env = env

        # LMP env wrapper
        cfg = copy.deepcopy(cfg_tabletop)
        cfg["env"] = dict()
        cfg["env"]["init_objs"] = list(env.obj_name_to_id.keys())
        cfg["env"]["coords"] = lmp_tabletop_coords
        LMP_env = LMPEnvWrapper(env, cfg)
        # creating APIs that the LMPs can interact with
        fixed_vars = {"np": np}
        fixed_vars.update(
            {
                name: eval(name)
                for name in shapely.geometry.__all__ + shapely.affinity.__all__
            }
        )
        variable_vars = {
            k: getattr(LMP_env, k)
            for k in [
                "get_bbox",
                "get_obj_pos",
                "get_color",
                "is_obj_visible",
                "denormalize_xy",
                "put_first_on_second",
                "get_obj_names",
                "get_corner_name",
                "get_side_name",
            ]
        }
        variable_vars["say"] = lambda msg: print(f"robot says: {msg}")

        # creating the function-generating LMP
        lmp_fgen = LMPFGen(
            self.get_action, cfg["lmps"]["fgen"], fixed_vars, variable_vars
        )

        # creating other low-level LMPs
        variable_vars.update(
            {
                k: LMP(
                    self.get_action,
                    k,
                    cfg["lmps"][k],
                    lmp_fgen,
                    fixed_vars,
                    variable_vars,
                )
                for k in [
                    "parse_obj_name",
                    "parse_position",
                    "parse_question",
                    "transform_shape_pts",
                ]
            }
        )

        # creating the LMP that deals w/ high-level language commands
        self.lmp_tabletop_ui = LMP(
            self.get_action,
            "tabletop_ui",
            cfg["lmps"]["tabletop_ui"],
            lmp_fgen,
            fixed_vars,
            variable_vars,
        )

    def __call__(self, query: str, tester: Tester):
        output = StringIO()
        crash_error = None
        with redirect_stdout(output):
            try:
                self.lmp_tabletop_ui(query, f"objects = {self.env.object_list}")
            except Exception as e:
                crash_error = e
        captured_output = output.getvalue()
        print(captured_output)

        if crash_error:
            results = {
                "crashed": True,
                "success": False,
                "query": query,
                "generation": captured_output,
                "crashed_error_msg": f"{str(crash_error)}",
            }
        else:
            success, final_states = tester.check_final_state()
            results = {
                "crashed": False,
                "success": bool(success),
                "query": query,
                "generation": captured_output,
                "final_states_log": final_states,
            }
        self.full_results.append(results)
        return results

    def aggregate_results(self):
        total_crashed = sum([s["crashed"] for s in self.full_results])
        total_success = sum([s["success"] for s in self.full_results])
        n_samples = len(self.full_results)
        results = {
            "crash_rate": total_crashed / n_samples,
            "success_rate": total_success / n_samples,
        }
        return results
