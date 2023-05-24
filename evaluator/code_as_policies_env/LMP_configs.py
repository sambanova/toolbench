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

from evaluator.code_as_policies_env.LMP_prompts import *

cfg_tabletop = {
    "lmps": {
        "tabletop_ui": {
            "prompt_text": prompt_tabletop_ui,
            "engine": "code-davinci-002",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#", "objects = ["],
            "maintain_session": True,
            "debug_mode": False,
            "include_context": True,
            "has_return": False,
            "return_val_name": "ret_val",
        },
        "parse_obj_name": {
            "prompt_text": prompt_parse_obj_name,
            "engine": "code-davinci-002",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#", "objects = ["],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "parse_position": {
            "prompt_text": prompt_parse_position,
            "engine": "code-davinci-002",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "parse_question": {
            "prompt_text": prompt_parse_question,
            "engine": "code-davinci-002",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#", "objects = ["],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "ret_val",
        },
        "transform_shape_pts": {
            "prompt_text": prompt_transform_shape_pts,
            "engine": "code-davinci-002",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# ",
            "query_suffix": ".",
            "stop": ["#"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
            "has_return": True,
            "return_val_name": "new_shape_pts",
        },
        "fgen": {
            "prompt_text": prompt_fgen,
            "engine": "code-davinci-002",
            "max_tokens": 512,
            "temperature": 0,
            "query_prefix": "# define function: ",
            "query_suffix": ".",
            "stop": ["# define", "# example"],
            "maintain_session": False,
            "debug_mode": False,
            "include_context": True,
        },
    }
}

lmp_tabletop_coords = {
    "top_left": (-0.3 + 0.05, -0.2 - 0.05),
    "top_side": (0, -0.2 - 0.05),
    "top_right": (0.3 - 0.05, -0.2 - 0.05),
    "left_side": (
        -0.3 + 0.05,
        -0.5,
    ),
    "middle": (
        0,
        -0.5,
    ),
    "right_side": (
        0.3 - 0.05,
        -0.5,
    ),
    "bottom_left": (-0.3 + 0.05, -0.8 + 0.05),
    "bottom_side": (0, -0.8 + 0.05),
    "bottom_right": (0.3 - 0.05, -0.8 + 0.05),
    "table_z": 0.0,
}
