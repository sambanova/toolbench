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

import ast
import astunparse
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter


class LMP:
    def __init__(self, get_action_func, name, cfg, lmp_fgen, fixed_vars, variable_vars):
        self.get_action_func = get_action_func
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg["prompt_text"]

        self._stop_tokens = list(self._cfg["stop"])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ""

    def clear_exec_hist(self):
        self.exec_hist = ""

    def build_prompt(self, query, context=""):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = (
                f"from utils import {', '.join(self._variable_vars.keys())}"
            )
        else:
            variable_vars_imports_str = ""
        prompt = self._base_prompt.replace(
            "{variable_vars_imports}", variable_vars_imports_str
        )

        if self._cfg["maintain_session"]:
            prompt += f"\n{self.exec_hist}"

        if context != "":
            prompt += f"\n{context}"

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f"\n{use_query}"

        return prompt, use_query

    def __call__(self, query, context="", **kwargs):
        prompt, use_query = self.build_prompt(query, context=context)

        _, code_str, crash_error = self.get_action_func(
            prompt, stop_tokens=self._stop_tokens
        )
        if crash_error:
            raise Exception(f"Language model generation error: str({crash_error})")

        if self._cfg["include_context"] and context != "":
            to_exec = f"{context}\n{code_str}"
            to_log = f"{context}\n{use_query}\n{code_str}"
        else:
            to_exec = code_str
            to_log = f"{use_query}\n{to_exec}"

        print(f"LMP {self._name} exec:\n\n{to_log}\n")

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg["debug_mode"]:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f"\n{to_exec}"

        if self._cfg["maintain_session"]:
            self._variable_vars.update(lvars)

        if self._cfg["has_return"]:
            return lvars[self._cfg["return_val_name"]]


class LMPFGen:
    def __init__(self, get_action_func, cfg, fixed_vars, variable_vars):
        self.get_action_func = get_action_func
        self._cfg = cfg

        self._stop_tokens = list(self._cfg["stop"])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg["prompt_text"]

    def create_f_from_sig(
        self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False
    ):
        print(f"Creating function: {f_sig}")

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f"{self._base_prompt}\n{use_query}"
        _, f_src, crash_error = self.get_action_func(
            prompt, stop_tokens=self._stop_tokens
        )
        if crash_error:
            raise Exception(f"Language model generation error: str({crash_error})")

        to_print = f"{use_query}\n{f_src}"
        print(f"LMP FGEN created:\n\n{to_print}\n")

        if fix_bugs:
            raise NotImplementedError
            # f_src = openai.Edit.create(
            #     model="code-davinci-edit-001",
            #     input="# " + f_src,
            #     temperature=0,
            #     instruction="Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.",
            # )["choices"][0]["text"].strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(
        self, code_str, other_vars=None, fix_bugs=False, return_src=False
    ):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts(
                [self._fixed_vars, self._variable_vars, new_fs, other_vars]
            )
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(
                    f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True
                )

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts(
                        [self._fixed_vars, self._variable_vars, new_fs, other_vars]
                    )
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):
    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    exec(code_str, custom_gvars, lvars)
