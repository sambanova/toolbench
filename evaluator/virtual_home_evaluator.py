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

import re
import json
from typing import List

from evaluator import BaseEvaluator


class VirtualHomeEvaluator(BaseEvaluator):
    def __init__(self, generator):
        super().__init__(generator)

        with open(f"{self.generator.context_dir}/valid_object_lists.json") as f:
            self.valid_object_lists = json.load(f)

    def ground_line(self, line):
        line = line.replace('"', "")
        line = line.replace("'", "")
        line = line.replace(";", "")
        line = re.sub("^\d+\. *", "", line)
        line = line.strip()

        api_call = line.split("(")[0]
        if not api_call in self.valid_object_lists.keys():
            return line, False
        input = re.findall("\((.*?)\)", line)
        if len(input) == 0:
            return line, False
        objects = input[0].split(",")
        if len(objects) == 2:
            object_keys = set(self.valid_object_lists[api_call].keys())
            if object_keys != {"object1", "object2"}:
                return line, False
            for i, object in enumerate(objects):
                object = object.strip()
                if not object in self.valid_object_lists[api_call][f"object{i+1}"]:
                    return line, False
            return line, True
        elif len(objects) == 1:
            object = objects[0].strip()
            if object:
                executable = (
                    "object" in self.valid_object_lists[api_call]
                    and object in self.valid_object_lists[api_call]["object"]
                )
            else:
                executable = self.valid_object_lists[api_call] == {}
            return line, executable
        else:
            return line, False

    def ground(self, raw_text):
        res = ""
        non_executable_lines = []
        for line in raw_text.split("\n"):
            line = line.strip()
            if line.startswith("Task:"):
                break
            if not line:
                continue
            line, line_executable = self.ground_line(line)
            if not line_executable:
                non_executable_lines.append(line)
            if len(line) == 0:
                continue
            res += line + "\n"
        return non_executable_lines, res

    def form_lines(self, text):
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
        return lines

    def compute_LCS(self, prediction: str, label: str):
        prediction, label = self.form_lines(prediction), self.form_lines(label)
        m = len(prediction)
        n = len(label)

        # declaring the array for storing the dp values
        L = [[None] * (n + 1) for i in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif prediction[i - 1] == label[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        lcs = L[m][n]
        return lcs / max(len(prediction), len(label))

    def aggregate_results(self):
        total_score = sum([s["lcs_score"] for s in self.full_results])
        total_executable = sum([not s["crashed"] for s in self.full_results])
        n_samples = len(self.full_results)

        scores_if_executable = [
            s["lcs_score"] for s in self.full_results if not s["crashed"]
        ]
        if len(scores_if_executable) == 0:
            lcs_score_if_executable = "n/a"
        else:
            lcs_score_if_executable = sum(scores_if_executable) / len(
                scores_if_executable
            )

        results = {
            "executability_rate": total_executable / n_samples,
            "lcs_score": total_score / n_samples,
            "lcs_score_if_executable": lcs_score_if_executable,
        }
        return results

    def __call__(self, query: str, labels: List[str]):
        prompt, prediction, crash_error = self.get_action(query)
        if crash_error:
            results = {
                "crashed": True,
                "lcs_score": 0,
                "prompt": prompt,
                "query": query,
                "prediction": prediction,
                "closest_label": labels[0],
                "crashed_error_msg": f"Language model generation error: {str(crash_error)}",
            }
            self.full_results.append(results)
            return results

        pred_non_executable_lines, grounded_prediction = self.ground(prediction)
        is_crashed = pred_non_executable_lines != []

        best_lcs_score = -1.0
        for label in labels:
            label_non_executable_lines, grounded_label = self.ground(label)
            assert (
                label_non_executable_lines == []
            ), f"label:\n{label}\nContains non-executable action(s): {label_non_executable_lines}"

            lcs_score = self.compute_LCS(grounded_prediction, grounded_label)
            if lcs_score > best_lcs_score:
                best_lcs_score = lcs_score
                results = {
                    "crashed": is_crashed,
                    "lcs_score": best_lcs_score,
                    "prompt": prompt,
                    "query": query,
                    "prediction": grounded_prediction,
                    "closest_label": label,
                    "non_executable_lines": f"{pred_non_executable_lines}",
                }
        self.full_results.append(results)
        return results
