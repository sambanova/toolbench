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

import subprocess
from typing import List
import os

from evaluator import BaseEvaluator
from evaluator.utils import compute_f1_between_dicts


class OpenWeatherEvaluator(BaseEvaluator):
    def __init__(self, generator):
        super().__init__(generator)
        self.api_key = os.environ.get("OPEN_WEATHER_KEY")

    def aggregate_results(self):
        total_crashed = sum([s["crashed"] for s in self.full_results])
        total_success = sum([s["success"] for s in self.full_results])
        total_precision = sum([s["precision"] for s in self.full_results])
        total_recall = sum([s["recall"] for s in self.full_results])
        total_f1 = sum([s["f1"] for s in self.full_results])
        n_samples = len(self.full_results)
        results = {
            "crash_rate": total_crashed / n_samples,
            "success_rate": total_success / n_samples,
            "precision": total_precision / n_samples,
            "recall": total_recall / n_samples,
            "f1": total_f1 / n_samples,
        }
        return results

    def __call__(self, query: str, labels: List[str]):
        crash_type = "Language model generation error"
        prompt, prediction, crash_error = self.get_action(query)
        if not crash_error:
            crash_type = "Task execution error"
            # Keep only the curl line
            for line in prediction.split("\n"):
                line = line.strip()
                if line.startswith("curl"):
                    prediction = line
                    break
            crash_error, pred_dict = self.exec_code(prediction)
        if crash_error:
            results = {
                "crashed": True,
                "success": False,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "prompt": prompt,
                "query": query,
                "prediction": prediction,
                "closest_label": labels[0],
                "crashed_error_msg": f"{crash_type}: {str(crash_error)}",
            }
            self.full_results.append(results)
            return results

        best_f1 = -1
        for label in labels:
            crash_error, label_dict = self.exec_code(label)
            assert crash_error == None, f"label: {label}\nerror: {str(crash_error)}"
            precision, recall, f1 = compute_f1_between_dicts(pred_dict, label_dict)
            if f1 > best_f1:
                results = {
                    "crashed": False,
                    "success": f1 == 1,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "prompt": prompt,
                    "query": query,
                    "prediction": prediction,
                    "closest_label": label,
                }
                best_f1 = f1

        self.full_results.append(results)
        return results

    def exec_code(self, code):
        try:
            code = code.strip("\n").format(API_KEY=self.api_key)
            assert code.endswith("'"), "code does not end with '"
        except Exception as e:
            return e, {}
        code = code.replace("%20", "+")

        if "air_pollution" not in code and "direct" not in code:
            # add default arguments
            code = code[:-1]
            if "lang" not in code:
                code += "&lang=en"
            if "units" not in code:
                code += "&units=standard"
            if "mode" not in code:
                code += "&mode=json"
            code += "'"

        try:
            process = subprocess.Popen(
                code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, _ = process.communicate()
            output = output.decode()
            return_code = process.returncode
            assert return_code == 0, f"shell process failed: return code {return_code}"

            result = eval(output)
            if type(result) is dict:
                if "cod" in result:
                    cod = int(result["cod"])
                else:
                    assert (
                        "air_pollution" in code
                    ), "HTTP response should contain 'cod' value."
                    cod = 200
            elif type(result) is list:
                cod = 200
                result_dict = {}
                for i, item in enumerate(result):
                    result_dict[i] = item
                result = result_dict

            assert cod == 200, f"HTTP request failed: status code {cod}"
            assert (
                type(result) is dict
            ), f"expected type(result) == <class 'dict'>, but get type(result) = {type(result)}"
        except Exception as e:
            return e, {}

        return None, result
