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
from typing import List
import subprocess
import os
import time

from evaluator import BaseEvaluator
from evaluator.utils import compute_f1_between_dicts

N_RETRY = 3

BREED_MAP = {
    "Abyssinian": "abys",
    "Aegean": "aege",
    "American Bobtail": "abob",
    "American Curl": "acur",
    "American Shorthair": "asho",
    "American Wirehair": "awir",
    "Arabian Mau": "amau",
    "Australian Mist": "amis",
    "Balinese": "bali",
    "Bambino": "bamb",
    "Bengal": "beng",
    "Birman": "birm",
    "Bombay": "bomb",
    "British Longhair": "bslo",
    "British Shorthair": "bsho",
    "Burmese": "bure",
    "Burmilla": "buri",
    "California Spangled": "cspa",
    "Chantilly-Tiffany": "ctif",
    "Chartreux": "char",
    "Chausie": "chau",
    "Cheetoh": "chee",
    "Colorpoint Shorthair": "csho",
    "Cornish Rex": "crex",
    "Cymric": "cymr",
    "Cyprus": "cypr",
    "Devon Rex": "drex",
    "Donskoy": "dons",
    "Dragon Li": "lihu",
    "Egyptian Mau": "emau",
    "European Burmese": "ebur",
    "Exotic Shorthair": "esho",
    "Havana Brown": "hbro",
    "Himalayan": "hima",
    "Japanese Bobtail": "jbob",
    "Javanese": "java",
    "Khao Manee": "khao",
    "Korat": "kora",
    "Kurilian": "kuri",
    "LaPerm": "lape",
    "Maine Coon": "mcoo",
    "Malayan": "mala",
    "Manx": "manx",
    "Munchkin": "munc",
    "Nebelung": "nebe",
    "Norwegian Forest Cat": "norw",
    "Ocicat": "ocic",
    "Oriental": "orie",
    "Persian": "pers",
    "Pixie-bob": "pixi",
    "Ragamuffin": "raga",
    "Ragdoll": "ragd",
    "Russian Blue": "rblu",
    "Savannah": "sava",
    "Scottish Fold": "sfol",
    "Selkirk Rex": "srex",
    "Siamese": "siam",
    "Siberian": "sibe",
    "Singapura": "sing",
    "Snowshoe": "snow",
    "Somali": "soma",
    "Sphynx": "sphy",
    "Tonkinese": "tonk",
    "Toyger": "toyg",
    "Turkish Angora": "tang",
    "Turkish Van": "tvan",
    "York Chocolate": "ycho",
}


class TheCatAPIEvaluator(BaseEvaluator):
    def __init__(self, generator):
        super().__init__(generator)
        self.api_key = os.environ.get("THE_CAT_API_KEY")

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

    def check_string(self, prediction, label):
        prediction = re.sub(r"[ \t]+", " ", prediction).strip()
        prediction = re.sub(r"\n+", "\n", prediction).strip(" \n")
        return prediction == label

    def __call__(self, query: str, labels: List[str], compare_string_only=False):
        crash_type = "Language model generation error"
        prompt, prediction, crash_error = self.get_action(query)
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
                "compare_generation_and_label_only": compare_string_only,
                "crashed_error_msg": f"{crash_type}: {str(crash_error)}",
            }
            self.full_results.append(results)
            return results

        # Keep only the curl line
        for line in prediction.split("\n"):
            line = line.strip()
            if line.startswith("curl"):
                prediction = line
                break

        crash_type = "Task execution error"
        if compare_string_only:
            res = False
            best_label = labels[0]
            for label in labels:
                res = self.check_string(prediction, label)
                if res:
                    best_label = label
                    break
            score = float(res)
            results = {
                "crashed": not res,
                "success": res,
                "precision": score,
                "recall": score,
                "f1": score,
                "prompt": prompt,
                "query": query,
                "prediction": prediction,
                "closest_label": best_label,
                "compare_generation_and_label_only": True,
                "crashed_error_msg": None
                if res
                else f"{crash_type}: string comparison between the generation and the label failed",
            }
            self.full_results.append(results)
            return results

        for _ in range(N_RETRY):
            crash_error, pred_dict = self.exec_code(prediction)
            if crash_error:
                time.sleep(2)
            else:
                break

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
                "compare_generation_and_label_only": False,
                "crashed_error_msg": f"{crash_type}: {str(crash_error)}",
            }
            self.full_results.append(results)
            return results

        best_f1 = -1
        for label in labels:
            for _ in range(N_RETRY):
                crash_error, label_dict = self.exec_code(label)
                if crash_error:
                    time.sleep(2)
                else:
                    break
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
                    "compare_generation_and_label_only": False,
                }
                best_f1 = f1

        self.full_results.append(results)
        return results

    def exec_code(self, code):
        code = code.strip("\n")
        code += f" -H 'Content-Type: application/json' -H 'x-api-key: {self.api_key}'"
        # print(code)

        try:
            process = subprocess.Popen(
                code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output, _ = process.communicate()
            output = output.decode()
            return_code = process.returncode
            assert return_code == 0, f"shell process failed: return code {return_code}"

            output = output.replace("null", '"null"')
            result = {}
            result = eval(output)
            if type(result) is dict:
                assert (
                    result["message"] == "SUCCESS"
                ), f"HTTP request failed: response: {result}"
                if "id" in result:
                    del result["id"]
            elif type(result) is list:
                result_dict = {}
                for i, item in enumerate(result):
                    result_dict[i] = item
                result = result_dict
            assert (
                type(result) is dict
            ), f"expected type(result) == <class 'dict'>, but get type(result) = {type(result)}"
        except Exception as e:
            return e, {}

        return None, result
