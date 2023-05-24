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

from typing import List

from evaluator import BaseEvaluator
from evaluator.utils import compute_f1_between_dicts


class HomeSearchAPI:
    def __init__(self):
        self.search_criteria = {}

    def clear(self):
        self.search_criteria = {}

    def search(self):
        # Submit criterion to get search results.
        # Add default values
        if "min_price" not in self.search_criteria:
            self.search_criteria["min_price"] = 0
        if "min_square_feet" not in self.search_criteria:
            self.search_criteria["min_square_feet"] = 0

    def select_home_type(self, home_types: List[str]):
        # To set home types for search. For home buying, home_types choices are: "House", "Townhouse", "Condo", "Land", "Multi-family", "Mobile", "Co-op"; for home renting, home_types choices are: "House", "Townhouse", "Condo", "Apartment".
        home_types = [t.lower() for t in home_types]
        for t in home_types:
            if self.search_criteria["buy_or_rent"] == "rent":
                assert t in [
                    "house",
                    "townhouse",
                    "condo",
                    "apartment",
                ], "Specified invalid 'rent' home type(s) for select_home_type"
            else:
                assert t in [
                    "house",
                    "townhouse",
                    "condo",
                    "land",
                    "multi-family",
                    "mobile",
                    "co-op",
                ], "Specified invalid 'buy' home type(s) for select_home_type"
        home_types.sort()
        self.search_criteria["home_type"] = ",".join(home_types)

    def set_buy_or_rent(self, value: str):
        # To specify whether to search homes for buying or renting. 'value' can be chosed from ['buy', 'rent'].
        assert value in ["buy", "rent"], "invalid value for set_buy_or_rent"
        assert (
            "location" in self.search_criteria
        ), "didn't call set_location before calling set_buy_or_rent"
        assert (
            len(self.search_criteria) == 1
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["buy_or_rent"] = value

    def set_location(self, value: str):
        # To set the location for the search area.
        assert (
            self.search_criteria == {}
        ), "didn't call set_location before setting other criterion"
        self.search_criteria["location"] = value

    def set_max_price(self, value: int):
        # To set the maximum home price in dollars
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["max_price"] = value

    def set_min_price(self, value: int):
        # To set the minimum home price in dollars
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["min_price"] = value

    def set_max_square_feet(self, value: int):
        # To set the maximum home size in square feet
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["max_square_feet"] = value

    def set_min_square_feet(self, value: int):
        # To set the minimum home size in square feet
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["min_square_feet"] = value

    def set_num_baths(self, value: float):
        # To set the number of bathroom(s)
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["num_baths"] = value

    def set_num_beds(self, value: int):
        # To set the number of bedroom(s)
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["num_beds"] = value

    def set_num_garages(self, value: int):
        # To set the number of garage(s)
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["num_garages"] = value

    def set_num_swimming_pools(self, value: int):
        # To set the number of swimming pool(s)
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["num_swimming_pools"] = value

    def set_floor_number(self, value: int):
        # To set the floor number
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["floor_number"] = value

    def set_max_commute_time(self, value: int):
        # To set the max commute time
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["max_commute_time"] = value

    def set_num_balconies(self, value: int):
        # To set the number of balconies
        assert (
            "buy_or_rent" in self.search_criteria
        ), "didn't call set_buy_or_rent before setting other criterion"
        self.search_criteria["num_balconies"] = value


class HomeSearchEvaluator(BaseEvaluator):
    def __init__(self, generator):
        super().__init__(generator)

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
            crash_error, pred_dict = self.exec_code(prediction)
        if crash_error:
            results = {
                "crashed": True,
                "success": False,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "lcs_score": 0,
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
                    "pred_dict": sorted(pred_dict.items()),
                    "label_dict": sorted(pred_dict.items()),
                }
                best_f1 = f1

        self.full_results.append(results)
        return results

    def compute_scores(self, pred_dict, label_dict):
        assert len(label_dict) > 0
        if len(pred_dict) == 0:
            return 0, 0, 0
        overlap = 0
        for k, v in pred_dict.items():
            if k in label_dict and v == label_dict[k]:
                overlap += 1

        precision = overlap / len(pred_dict)
        recall = overlap / len(label_dict)
        if precision + recall == 0:
            return 0, 0, 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def exec_code(self, code):
        API = HomeSearchAPI()

        try:
            exec(code, {"API": API})
            result = API.search_criteria
            assert (
                type(result) is dict
            ), f"expected type(result) == <class 'dict'>, but get type(result) = {type(result)}"
        except Exception as e:
            return e, {}

        return None, result
