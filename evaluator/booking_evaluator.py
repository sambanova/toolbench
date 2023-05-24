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


class Loc:
    def __init__(self, location):
        self.location = location


class Date:
    def __init__(self, month, day, year):
        self.date = f"{month}-{day}-{year}"


class BookingAPI:
    def __init__(self):
        self.search_criteria = {}

    def clear(self):
        self.search_criteria = {}

    def search(self):
        # Submit criterion to get search results.
        if "min_price_ticket" not in self.search_criteria:
            self.search_criteria["min_price_ticket"] = 0
        if "min_price_room" not in self.search_criteria:
            self.search_criteria["min_price_room"] = 0

        # required parameters check
        assert "booking_type" in self.search_criteria
        booking_type = self.search_criteria["booking_type"]
        if booking_type == "trip tickets":
            assert (
                "means_of_transportation" in self.search_criteria
            ), "transportation method not set for trip tickets search"
            assert (
                "location_from" in self.search_criteria
            ), "origin not set for trip tickets search"
            assert (
                "location_to" in self.search_criteria
            ), "destination not set for trip tickets search"
            assert (
                "departure_date" in self.search_criteria
            ), "departure date not set for trip tickets search"

        elif booking_type == "hotels":
            assert (
                "location" in self.search_criteria
            ), "hotel location not set for hotels search"
            assert (
                "checkin_date" in self.search_criteria
            ), "checkin date not set for hotels search"
            assert (
                "checkout_date" in self.search_criteria
            ), "checkout date not set for hotels search"
            if "num_rooms" not in self.search_criteria:
                self.search_criteria["num_rooms"] = 1

        elif booking_type == "both":
            assert (
                "means_of_transportation" in self.search_criteria
            ), "transportation method not set for 'trip tickets and hotels' search"
            assert (
                "location_from" in self.search_criteria
            ), "origin not set for 'trip tickets and hotels' search"
            assert (
                "location_to" in self.search_criteria
            ), "destination not set for 'trip tickets and hotels' search"
            assert (
                "location" in self.search_criteria
            ), "hotel location not set for 'trip tickets and hotels' search"
            assert (
                "checkin_date" in self.search_criteria
            ), "checkin date not set for 'trip tickets and hotels' search"
            assert (
                "checkout_date" in self.search_criteria
            ), "checkout date not set for 'trip tickets and hotels' search"

            if "departure_date" not in self.search_criteria:
                self.search_criteria["departure_date"] = self.search_criteria[
                    "checkin_date"
                ]
            if "return_date" not in self.search_criteria:
                self.search_criteria["return_date"] = self.search_criteria[
                    "checkout_date"
                ]

            if "num_rooms" not in self.search_criteria:
                self.search_criteria["num_rooms"] = 1

    def select_booking_type(self, booking_type: List[str]):
        # To select the booking type from ['hotels', 'trip tickets', 'both'].
        assert booking_type in [
            "hotels",
            "trip tickets",
            "both",
        ], f"booking type '{booking_type}' not valid"
        assert self.search_criteria == {}, "need to set booking type first"
        self.search_criteria["booking_type"] = booking_type

    # trip ticket
    def select_transportation(self, transportation_type: str):
        # To select the transportation type from ['flight', 'train', 'bus', 'cruise'].
        assert transportation_type in [
            "flight",
            "train",
            "bus",
            "cruise",
        ], f"transportation type '{transportation_type}' not valid"
        self.search_criteria["means_of_transportation"] = transportation_type

    def set_origin(self, value: Loc):
        # To set the location for departure, given a Loc object.
        assert type(value) == Loc, "origin not a Loc object"
        self.search_criteria["location_from"] = value.location

    def set_destination(self, value: Loc):
        # To set the location for arrival, given a Loc object.
        assert type(value) == Loc, "destination not a Loc object"
        self.search_criteria["location_to"] = value.location

    def set_min_ticket_price(self, value: int):
        # To set minimum ticket price.
        self.search_criteria["min_price_ticket"] = value

    def set_max_ticket_price(self, value: int):
        # To set maximum ticket price.
        self.search_criteria["max_price_ticket"] = value

    def set_num_adults(self, value: int):
        # To set the number of adult tickets to purchase.
        self.search_criteria["num_adults"] = value

    def set_num_children(self, value: int):
        # To set the number of child tickets to purchase.
        self.search_criteria["num_children"] = value

    def set_departure_date(self, value: Date):
        # To set the departure date of the trip, given a Date object.
        assert type(value) == Date, "departure date not a Date object"
        self.search_criteria["departure_date"] = value.date

    def set_return_date(self, value: Date):
        # To set the return date of the trip, given a Date object.
        assert type(value) == Date, "return date not a Date object"
        self.search_criteria["return_date"] = value.date

    # hotel
    def set_hotel_location(self, value: Loc):
        # To set the location for hotel search, given a Loc object.
        assert type(value) == Loc, "hotel location not a Loc object"
        self.search_criteria["location"] = value.location

    def set_checkin_date(self, value: Date):
        # To set the hotel check-in date, given a Date object.
        assert type(value) == Date, "checkin date not a Date object"
        self.search_criteria["checkin_date"] = value.date

    def set_checkout_date(self, value: Date):
        # To set the hotel check-out date, given a Date object.
        assert type(value) == Date, "checkout date not a Date object"
        self.search_criteria["checkout_date"] = value.date

    def set_num_rooms(self, value: int):
        # To set the number of hotel rooms to book.
        self.search_criteria["num_rooms"] = value

    def select_room_type(self, value: str):
        # To select the hotel room type from ['King Bed', 'Queen Bed', 'Double', 'Luxury'].
        self.search_criteria["room_type"] = value

    def set_min_room_price(self, value: int):
        # To set minimum hotel room price.
        self.search_criteria["min_price_room"] = value

    def set_max_room_price(self, value: int):
        # To set maximum hotel room price.
        self.search_criteria["max_price_room"] = value


class BookingEvaluator(BaseEvaluator):
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
                    "label_dict": sorted(label_dict.items()),
                }
                best_f1 = f1

        self.full_results.append(results)
        return results

    def exec_code(self, code):
        API = BookingAPI()

        try:
            exec(code, {"API": API, "Date": Date, "Loc": Loc})
            result = API.search_criteria
            assert (
                type(result) is dict
            ), f"expected type(result) == <class 'dict'>, but get type(result) = {type(result)}"
        except Exception as e:
            return e, {}

        return None, result
