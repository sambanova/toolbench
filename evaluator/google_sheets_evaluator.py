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
import os
import pandas as pd
import numpy as np

import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from gspread_formatting import get_effective_format
from gspread.exceptions import APIError
import backoff

from evaluator import BaseEvaluator


def on_error(err):
    assert err["exception"].response.status_code == 429, err
    print("GoogleSheetsEvaluator is backing off on rate limit.")


class GoogleSheetsEvaluator(BaseEvaluator):
    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def __init__(self, generator):
        super().__init__(generator)

        api_json = os.environ.get("GOOGLE_CLOUD_CREDENTIAL")
        test_sheet_key = "1dgsg17hqRHkrJnKvWQyFwinMJNrsi1z2uhWNiJCUVIQ"
        self.gc = gspread.service_account(api_json)
        self.test_sheet = self.gc.open_by_key(test_sheet_key)
        self.pid = os.getpid()
        self.sheet = self.gc.create(f"test_spreadsheet_dup_{self.pid}")

    # @backoff.on_exception(backoff.constant, APIError, jitter=None, interval=2)
    # def __del__(self):
    #     self.gc.del_spreadsheet(self.sheet.id)

    def aggregate_results(self):
        total_crashed = sum([s["crashed"] for s in self.full_results])
        total_success = sum([s["success"] for s in self.full_results])
        n_samples = len(self.full_results)
        results = {
            "crash_rate": total_crashed / n_samples,
            "success_rate": total_success / n_samples,
        }
        return results

    def check_results(self, pred_df, label_df, pred_format, label_format):
        if not pred_df.equals(label_df):
            return False

        if pred_format != label_format:
            return False

        return True

    def __call__(
        self,
        query: str,
        labels: List[str],
        question_sheet_name: str,
        check_format=False,
    ):
        prompt, prediction, crash_error = self.get_action(query)
        if crash_error:
            results = {
                "crashed": True,
                "success": False,
                "prompt": prompt,
                "query": query,
                "prediction": prediction,
                "closest_label": labels[0],
                "crashed_error_msg": f"Language model generation error: {str(crash_error)}",
            }
            self.full_results.append(results)
            return results
        return self.evaluate(
            query, prompt, prediction, labels, question_sheet_name, check_format
        )

    def evaluate(
        self,
        query: str,
        prompt: str,
        prediction: str,
        labels: List[str],
        question_sheet_name: str,
        check_format=False,
    ):
        # skip execution if prediction matches any label verbatim
        prediction = prediction.strip("\n")
        for label in labels:
            if prediction == label:
                results = {
                    "crashed": False,
                    "success": True,
                    "prompt": prompt,
                    "query": query,
                    "prediction": prediction,
                    "closest_label": label,
                }
                self.full_results.append(results)
                return results

        # Get the raw test data
        test_worksheet = self.get_worksheet_safe(
            self.test_sheet, worksheet_name=question_sheet_name
        )
        df = self.get_as_dataframe_safe(test_worksheet)

        # Execute prediction
        crash_error, pred_df, pred_format = self.run(prediction, df, check_format)
        if crash_error:
            results = {
                "crashed": True,
                "success": False,
                "prompt": prompt,
                "query": query,
                "prediction": prediction,
                "closest_label": labels[0],
                "crashed_error_msg": f"Task execution error: {str(crash_error)}",
            }
            self.full_results.append(results)
            return results

        # Execute label
        for label in labels:
            crash_error, label_df, label_format = self.run(label, df, check_format)
            assert crash_error == None, f"label: {label}\nerror: {str(crash_error)}"

            success = self.check_results(pred_df, label_df, pred_format, label_format)
            results = {
                "crashed": False,
                "success": success,
                "prompt": prompt,
                "query": query,
                "prediction": prediction,
                "closest_label": label,
                "pred_df": pred_df.to_string(),
                "label_df": label_df.to_string(),
            }
            if success:
                break
        self.full_results.append(results)
        return results

    def get_format(self, ws, shape):
        format = []
        n_row, n_col = shape
        for j in range(n_row):
            for k in range(n_col):
                label = chr(ord("A") + k) + str(j + 1)
                cell_format = self.get_cell_format(ws, label)
                format.append(cell_format)
        return format

    def run(self, code, df, check_format):
        self.delete_redundant_worksheets_safe()
        worksheet = self.add_worksheet_safe(self.sheet)
        worksheet = self.set_with_dataframe_safe(worksheet, df)

        try:
            self.exce_safe(code, self.sheet, worksheet)
        except Exception as e:
            self.delete_redundant_worksheets_safe()
            return e, [], []

        ws = self.get_worksheet_safe(self.sheet)
        df = self.get_as_dataframe_safe(ws)

        format = []
        if check_format:
            format = self.get_format(ws, df.shape)

        self.delete_redundant_worksheets_safe()
        return None, df, format

    ########################
    # Gspread API wrappers #
    ########################
    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def add_worksheet_safe(self, sheet):
        n_row, n_col = 1000, 26
        worksheet = sheet.add_worksheet(title="sheet_1", rows=n_row, cols=n_col)
        return worksheet

    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def set_with_dataframe_safe(self, worksheet, df):
        worksheet.clear()
        set_with_dataframe(
            worksheet, df, include_index=False, include_column_header=True
        )
        return worksheet

    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def delete_redundant_worksheets_safe(self):
        while len(self.sheet.worksheets()) > 1:
            ws = self.sheet.get_worksheet(1)
            self.sheet.del_worksheet(ws)

    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def get_as_dataframe_safe(self, worksheet):
        df = get_as_dataframe(worksheet, evaluate_formulas=True)
        df = df.dropna(how="all", axis=0)
        df = df.dropna(how="all", axis=1)
        return df

    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def get_cell_format(self, worksheet, label):
        cell_format = get_effective_format(worksheet, label)
        return cell_format

    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def exce_safe(self, code, sheet, worksheet):
        exec(
            code,
            {
                "sheet": sheet,
                "worksheet": worksheet,
                "get_as_dataframe": get_as_dataframe,
                "set_with_dataframe": set_with_dataframe,
                "np": np,
                "pd": pd,
            },
        )

    @backoff.on_exception(
        backoff.constant, APIError, jitter=None, interval=2, on_backoff=on_error
    )
    def get_worksheet_safe(self, sheet, worksheet_name=None):
        if worksheet_name:
            ws = sheet.worksheet(worksheet_name)
        else:
            worksheet_idx = len(sheet.worksheets()) - 1
            ws = sheet.get_worksheet(worksheet_idx)
        return ws


if __name__ == "__main__":
    prediction = 'df = get_as_dataframe(worksheet)\ndf = df.dropna(how="all", axis=0)\ndf = df.dropna(how="all", axis=1)\ndf = df.drop([2])\nset_with_dataframe(worksheet, df, include_index=False, include_column_header=True)'
    label = "worksheet.delete_rows(4)"

    prediction = "df = get_as_dataframe(worksheet)\ndf.loc[df['Product'] == 'chicken', 'Cost'] += 2\nset_with_dataframe(worksheet, df, include_index=False, include_column_header=True)"
    label = "worksheet.update('C4', 13, raw=False)"

    e = GoogleSheetsEvaluator(None)
    results = e.evaluate(
        "Update cell D4 to asd",
        "Update cell D4 to asd",
        prediction,
        [label],
        "Sheet1",
        False,
    )
    print(results)
