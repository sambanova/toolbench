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


def compute_f1_between_dicts(pred_dict, label_dict):
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
