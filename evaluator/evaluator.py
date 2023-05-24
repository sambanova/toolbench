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

import abc
from typing import List


class BaseEvaluator(metaclass=abc.ABCMeta):
    """A evaluator that evaluates a action generator's execution accuracy."""

    def __init__(self, generator):
        # a list of dictionaries
        self.full_results = []

        # the action generator
        self.generator = generator

    def get_action(
        self, query: str, stop_tokens: List = None, max_output_len: int = None
    ):
        """Generate actions based on a given query.

        Args:
            query: samples to collate
            stop_tokens: a list of tokens to stop generation
            max_output_len: the maximum number of tokens to generate

        Returns:
            prompt: the real prompt send to the LLM
            prediction: the generated action
            error: the error message if the program crashes
        """
        prompt, prediction, error = self.generator.generate(
            query, stop_tokens, max_output_len
        )
        return prompt, prediction, error

    @abc.abstractmethod
    def __call__(self, query: str, labels: List[str]):
        """Generate action based a give query and check the result against the labels"""
        pass

    @abc.abstractmethod
    def aggregate_results(self):
        """Aggregate the results into self.full_results"""

        raise NotImplementedError
