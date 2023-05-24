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


class BaseGenerator(metaclass=abc.ABCMeta):
    """A generator that will turn a query into a series of API calls"""

    @abc.abstractmethod
    def generate(self, query: str, stop_tokens=None, max_output_len=None):
        pass
