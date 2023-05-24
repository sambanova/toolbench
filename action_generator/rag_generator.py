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

from action_generator.components.retriever import (
    RetrieverWithBM25,
    RetrieverWithEmbedding,
)
from action_generator.components.manifest_llm import ManifestLLM

from action_generator.base_generator import BaseGenerator


class RagGenerator(BaseGenerator):
    def __init__(
        self,
        client_name,
        client_connection=None,
        model_name=None,
        context_dir=None,
        max_output_token=128,
        stop_tokens=None,
        top_k_api=0,
        top_k_example=0,
        use_dense_retriever=False,
        query_template="{query}",
    ):
        self.query_template = query_template
        self.context_dir = context_dir
        self.api_retriever = None
        self.example_retriever = None

        ## 1. build retriever
        if top_k_api > 0:
            if use_dense_retriever:
                self.api_retriever = RetrieverWithEmbedding(
                    f"{context_dir}/functions", top_k=top_k_api
                )
            else:
                self.api_retriever = RetrieverWithBM25(
                    f"{context_dir}/functions", top_k=top_k_api
                )
        if top_k_example > 0:
            if use_dense_retriever:
                self.example_retriever = RetrieverWithEmbedding(
                    f"{context_dir}/examples", top_k=top_k_example
                )
            else:
                self.example_retriever = RetrieverWithBM25(
                    f"{context_dir}/examples", top_k=top_k_example
                )

        ## 2. build predictor
        stop_tokens = stop_tokens if stop_tokens else []
        stop_tokens += ["------", "Task:"]

        self.llm = ManifestLLM(
            client_name=client_name,
            model_name=model_name,
            client_connection=client_connection,
            max_output_token=max_output_token,
            stop_tokens=stop_tokens,
        )

    def make_prompt(self, query: str):
        api_docs = ""
        if self.api_retriever:
            candidate_api_functions = self.api_retriever(query)
            api_docs += "I have the following set of API:\n\n"
            for d in candidate_api_functions:
                api_docs += d + "\n"
            api_docs += "-------------"

        examples_str = ""
        if self.example_retriever:
            examples = self.example_retriever(query)
            examples_str += "I have the following set of examples:\n\n"
            for d in examples:
                examples_str += d + "\n"
            examples_str += "-------------"

        return self.query_template.format(
            api_docs=api_docs, examples=examples_str, query=query
        )

    def generate(
        self,
        query: str,
        additional_stop_tokens: List = None,
        max_output_len: int = None,
    ):
        prompt = self.make_prompt(query)
        text, error = self.llm.generate(prompt, additional_stop_tokens, max_output_len)
        if text:
            text = text.strip()
        return prompt, text, error
