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

from action_generator import RagGenerator


def test_rag_generator():
    generator = RagGenerator(
        client_name="openai",
        model_name="text-curie-001",
        context_dir="data/home_search/v0",
        max_output_token=256,
        top_k_api=10,
        top_k_example=3,
        query_template="{api_docs}\n{examples}\nTask: {query}\nActions:\n",
    )
    query = "Find a home with 12 bed above $961000 in Birmingham."
    prompt, text, error = generator.generate(query)
    assert error is None, error
