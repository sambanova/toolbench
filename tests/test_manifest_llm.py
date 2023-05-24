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

from action_generator.components.manifest_llm import ManifestLLM


def test_openai():
    query = "whys is grass green?"

    llm = ManifestLLM(
        client_name="openai", model_name="text-curie-001", max_output_token=256
    )
    text, error = llm.generate(query)
    assert error is None, error

    expected_str = """\n\nThe chlorophyll in grass absorbs sunlight to create energy which is then used by the plant to create glucose from water and carbon dioxide. This process creates a green color because the light is scattered in all directions and some of it is absorbed by the chlorophyll which gives off a green light."""
    assert text == expected_str
