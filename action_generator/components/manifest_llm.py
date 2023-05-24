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

import os
from typing import List

from manifest import Manifest


class ManifestLLM:
    def __init__(
        self,
        client_name,
        model_name,
        client_connection=None,
        max_output_token=256,
        stop_tokens=None,
    ):
        self.client_name = client_name
        self.max_output_token = max_output_token
        self.stop_tokens = stop_tokens if stop_tokens else []

        if client_name == "openai":
            if model_name is None:
                model_name = "text-davinci-003"

            self.manifest = Manifest(
                engine=model_name,
                client_connection=os.environ.get("OPENAI_KEY"),
                client_name="openai",
                temperature=0,
                max_tokens=max_output_token,
                cache_name="sqlite",
                cache_connection=f"openai_{model_name}.sqlite",
            )
        elif client_name == "openaichat":
            if model_name is None:
                model_name = "gpt-3.5-turbo"

            self.manifest = Manifest(
                engine=model_name,
                client_connection=os.environ.get("OPENAI_KEY"),
                client_name="openaichat",
                temperature=0,
                max_tokens=max_output_token,
                cache_name="sqlite",
                cache_connection=f"openaichat_{model_name}.sqlite",
            )
        elif client_name == "cohere":
            self.manifest = Manifest(
                client_connection=os.environ.get("COHERE_KEY"),
                client_name="cohere",
                max_tokens=max_output_token,
                cache_name="sqlite",
                cache_connection="cohere.sqlite",
            )
        elif client_name == "huggingface":
            assert client_connection.startswith("http")
            tmp_name = model_name.replace("/", "-")
            self.manifest = Manifest(
                client_connection=client_connection,
                client_name="huggingface",
                do_sample=False,
                max_tokens=max_output_token,
                cache_name="sqlite",
                cache_connection=f"huggingface_{tmp_name}.sqlite",
            )
        else:
            raise ValueError("unsupported model:", client_name)

    def generate(self, prompt, stop_tokens: List = None, max_output_token: int = None):
        if max_output_token is None:
            max_output_token = self.max_output_token
        stop_tokens = stop_tokens if stop_tokens else []
        stop_tokens += self.stop_tokens
        stop_tokens = list(set(stop_tokens))
        stop_tokens = stop_tokens if len(stop_tokens) > 0 else None
        if stop_tokens and "openai" in self.client_name:
            # Openai cannot handle more than 4 stop sequences
            stop_tokens = stop_tokens[-4:]

        try:
            result_object = self.manifest.run(
                prompt,
                return_response=True,
                stop_sequences=stop_tokens,
                max_tokens=max_output_token,
            )
            text = result_object.get_json_response()["choices"][0]["text"]

            if self.client_name == "huggingface" and stop_tokens:
                for i in range(len(text)):
                    new_text = text[: i + 1]
                    for s in stop_tokens:
                        if new_text.endswith(s):
                            return new_text[: -len(s)], None
            return text, None
        except Exception as e:
            return None, e
