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

from action_generator.components.retriever import (
    RetrieverWithBM25,
    RetrieverWithEmbedding,
)


def test_bm25_retriever():
    retriever = RetrieverWithBM25("data/virtual_home/v0/functions", top_k=3)
    candidate_documents = retriever("Apply lotion on agent face")
    candidate_documents_str = "".join(candidate_documents)
    expected_str = """# Lie on an object. 'object' can only be: ['bed', 'couch']. 
Agent.LieOn(object)
# Type on an object. 'object' can only be: ['keyboard', 'phone']. 
Agent.TypeOn(object)
# Sit on an object. 'object' can only be: ['bed', 'chair', 'toilet', 'couch', 'love_seat']. 
Agent.SitOn(object)
"""
    assert candidate_documents_str == expected_str


def test_dense_retriever():
    # No dataset's prepared in this way for now
    return
    retriever = RetrieverWithEmbedding("data/virtualhome/v0/functions", top_k=10)
    candidate_documents = retriever("Apply lotion on agent face")
    print("".join(candidate_documents))
    print(" ------------------ ")
