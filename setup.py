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

from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Action Generation Evaluation Suite"
LONG_DESCRIPTION = "A package that makes it easy to evaluate LLM pipelines on real world action generation tasks."

REQUIRED = [
    "farm-haystack==1.15.1",
    "manifest-ml==0.1.3",
    "faiss-cpu==1.7.2",
    "numpy==1.22.4",
    "scipy==1.10.1",
    "shapely==2.0.1",
    "astunparse==1.6.3",
    "pygments==2.15.0",
    "pybullet==3.2.5",
    "gdown==4.7.1",
    "moviepy==1.0.3",
    "gspread==5.8.0",
    "gspread_dataframe==3.3.0",
    "gspread_formatting==1.1.2",
    "black==23.3.0",
]

setup(
    name="toolbench-eval",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Qiantong Xu",
    author_email="xqt0904@gmail.com",
    license="MIT",
    python_requires=">=3.8.0",
    packages=find_packages(exclude=["tests", "scripts", "data", "api"]),
    install_requires=REQUIRED,
    keywords="action generation evaluation",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
