from setuptools import setup, find_packages

setup(
    name="QwenEdit",
    version="0.1.0",
    author="Rico",
    packages=find_packages(), 
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
    ],
    python_requires=">=3.10",
)
