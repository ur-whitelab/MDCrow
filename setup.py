from setuptools import find_packages, setup

# fake to satisfy mypy
__version__ = "0.0.0"
exec(open("mdcrow/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MDCrow",
    version=__version__,
    description="Collection of MD tools for use with language models",
    author="Andrew White",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/MDCrow",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "langchain==0.2.12",
        "langchain-community",
        "langchain-openai==0.1.19",
        "matplotlib",
        "nbformat",
        "openai",
        "paper-qa==5.0.6",
        "pandas",
        "pydantic>=2.6",
        "python-dotenv",
        "rdkit",
        "requests",
        "seaborn",
        "scikit-learn",
        "scipy==1.14.0",
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
