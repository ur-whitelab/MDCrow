from setuptools import find_packages, setup

# fake to satisfy mypy
__version__ = "0.0.0"
exec(open("mdagent/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="md-agent",
    version=__version__,
    description="Collection of MD tools for use with language models",
    author="Andrew White",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/md-agent",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "chromadb==0.4.24",
        "google-search-results",
        "langchain==0.0.336",
        "langchain_experimental",
        "matplotlib",
        "MDAnalysis",
        "nbformat",
        "openai",
        "outlines",
        "paper-qa==4.0.0rc8 ",
        "paper-scraper @ git+https://github.com/blackadad/paper-scraper.git",
        "pydantic>=2.6",
        "python-dotenv",
        "rdkit",
        "requests",
        "streamlit",
        "tiktoken",
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
