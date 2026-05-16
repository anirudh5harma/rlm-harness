from setuptools import find_packages, setup

setup(
    name="rlm-harness",
    version="0.1.0",
    description="A trace-first RLM coding agent harness.",
    packages=find_packages(include=["rlm_harness", "rlm_harness.*"]),
    install_requires=["pydantic>=2"],
    extras_require={
        "dev": ["pytest>=8", "ruff>=0.6"],
        "graph": ["langgraph>=0.2"],
        "mlx": ["mlx-lm"],
    },
    python_requires=">=3.9",
    entry_points={"console_scripts": ["rlm-harness=rlm_harness.cli:main"]},
)
