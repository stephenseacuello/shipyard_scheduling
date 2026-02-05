from setuptools import setup, find_packages

setup(
    name="shipyard_scheduling",
    version="0.1.0",
    description="Health-aware shipyard scheduling with reinforcement learning and simulation",
    author="Shipyard Scheduling Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "gymnasium>=0.26",
        "simpy>=4.0",
        "networkx>=3.0",
        "numpy>=1.20",
        "pandas>=1.3",
        "torch>=2.0",
        "torch-geometric>=2.4",
        "dash>=2.10",
        "plotly>=5.15",
        "flask>=2.3",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "shipyard-train=shipyard_scheduling.experiments.train:main",
            "shipyard-evaluate=shipyard_scheduling.experiments.evaluate:main",
            "shipyard-dashboard=shipyard_scheduling.mes.app:main",
        ]
    },
    python_requires=">=3.8",
)