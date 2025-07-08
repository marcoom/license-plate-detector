from setuptools import setup, find_packages
from pathlib import Path

def load_requirements(file_path: str) -> list:
    """Load requirements from a file, ignoring comments and empty lines."""
    requirements = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

# Load requirements
install_requires = load_requirements('requirements.txt')
dev_requires = load_requirements('requirements-dev.txt')

setup(
    name="license_plate_detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
    },
    python_requires=">=3.11",
)