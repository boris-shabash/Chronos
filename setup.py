from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="chronos-forecast", 
    version="0.0.5",
    author="Boris Shabash",
    license="MIT",
    author_email="boris.shabash@gmail.com",
    description="Time series prediction using probabilistic programming",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boris-shabash/Chronos",
    py_modules = ["chronos", "chronos_utils", "chronos_plotting"],
    package_dir={"": "chronos"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.8',    
    install_requires=["pandas>=1.1.5",
                      "numpy>=1.19.2",
                      "matplotlib>=3.2.2",
                      "torch>=1.5.1",
                      "pyro-ppl>=1.3.1"],
    extras_require={
        "dev": ["pytest==6.2.1"]
    }
)