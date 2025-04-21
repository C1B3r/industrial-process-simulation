from setuptools import setup, find_packages

setup(
    name="industrial_process_simulation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch==2.1.0",
        "torchvision==0.15.0",
        "pandas==2.0.0",
        "scikit-learn==1.2.2",
        "matplotlib==3.7.1",
        "pytest==7.1.2",
        "numpy==1.24.2",
        "tqdm==4.65.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A GAN-based system for simulating industrial processes.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/industrial-process-simulation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
