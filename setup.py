from setuptools import setup, find_packages

import re

def read_requirements():
    with open('requirements.txt') as req:
        requirements = req.read().splitlines()
    
    # Exclude editable requirements
    requirements = [req for req in requirements if not req.startswith('-e')]
    
    # Remove version specifiers
    #requirements = [re.split(r'[<>=]', req)[0] for req in requirements]
    
    return requirements

def read_links():
    with open('requirements.txt') as req:
        requirements = req.read().splitlines()
    
    # Include only editable requirements
    links = [req for req in requirements if req.startswith('-e')]
    
    # Remove the '-e' prefix
    links = [req.split('-e ')[1] for req in links]
    
    return links
    
setup(
    name="FLOWGEN",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements(),
    dependency_links=read_links(),
    author="Fernando Gonzalez",
    author_email="gonzalez@cerfacs.fr",
    description="Repository for experiments of the flowen project",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ferngonzalezp/flowgen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={},
)