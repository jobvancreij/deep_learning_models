import setuptools
import os
import sys
with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    required = f.read().splitlines()

if sys.version_info[0] != 3: #raise exception when not using python3
    raise ImportError("This repo needs python3 to be installed, you use python {}".format(sys.version_info[0]))

github_repos_installs = [
    "jobvancreij/LJT_helper_functions@master",
    "jobvancreij/LJT_database@master"
    ]

''' try to install it with python3 (needed for some systems) if that does not return 0 (success) install it 
    just for python. This helps to make it compatible for both linux and windows and prevents future problems 
'''
for repo in github_repos_installs:
    install = os.system('python3 -m pip install --upgrade git+ssh://git@github.com/{}'.format(repo))
    if install != 0:
        install = os.system('python -m pip install --upgrade git+ssh://git@github.com/{}'.format(repo))


setuptools.setup(
    name="deep_learning_models",
    version="0.0.2",
    author="Job van Creij & Lex Fons",
    author_email="jobvancrey@hotmail.com",
    description="models",
    long_description=long_description,
    url="https://github.com/jobvancreij/deep_learning_models",
    packages=setuptools.find_packages(),
    install_requires=required,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)