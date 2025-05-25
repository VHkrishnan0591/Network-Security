from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

REPO_NAME = "Network-Security"
AUTHOR_USER_NAME = "VHkrishnan0591"
SRC_REPO = "src"
AUTHOR_EMAIL = "harikrishnanv0591@gmail.com"
VERSION = '1.0'
LICENSE = 'MIT'

def parse_documents(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines 
        

setup(
    name=REPO_NAME,  # name of the package
    version=VERSION,  # version of the package
    description="A small webapp predict the correctness of network",  # description of the package
    author=AUTHOR_USER_NAME, # author of the package
    author_email=AUTHOR_EMAIL, # email id of the author
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}", # homepage URL for the package, such as the GitHub repository or project documentation
    license=LICENSE, # license under which package will be distributed
    long_description = LONG_DESCRIPTION, # comprehensive description of your package, which will appear on your package’s PyPI page
    package_dir={"": "src"}, # directory where packages are located
    packages=find_packages(where="src"), # find all packages and list them
    python_requires=">=3.10", # compatible python version
    install_requires= parse_documents('requirements.txt'), # external packages required for our package to work
)