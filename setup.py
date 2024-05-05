from setuptools import find_packages, setup

setup(
    name="npgrad",
    version="0.1",
    url="https://github.com/lulor/npgrad",
    package_data={"npgrad": ["py.typed"]},
    packages=find_packages(),
    install_requires=["numpy"],
)
