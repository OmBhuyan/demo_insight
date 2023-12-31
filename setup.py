import io
import os
import os.path as op

from setuptools import PEP420PackageFinder  # isort: skip
from distutils.core import setup  # isort: skip


ROOT = op.dirname(op.abspath(__file__))
SRC = op.join(ROOT, "src")


def get_install_req():
    if os.name == "nt":
        with io.open("deploy/requirements_windows.txt") as fh:
            install_reqs = fh.read()
    else:
        with io.open("deploy/requirements_linux.txt") as fh:
            install_reqs = fh.read()
    install_reqs = [l for l in install_reqs.split("\n") if len(l) > 1]
    return install_reqs


install_reqs = get_install_req()


def get_version_info():
    """Extract version information as a dictionary from version.py."""
    version_info = {}
    version_filename = os.path.join("src", "query_insights", "version.py")
    with open(version_filename, "r") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
    exec(version_code, version_info)
    return version_info


setup(
    name="query_insights",
    version=get_version_info()["version"],
    package_dir={"": "src"},
    description="DS Innovation - NLP",
    author="Tiger Analytics",
    packages=PEP420PackageFinder.find(where=str(SRC)),
    python_requires=">=3.8.1",
    install_requires=install_reqs,
)
