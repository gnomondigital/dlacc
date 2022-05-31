import io
import re

from setuptools import find_packages
from setuptools import setup

with io.open("README.md", encoding="utf8") as f:
    readme = f.read()

setup(
    name="dlacc",
    version=1.0,
    url="https://gitlab.gnomondigital.com/fzyuan/dl_acceleration",
    project_urls={
    },
    license="BSD-3-Clause",
    author="author",
    author_email="author@gmail.com",
    description="A simple framework for accelerating deep learning inference runtime.",
    long_description=readme,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Web Environment",
        "Framework :: Flask",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.9",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "onnx",
        "onnxruntime",
        "pandas",
        "google-cloud-storage"
    ],
    extras_require={},
)