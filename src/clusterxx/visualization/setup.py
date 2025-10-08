from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="clusterxx_visualization",
    version="0.0.1",
    description="ClusterXX visualization helper API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Spiros Maggioros",
    author_email="spirosmag@ieee.org",
    maintainer="Spiros Maggioros",
    maintainer_email="spirosmag@ieee.org",
    download_url="https://github.com/spirosmaggioros/ClusterXX/",
    url="https://github.com/spirosmaggioros/ClusterXX/",
    packages=find_packages(exclude=[".github"]),
    python_requires=">=3.9",
    install_requires=[
        "matplotlib",
    ],
    entry_points={"console_scripts": ["clusterxx_visualization = clusterxx_visualization.__main__:main"]},
    license="MIT",
    keywords=[
        "data visualization",
    ],
    package_data={"clusterxx_visualization": ["VERSION"]},
)
