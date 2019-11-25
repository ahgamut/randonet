from setuptools import setup

helpfile = open("README.md", "r")
readme_txt = helpfile.read()
helpfile.close()

setup(
    name="randonet",
    version="0.0.0a",
    author="Gautham Venkatasubramanian",
    author_email="ahgamut@gmail.com",
    description="randomly generating neural networks",
    long_description=readme_txt,
    install_requires=["numpy", "click", "torch"],
    url="https://github.com/ahgamut/randonet",
    package_dir={"": "src"},
)
