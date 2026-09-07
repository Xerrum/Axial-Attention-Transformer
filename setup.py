from setuptools import setup, find_packages

setup(
    name="tud-presence-prediction",
    version="0.1.3",
    packages=find_packages(),
    package_data={
        "tud_presence_prediction": ["data/users/*.txt"],
    },
    install_requires=[
        "pytorch-lightning",
        "scikit-learn",
        "holidays==0.14.2",
        "requests==2.28.0",
    ],
)
