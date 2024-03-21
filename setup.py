import platform

from setuptools import find_packages, setup


def do_setup():
    setup(
        name="kdd_integrated_anomaly_detection",
        version="0.0.1",
        description="KDD Integrated anomaly detection demo",
        packages=find_packages(),
        install_requires=[
            "cloudpickle",
            "mlflow",
            "numpy>=1.20",
            "pandas>=1.0",
            "scipy",
            "scikit-learn",
            "statsmodels",
            "torch",
            "prophet>=1.1.3",
            "tensorflow-macos" if platform.machine() == "arm64" else "tensorflow",
            "spectrum"
        ],
    )


if __name__ == "__main__":
    do_setup()
