from setuptools import setup, find_packages

setup(
    name="cann_bench",
    version="1.0.0",
    description="TileLang R4 fused-sampling operators for CANN Bench",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch_npu",
    ],
    python_requires=">=3.8",
)
