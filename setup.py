from setuptools import setup, find_packages

setup(
    name="spelke_net",
    version="0.1",
    packages=find_packages(),
    description="SpelkeNet: Object segmentation through virtual poking",
    author="Stanford NeuroAI Lab",
    install_requires=[
        'pycocotools',
        'numpy==1.26.4',
        'torch==2.1.2',
        'scipy',
        'tqdm',
        'wandb',
        'einops',
        'matplotlib',
        'h5py',
        'torchvision',
        'future',
        'opencv-python',
        'decord',
        'pandas',
        'matplotlib',
        'moviepy',
        'scikit-image',
        'scikit-learn',
        'vector_quantize_pytorch',
        'google-cloud-storage',
        "lpips",
        "segment_anything",
        "ptlflow",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "spelkebench-infer = spelke_net.inference.spelke_object_discovery.run_inference:main",
            "spelkebench-launch = spelke_net.inference.spelke_object_discovery.run_inference_parallel:main",
            "spelkebench-evaluate = spelke_net.inference.spelke_object_discovery.evaluate_folder:main"
        ]
    }
)