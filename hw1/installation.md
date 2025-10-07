We recommend you to use our pre-configured docker image `luminoctua/gcb6206:latest`, which already has all necessary packages for this homework.
If you want to set up your own environment, please follow the instructions below to install necessary packages.

## Install libraries

1. Update your apt
	```
	export DEBIAN_FRONTEND=noninteractive
	apt-get update
	```

2. Install the libraries with the apt
	```
	apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 libgl1-mesa-dev libglew-dev patchelf ffmpeg
	```

## Install dependencies

There are two options:

A. (Recommended) Install with conda:

1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

2. Create a conda environment that will contain python 3:
	```
	conda create -n gcb6206 python=3.10
	```
3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	conda activate gcb6206
	```
4. Install the requirements into this conda environment
	```
	pip install swig
	pip install -r requirements.txt
	```
5. Allow your code to be able to see 'gcb6206'
	```
	cd <path_to_hw1>
	$ pip install -e .
	```

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


B. Install on system Python

1. Install the requirements into this environment
	```
	pip install swig
	pip install -r requirements.txt
	```

2. Allow your code to be able to see 'gcb6206'
	```
	cd <path_to_hw1>
	$ pip install -e .
	```

## Troubleshooting

You may encounter the following GLFW errors if running on machine without a display:

GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
  warnings.warn(message, GLFWError)
GLFWError: (65537) b'The GLFW library is not initialized'

These can be resolved with:
```
export MUJOCO_GL=egl
```
