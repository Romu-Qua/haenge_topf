# 3D Fractal Tree Generator

## Overview
This Blender script generates a 3D-printable fractal tree structure that resembles natural growth. The tree is created using recursive branching constrained between two conical boundaries, ensuring that the model is suitable for 3D printing without requiring additional support structures.
The corresponding pot files are in the Blender project. 
## Features
- Generates a fractal tree structure with customizable parameters.
- Constrained growth between two cones to ensure printability.
- Automatically integrates with provided plant pot models.
- Creates independent copies of objects within Blender.

## Requirements
- Blender (version 2.8 or higher recommended)
- Python and some math packages


## Installation & Usage
1. Clone this repository or download the script file.
2. Open the Blender project and switch to the Scripting workspace.
3. Open the python script
4. Ensure that the required pot models ("Bottom_to_Copy" and "TOP_to_Copy") are present in your Blender scene, the script will create the hanging structure. It might take some time to execute, you can check the Blender terminal to see if it runs.



![Fractal Tree Example](https://github.com/Romu-Qua/haenge_topf/blob/master/rendered_model.png?raw=true)
