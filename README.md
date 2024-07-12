# mesh_masking
A tool for creating vertex mesh masks written in Python



https://github.com/user-attachments/assets/2e65b226-f1aa-4573-b141-2d3aba7d1bce



## Usage

`python mesh_masking.py <path to mesh> <path to mask> <-im to store the mask as vertex indices sequence>`

example:

`python mesh_masking.py meshes/mesh.obj masks/nose.npy`

To save the result, quit the application by closing the window or pressing `q`.

## Notes

This tool uses Pyvista and PyvistaQt to interactively mark vertices of a 3D mesh as masked or not. Vertex masks are useful for many applications where only a subset of a mesh is relevant. 

## Controls

The controls are displayed in the console when launching the application.

| Key    | Action |
| -------- | ------- |
| Space  | Mark the area under the cursor as masked    |
| Left control | Unmask the area under the cursor if masked     |
| Numpad Plus    | Increase selection area size |
| Numpad Minus    | Decrease selection area size |

## Color coding

The mesh is colored in real-time to ease the process.

| Color    | Meaning |
| -------- | ------- |
| <span style="color:lightblue">blue</span>  | Default |
| <span style="color:red">red</span>  | masked |
| <span style="color:green">green</span>  | Under the cursor |

## Requirements
- python 3.7+
- pyvista
- pyvistaqt
- trimesh
- numpy
- click
