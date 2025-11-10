# How to install as a library
## In uv project
Add following components in root project.toml.
```
[project]
dependencies = ["viser-wrapper"]

[tool.uv.sources]
viser-wrapper = { git = "<url/of/this/git-reop>"}
```

## In pip project
```
pip install <url/of/this/git-reop>
```
# How to call functions
## Multiview videos scene
```
from viser_wrapper import run_multiview_videos_viser

_ = run_multiview_videos_viser(
    points,  # np.ndarray. Shape: (V, F, H, W, 3). Type: np.floating. V is number of videos (cameras), F is number of frames (timesteps).
    extrinsics,  # np.ndarray. Shape: (V, F, 3, 4). Type: np.floating
    intrinsics,  # np.ndarray. Shape: (V, F, 3, 3). Type: np.floating
)
```
Details are in docstrings of the function.
