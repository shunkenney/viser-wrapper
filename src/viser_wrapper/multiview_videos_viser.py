import time
import threading
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
from jaxtyping import Float, Bool, Integer
from beartype import beartype


@beartype
def run_multiview_videos_viser(
    points: Sequence[Sequence[Float[NDArray[np.floating], "H W 3"]]] | Float[NDArray[np.floating], "V F H W 3"],
    extrinsics: Sequence[Sequence[Float[NDArray[np.floating], "3 4"]]] | Float[NDArray[np.floating], "V F 3 4"],
    intrinsics: Sequence[Sequence[Float[NDArray[np.floating], "3 3"]]] | Float[NDArray[np.floating], "V F 3 3"],
    masks: Sequence[Sequence[Bool[NDArray[np.bool_], "H W"]]] | Bool[NDArray[np.bool_], "V F H W"] | None = None,
    confs: Sequence[Sequence[Float[NDArray[np.floating], "H W"]]] | Float[NDArray[np.floating], "V F H W"] | None = None,
    images: Sequence[Sequence[Integer[NDArray[np.integer], "H W 3"]]] | Integer[NDArray[np.integer], "V F H W 3"] | None = None,
    port: int | None = 8080,
    init_conf_threshold: float | None = 0.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    background_mode: bool | None = False,
    verbose: bool | None = True,
) -> viser.ViserServer:
    """
    Visualize 3D points and camera poses with viser.
    \nBasic shape of each variable is (V, F, ...), where V is number of videos (cameras), F is number of frames (timesteps).
    \nYou can provide them with totally ndarrays with shape of (V, F, ...), or list of list of ndarrays with shape of (...).

    Args:
        points (type: list[list[NDArray[np.floating]]], shape: (V, F, H, W, 3)): 3D points in world coordinates.
        extrinsics (type: list[list[NDArray[np.floating]]], shape: (V, F, 3, 4)): Camera extrinsics. Camera to world.
        intrinsics (type: list[list[NDArray[np.floating]]], shape: (V, F, 3, 3)): Camera intrinsics.
        masks (Optional. type: list[list[NDArray[np.bool_]]], shape: (V, F, H, W)): Binary masks for each point.
        confs (Optional. type: list[list[NDArray[np.floating]]], shape: (V, F, H, W)): Confidence scores for each point.
        images (Optional. type: list[list[NDArray[np.integer]]], shape: (V, F, H, W, 3)): Input images.
        port (Optional. type: int): Port number for the viser server.
        init_conf_threshold (Optional. type: float): Initial percentage of low-confidence points to filter out.
        background_mode (Optional. type: bool): Whether to run the server in background thread.
        verbose (Optional. type: bool): Whether to print verbose logs.
    """
    if verbose:
        print(f"Starting viser server on port {port}")

    V, F = len(points), len(points[0])
    H, W = points[0][0].shape[:2]
    if verbose:
        print(f"Number of videos (cameras): {V}")
        print(f"Number of frames (timesteps): {F}")
        print(f"Image size (W x H): {W}x{H}")

    server = viser.ViserServer(host="127.0.0.1", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Reshape points.
    if not isinstance(points, np.ndarray):
        points = np.array(points)    
    points = points.reshape(V, F, H * W, 3)
    
    # Colors for each point. If images is None, set points black.
    if images is not None:
        if not isinstance(images, np.ndarray):
            images = np.array(images)
        images = images.reshape(V, F, H, W, 3)
        colors = images.reshape(V, F, H * W, 3).copy()
    else:
        colors = np.zeros_like(points)
    
    # Reshape masks.
    if masks is not None:
        if not isinstance(masks, np.ndarray):
            masks = np.array(masks)
        masks = masks.reshape(V, F, H * W)

    # Reshape confs.
    if confs is not None:
        if not isinstance(confs, np.ndarray):
            confs = np.array(confs)
        # Apply color map (COLORMAP_JET is commonly used)
        conf_min = confs.min(axis=(-1, -2), keepdims=True)
        conf_norm = (confs - conf_min) / (confs.max(axis=(-1, -2), keepdims=True) - conf_min + 1e-8)
        conf_255 = (conf_norm * 255).astype(np.uint8)
        conf2colors_raw = [[] for _ in range(V)]
        conf2colors_equalized = [[] for _ in range(V)]
        for camera_idx in range(V):
            for frame_idx in range(F):
                conf2colors_raw[camera_idx].append(cv2.applyColorMap(conf_255[camera_idx][frame_idx], cv2.COLORMAP_JET)[..., ::-1])  # Convert BGR to RGB
                conf2colors_equalized[camera_idx].append(cv2.applyColorMap(cv2.equalizeHist(conf_255[camera_idx][frame_idx]), cv2.COLORMAP_JET)[..., ::-1])
        conf2colors_raw = np.array(conf2colors_raw).reshape(V, F, H * W, 3)  # shape (V, F, H * W, 3)
        conf2colors_equalized = np.array(conf2colors_equalized).reshape(V, F, H * W, 3)  # shape (V, F, H * W, 3)
        # Reshape confs.
        confs = confs.reshape(V, F, H * W)

    # We will store references to point_clouds $ frames & frustums so we can toggle visibility from the GUI.
    point_clouds: list[list[viser.PointCloudHandle]] = []
    frames: list[list[viser.FrameHandle]] = []
    frustums: list[list[viser.CameraFrustumHandle]] = []

    with server.gui.add_folder("Config"):
        # Build the viser GUI
        gui_show_point_clouds = server.gui.add_checkbox("Show Point Clouds", initial_value=True)
        gui_show_frames = server.gui.add_checkbox("Show Frames", initial_value=False)
        gui_show_frustums = server.gui.add_checkbox("Show Frustums", initial_value=True)
        gui_change_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.03, step=0.001, initial_value=0.01)
        if masks is not None:
            gui_points_mask = server.gui.add_checkbox("Apply Mask", initial_value=True)
        if confs is not None:
            gui_points_conf = server.gui.add_slider("Confidence percent threshold to prune", min=0, max=100, step=0.1, initial_value=init_conf_threshold)
            gui_point_color_mode = server.gui.add_dropdown("Point Color Mode", options=("Frame", "Confidence (raw)", "Confidence (equalized)"), initial_value="Frame",
                hint="Choose point color mode between original frame colors and other colors.")
        gui_prune_points = server.gui.add_slider("Prune Point Clouds", min=0, max=13, step=1, initial_value=min(int(V * F / 100), 6), 
            hint="Prune point clouds by this frequency (2^value).\nHigher value means lighter visualization.\n0 is no pruning.")
        
    def apply_standard_visibility() -> None:
        with server.atomic():
            for c in range(V):
                cam_vis = bool(camera_vis[c].value)
                for f in range(F):
                    point_clouds[c][f].visible = cam_vis and bool(gui_show_point_clouds.value)
                    frames[c][f].visible       = cam_vis and bool(gui_show_frames.value)
                    frustums[c][f].visible     = cam_vis and bool(gui_show_frustums.value)
                    frustums[c][f].color = (0, 0, 0)

    def apply_animation_visibility(t: int) -> None:
        t = int(t)
        mode = gui_anim_mode.value  # str
        with server.atomic():
            for c in range(V):
                cam_vis = bool(camera_vis[c].value)
                for f in range(F):
                    if mode == "Current frame only":
                        is_visible_step = (f == t)
                    elif mode == "Cumulative (up to frame)":
                        is_visible_step = (f <= t)
                    else:
                        raise ValueError(f"Unknown animation mode: {mode}")

                    point_clouds[c][f].visible = is_visible_step and cam_vis and bool(gui_show_point_clouds.value)
                    frames[c][f].visible = is_visible_step and cam_vis and bool(gui_show_frames.value)
                    frustums[c][f].visible = is_visible_step and cam_vis and bool(gui_show_frustums.value)
                    # Latest frame frustum will be red.
                    frustums[c][f].color = (0, 0, 0) if f != t else (255, 0, 0)

    camera_vis:list[viser.GuiCheckboxHandle] = []

    with server.gui.add_folder("Camera Controls"):
        gui_all_cameras_vis = server.gui.add_button(label="All cameras")

        for camera_idx in range(V):
            gui_camera_vis = server.gui.add_checkbox(label=f"camera{camera_idx:05d}", initial_value=True)
            camera_vis.append(gui_camera_vis)
            @gui_camera_vis.on_update
            def _(_, c_idx=camera_idx, handle=gui_camera_vis) -> None:
                vis = bool(handle.value)
                for f_idx in range(F):
                    point_clouds[c_idx][f_idx].visible = vis and gui_show_point_clouds.value
                    frames[c_idx][f_idx].visible = vis and gui_show_frames.value
                    frustums[c_idx][f_idx].visible = vis and gui_show_frustums.value
                if animation_active["on"]:
                    apply_animation_visibility(int(gui_timestep.value))

    @gui_all_cameras_vis.on_click
    def _(_) -> None:
        for camera in camera_vis:
            camera.value = True

        if animation_active["on"]:
            apply_animation_visibility(int(gui_timestep.value))
        else:
            apply_standard_visibility()

    animation_active = {"on": False}
    gui_animation_toggle = server.gui.add_button(label="Start animation")  # toggle of "Start animation" and "Finish animation"

    @gui_animation_toggle.on_click
    def _(_):
        animation_active["on"] = not animation_active["on"]
        gui_animation_toggle.label = "Finish animation" if animation_active["on"] else "Start animation"
        _set_animation_ui_enabled(animation_active["on"])
        if not animation_active["on"]:
            playing["on"] = False
            apply_standard_visibility()
        else:
            apply_animation_visibility(int(gui_timestep.value))

    # --- Animation (add) ---
    animation_folder = server.gui.add_folder("Animation")
    with animation_folder:
        # This crush the server somehow.
        gui_anim_mode = server.gui.add_dropdown(
            "Mode",
            options=("Current frame only", "Cumulative (up to frame)"),
            initial_value="Current frame only",
            hint="Choose how frames are shown during animation."
        )
        #gui_anim_mode = server.gui.add_button_group("Mode", ("Current frame only", "Cumulative (up to frame)"), initial_value="Current frame only")
        gui_control = server.gui.add_button_group("Control", ("Start/Stop", "Init"))
        gui_skip = server.gui.add_button_group("Skip", ("<<", "<", ">", ">>"))
        gui_timestep = server.gui.add_slider("Frame", min=0, max=F - 1, step=1, initial_value=0)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=1)
    playing = {"on": False}

    def _set_animation_ui_enabled(enabled: bool) -> None:
        animation_folder.visible = bool(enabled)
    
    _set_animation_ui_enabled(False)  # scene initialization

    def _step(delta: int) -> None:
        gui_timestep.value = (int(gui_timestep.value) + delta) % F  # wrap

    @gui_control.on_click
    def _(_):
        val = gui_control.value
        if not animation_active["on"]:
            return
        if val == "Start/Stop":
            playing["on"] = not playing["on"]
        elif val == "Init":
            gui_timestep.value = 0 
    
    @gui_skip.on_click
    def _(_):
        if not animation_active["on"]:
            return  # アニメーション無効時は操作できない
        val = gui_skip.value
        if val == "<<":
            _step(-10)
        elif val == "<":
            _step(-1)
        elif val == ">":
            _step(+1)
        elif val == ">>":
            _step(+10)

    def initialize_scene() -> None:
        # Add the point_clouds, frames and frustums to the scene.
        # Register frames and frustums to on_click method.

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        # Filter point clouds from initialization.
        if confs is not None:
            conf_percentage = gui_points_conf.value
        else:
            conf_percentage = None
        if masks is not None:
            apply_mask = gui_points_mask.value
        else:
            apply_mask = False
        prune_freq = int(2**gui_prune_points.value)

        for camera_idx in tqdm(range(V)):
            point_clouds_ = []
            frames_ = []
            frustums_ = []
            for frame_idx in range(F):
                # point clouds -----------------------------------------------
                valid_mask = get_valid_mask(camera_idx, frame_idx, prune_freq, conf_percentage, apply_mask)
                point = points[camera_idx][frame_idx][valid_mask]
                color = colors[camera_idx][frame_idx][valid_mask]
                
                point_clouds_.append(server.scene.add_point_cloud(
                    name=f"camera{camera_idx:05d}/frame{frame_idx:05d}/points",
                    points=point,
                    colors=color,
                    point_size=0.003,
                    point_shape="circle",
                ))

                # frames -----------------------------------------------
                cam2world_3x4 = extrinsics[camera_idx][frame_idx]
                T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

                # Add a small frame axis
                frame_axis = server.scene.add_frame(
                    f"camera{camera_idx:05d}/frame{frame_idx:05d}/frame",
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    axes_length=0.05,
                    axes_radius=0.002,
                    origin_radius=0.002,
                    visible=False
                )
                frames_.append(frame_axis)

                # frustums -----------------------------------------------
                # Convert the image for the frustum
                if images is not None:
                    image = images[camera_idx][frame_idx]  # shape (H, W, 3)

                K = intrinsics[camera_idx][frame_idx]
                fy = float(K[1, 1])
                fov = 2 * np.arctan2(H / 2, fy)

                # Add the frustum
                frustum = server.scene.add_camera_frustum(
                    f"camera{camera_idx:05d}/frame{frame_idx:05d}/frustum",
                    fov=fov,
                    aspect=W / H,
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    scale=0.05,
                    #image=image,  Image just make it to see GUI
                    line_width=1.0
                )
                frustums_.append(frustum)

                attach_callback(frustum, frame_axis)

            point_clouds.append(point_clouds_)
            frames.append(frames_)
            frustums.append(frustums_)

    def update_point_clouds() -> None:
        """
        Update the point cloud based on current GUI selections.
        3 components for update:
            - Confidence: Confidence percentage.
            - Mask: Binary mask indicating valid points.
            - Prune: Pruning some points to make visualization lighter.
        """
        if confs is not None:
            conf_percentage = gui_points_conf.value
        else:
            conf_percentage = None
        if masks is not None:
            apply_mask = gui_points_mask.value
        else:
            apply_mask = False
        prune_freq = int(2**gui_prune_points.value)

        for camera_idx in tqdm(range(V)):
            for frame_idx in range(F):
                point_cloud_visible = point_clouds[camera_idx][frame_idx].visible
                
                valid_mask = get_valid_mask(camera_idx, frame_idx, prune_freq, conf_percentage, apply_mask)

                # Apply valid_mask to the point clouds of specified index
                point =  points[camera_idx][frame_idx][valid_mask]
                if confs is not None and gui_point_color_mode.value == "Confidence (raw)":
                    color = conf2colors_raw[camera_idx][frame_idx][valid_mask]
                elif confs is not None and gui_point_color_mode.value == "Confidence (equalized)":
                    color = conf2colors_equalized[camera_idx][frame_idx][valid_mask]
                else:
                    color = colors[camera_idx][frame_idx][valid_mask]

                # Update the point cloud
                point_clouds[camera_idx][frame_idx].points = point
                point_clouds[camera_idx][frame_idx].colors = color
                point_clouds[camera_idx][frame_idx].visible = point_cloud_visible

    def get_valid_mask(camera_idx: int, frame_idx: int, prune_freq: int, conf_percentage: float | None, apply_mask: bool) -> NDArray[np.bool_]:
        """
        Get valid mask based on current GUI selections.
        """
        # Base masks.
        # Ex) When prune_freq = 4, [True, False, False, False, True, False, False, False, True,...].
        valid_mask = (np.arange(points[camera_idx][frame_idx].shape[0]) % prune_freq == 0)
        
        # Apply conf mask.
        if conf_percentage is not None:
            conf = confs[camera_idx][frame_idx]
            # Here we compute the threshold value based on the current percentage
            threshold_val = np.percentile(conf, conf_percentage)
            conf_mask = (conf >= threshold_val) #& (conf > 1e-5)
            valid_mask &= conf_mask 

        # Apply mask
        if apply_mask:
            mask = masks[camera_idx][frame_idx].astype(bool)
            valid_mask &= mask

        return valid_mask
    
    # Add the point_clouds, frames and frustums to the scene, and register frames and frustums to on_click method.
    initialize_scene()

    @gui_timestep.on_update
    def _(_):
        if animation_active["on"]:
            apply_animation_visibility(int(gui_timestep.value))  # アニメーション時のみ単一フレーム制御

    if confs is not None:
        @gui_points_conf.on_update
        def _(_) -> None:
            update_point_clouds()
        @gui_point_color_mode.on_update
        def _(_) -> None:
            mode = gui_point_color_mode.value  # str
            for camera_idx in range(V):
                for frame_idx in range(F):
                    if mode == "Frame":
                        point_clouds[camera_idx][frame_idx].colors = colors[camera_idx][frame_idx]
                    elif mode == "Confidence (raw)":
                        point_clouds[camera_idx][frame_idx].colors = conf2colors_raw[camera_idx][frame_idx]
                    elif mode == "Confidence (equalized)":
                        point_clouds[camera_idx][frame_idx].colors = conf2colors_equalized[camera_idx][frame_idx]
                    else:
                        raise ValueError(f"Unknown point color mode: {mode}")

    if masks is not None:
        @gui_points_mask.on_update
        def _(_) -> None:
            update_point_clouds()
    
    @gui_prune_points.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_change_point_size.on_update
    def _(_) -> None:
        point_size = float(gui_change_point_size.value)
        for camera_idx in range(V):
            for frame_idx in range(F):
                point_clouds[camera_idx][frame_idx].point_size = point_size

    @gui_show_point_clouds.on_update
    def _(_) -> None:
        """Toggle visibility of point clouds."""
        for camera_idx in range(V):
            for frame_idx in range(F):
                if gui_show_point_clouds.value is False:
                    point_clouds[camera_idx][frame_idx].visible = False
                else:
                    point_clouds[camera_idx][frame_idx].visible = frustums[camera_idx][frame_idx].visible or frames[camera_idx][frame_idx].visible 
        if animation_active["on"]:
            apply_animation_visibility(int(gui_timestep.value))
        

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames."""
        for camera_idx in range(V):
            for frame_idx in range(F):
                if gui_show_frames.value is False:
                    frames[camera_idx][frame_idx].visible = False
                else:
                    frames[camera_idx][frame_idx].visible = frustums[camera_idx][frame_idx].visible or point_clouds[camera_idx][frame_idx].visible 
        if animation_active["on"]:
            apply_animation_visibility(int(gui_timestep.value))


    @gui_show_frustums.on_update
    def _(_) -> None:
        """Toggle visibility of camera frustums."""
        for camera_idx in range(V):
            for frame_idx in range(F):
                if gui_show_frustums.value is False:
                    frustums[camera_idx][frame_idx].visible = False
                else:
                    frustums[camera_idx][frame_idx].visible = frames[camera_idx][frame_idx].visible or point_clouds[camera_idx][frame_idx].visible 
        if animation_active["on"]:
            apply_animation_visibility(int(gui_timestep.value))

    if verbose:
        print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                if animation_active["on"] and playing["on"]:
                    gui_timestep.value = (int(gui_timestep.value) + 1) % F
                    time.sleep(1.0 / max(1e-6, float(gui_framerate.value)))  # FPSは無停止で反映
                    continue

                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            if animation_active["on"] and playing["on"]:
                with server.atomic():
                    gui_timestep.value = (int(gui_timestep.value) + 1) % F
                time.sleep(1.0 / max(1e-6, float(gui_framerate.value)))  # FPSは無停止で反映
                continue

            time.sleep(0.02)

    return server
