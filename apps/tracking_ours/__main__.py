import argparse
from pathlib import Path
import pickle
import time
from typing import Dict, List
import os
import sys

from PIL import Image
import cv2 as cv
import kornia
from logzero import logger
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from edam.dataset import hamlyn
from edam.dataset.daVinci import daVinci
from edam.dataset.aesculap import aesculap
from edam.optimization.frame import create_frame
from edam.optimization.pose_estimation import PoseEstimation
from edam.optimization.utils_frame import synthetise_image_and_error
from edam.utils.file import list_files
from edam.utils.image.convertions import (
    numpy_array_to_pilimage,
    pilimage_to_numpy_array,
)
from edam.utils.image.pilimage import (
    pilimage_h_concat,
    pilimage_rgb_to_bgr,
    pilimage_v_concat,
)
from edam.utils.depth import depth_to_color, depth_edge_mask_from_angles
from edam.utils.LineMesh import LineMesh


def parse_args() -> argparse.Namespace:
    """Returns the ArgumentParser of this app.

    Returns:
        argparse.Namespace -- Arguments
    """
    parser = argparse.ArgumentParser(
        description="Shows the images from Hamlyn Dataset"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--input_root_directory",
        type=str,
        help="Root directory where scans are find. E.G. path/to/hamlyn_tracking_test_data",
    )
    group.add_argument(
        "--input_scene_directory",
        type=str,
        help="Alternatively to giving the root, pass a scene directly using this argument."
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device in where to run the optimization E.G. cpu, cuda:0, cuda:1",
    )
    parser.add_argument(
        "-fr",
        "--ratio_frame_keyframe",
        type=int,
        default=2,
        help="Number of frames until a new keyframes.",
    )
    parser.add_argument(
        "-s",
        "--start_scene",
        type=int,
        default=0,
        help="Select the start scene",
    )
    parser.add_argument(
        "-o",
        "--folder_output",
        type=str,
        default="results",
        help="Folder where odometries are saved.",
    )
    parser.add_argument(
        "-st",
        "--scales_tracking",
        type=int,
        default=2,
        help="Number of floors of the pyramid.",
    )
    parser.add_argument(
        "--no_vis_2d",
        action="store_true",
    )
    parser.add_argument(
        "--no_vis_3d",
        action="store_true",
    )
    parser.add_argument(
        "--depth_folder",
        type=str,
        default="depth_anything_v2")

    return parser.parse_args()


def main():
    args = parse_args()
    device_name = args.device
    if args.input_root_directory:
        list_scenes = list_files(args.input_root_directory)
    else:
        list_scenes = [args.input_scene_directory]
    done = False
    scene_number = args.start_scene
    frame_number = 0
    scene_info = None
    no_vis_3d = args.no_vis_3d
    no_vis_2d = args.no_vis_2d
    if not no_vis_3d:
        vis_depth = o3d.visualization.Visualizer()
        vis_3d = o3d.visualization.Visualizer()
        vis_3d.create_window("HAMLYN3D", 1920 // 2, 1080 // 2)
        vis_depth.create_window("Depth", 256, 192, visible=False)
    estimated_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.0075)
    estimated_pose = np.eye(4)
    automode = False

    pe = PoseEstimation()

    keyframe_image = []
    keyframe_depth_image = []
    poses_register = dict()
    poses_register["scene_info"] = []
    poses_register["frame_number"] = []
    poses_register["estimated_pose"] = []
    folder_to_save_results = Path(args.folder_output)
    if not (folder_to_save_results.exists()):
        folder_to_save_results.mkdir(parents=True, exist_ok=True)
    output_path = folder_to_save_results / (time.strftime("%Y%m%d-%H%M%S") + ".pkl")
    scales = args.scales_tracking

    MAX_FRAME = 200
    
    prev_pos = None
    #prev_dir = None

    while not done:
        begin = time.time()

        # -- Load info for the scene if it has not been loaded yet.
        if scene_info is None:
            logger.info("Reseting scene.")
            scene = list_scenes[scene_number]
            #scene_info = hamlyn.load_scene_files(scene)
            color_path = os.path.join( scene, "color" )
            #depth_path = os.path.join( scene, "depth" )
            depth_path = os.path.join( scene, args.depth_folder )
            scene_info = aesculap.load_scene_files( color_path, depth_path )
            frame_number = 0
            points = [np.array([0, 0, 0])]
            if not no_vis_3d:
                vis_3d.clear_geometries()
                vis_depth.clear_geometries()
                vis_3d.add_geometry(estimated_frame)

        # Get current camera
        if not no_vis_3d:
            ctr = vis_3d.get_view_control()
            print(vis_3d.get_view_control(), vis_3d)
            cam = ctr.convert_to_pinhole_camera_parameters()

        # -- Open pose
        estimated_frame.transform(np.linalg.inv(estimated_pose))

        # -- Open data
        (
            rgb_image_registered,
            depth_np,
            k_depth,
            h_d,
            w_d,
        ) = load_aesculap_frame(scene_info, frame_number)

        depth_image = numpy_array_to_pilimage(
            (
                depth_to_color(
                    depth_np,
                    cmap="jet",
                    max_depth=np.max(depth_np),
                    min_depth=np.min(depth_np),
                )
            ).astype(np.uint8)
        )

        gray = cv.cvtColor(
            pilimage_to_numpy_array(rgb_image_registered), cv.COLOR_BGR2GRAY
        )

        new_frame = create_frame(
            c_pose_w=np.linalg.inv(estimated_pose),
            c_pose_w_gt=None,
            gray_image=gray,
            rgbimage=None,
            depth=depth_np,
            k=k_depth.numpy().reshape(3, 3),
            idx=frame_number,
            ref_camera=(frame_number == 0),
            scales=scales,
            code_size=128,
            device_name=device_name,
            uncertainty=None,
        )

        # Keyframe updating
        if frame_number == 0:
            new_frame.modify_pose(c_pose_w=np.linalg.inv(np.eye(4)))
            if not no_vis_3d:
                cam.extrinsic = np.eye(4)


        if frame_number >= MAX_FRAME:
            done = True

        if frame_number % args.ratio_frame_keyframe == 0:
            logger.info("KEYFRAME INSERTED")
            new_frame_ = create_frame(
                c_pose_w=new_frame.c_pose_w,
                c_pose_w_gt=None,
                gray_image=gray,
                rgbimage=None,
                depth=depth_np,
                k=k_depth.numpy().reshape(3, 3),
                idx=frame_number,
                ref_camera=True,
                scales=1,
                code_size=128,
                device_name=device_name,
                uncertainty=None,
            )
            if not (frame_number == 0):
                pe.run(new_frame_, True)

            pe.set_ref_keyframe(new_frame_)
            ref_kf = pe.reference_keyframe
            keyframe_image = rgb_image_registered.copy()
            keyframe_depth_image = depth_image.copy()
            print(depth_np.dtype)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    cv.cvtColor(
                        pilimage_to_numpy_array(keyframe_image), cv.COLOR_BGR2RGB
                    )
                ),
                #o3d.geometry.Image(depth_np * 1000),
                o3d.geometry.Image(depth_np),
                depth_trunc=0.25,
                convert_rgb_to_intensity=False,
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                w_d,
                h_d,
                k_depth[0, 0, 0],  # fx
                k_depth[0, 1, 1],  # fy
                k_depth[0, 0, 2],  # cx
                k_depth[0, 1, 2],  # cy
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic, new_frame.c_pose_w,
            )

            if not no_vis_3d:
                vis_3d.add_geometry(pcd)

        else:
            pe.run(new_frame, True)
            ref_kf = pe.reference_keyframe

        end = time.time()
        time_one_image = round((end - begin) * 1000)
        print("   Computing time {}ms".format(time_one_image))

        print("Estimated pose:", new_frame.c_pose_w)
        # Check for large jumps:
        cur_pos = new_frame.c_pose_w[0:3,3]
        print("\tExtracted pos:", cur_pos)
        if prev_pos is not None:
            dist = np.linalg.norm( cur_pos - prev_pos )
            print(f"\tdist to prev pos: {dist}")
            if dist > 1e-2:
                print("\t", "Large jump detected, ending!")
                break

        prev_pos = cur_pos  # Save for next frame

        (error_np, synthetic_image_np) = synthetise_image_and_error(
            i_pose_w=ref_kf.c_pose_w_tc,
            depth_ref=ref_kf.depth,
            k_ref=ref_kf.k_tc,
            j_pose_w=new_frame.c_pose_w_tc,
            k_target=new_frame.pyr_k_tc[0],
            gray_ref=ref_kf.gray_tc,
            gray_target=new_frame.pyr_gray_tc[0],
        )

        error = numpy_array_to_pilimage(
            (
                depth_to_color(
                    error_np,
                    cmap="jet",
                    max_depth=np.max(error_np),
                    min_depth=np.min(error_np),
                )
            ).astype(np.uint8)
        )

        synthetic_image = numpy_array_to_pilimage(synthetic_image_np)

        frame = pilimage_v_concat(
            [
                pilimage_h_concat(
                    [keyframe_image.resize((round(1920/5), round(h_d*(1920/5)/w_d)), resample=Image.BILINEAR),
                     keyframe_depth_image.resize((round(1920/5), round(h_d*(1920/5)/w_d)), resample=Image.BILINEAR),
                     rgb_image_registered.resize((round(1920/5), round(h_d*(1920/5)/w_d)), resample=Image.BILINEAR),
                     synthetic_image.resize((round(1920/5), round(h_d*(1920/5)/w_d)), resample=Image.BILINEAR),
                     error.resize((round(1920/5), round(h_d*(1920/5)/w_d)), resample=Image.BILINEAR)]
                ),
            ]
        )

        estimated_pose = np.linalg.inv(new_frame.c_pose_w)
        estimated_frame.transform(estimated_pose)
        if not no_vis_3d:
            vis_3d.update_geometry(estimated_frame)

        if frame_number > 0:
            translation = estimated_pose[:, 3]
            new_point = translation[:3]
            points.append(new_point)
            line_mesh = LineMesh(points, radius=0.0005)
            line_mesh_geoms = line_mesh.cylinder_segments
            if not no_vis_3d:
                vis_3d.add_geometry(*line_mesh_geoms)
            del points[0]

        logger.info(
            f"frame {frame_number:d}"
        )
        logger.info(f"")

        poses_register["scene_info"].append(scene_info)
        poses_register["frame_number"].append(frame_number)
        poses_register["estimated_pose"].append(new_frame.c_pose_w.copy())

        a_file = open(str(output_path), "wb")
        pickle.dump(poses_register, a_file)
        a_file.close()

        if not no_vis_3d:
            camera_ = o3d.camera.PinholeCameraParameters()
            camera_.intrinsic = cam.intrinsic
            camera_.extrinsic = cam.extrinsic
            ctr.convert_from_pinhole_camera_parameters(camera_)
        if not no_vis_2d:
            cv.imshow("HAMLYN", pilimage_to_numpy_array(frame))

        if no_vis_2d:
            automode = True

        while True:

            if not no_vis_2d:
                if cv.getWindowProperty("HAMLYN", 0) < 0:
                    break

            if not no_vis_3d:
                vis_3d.poll_events()
                vis_3d.update_renderer()
            if no_vis_2d:
                frame_number = (frame_number + 1) % len(scene_info["list_color_images"])
                if frame_number +1 >= len(scene_info["list_color_images"]):
                    done = True
                break
            else:
                key = cv.waitKey(1) & 0xFF
                if automode:
                    if key == ord("s"):
                        automode = False
                    if frame_number +1 >= len(scene_info["list_color_images"]):
                        done = True
                    frame_number = (frame_number + 1) % len(scene_info["list_color_images"])
                    break
                if key == ord("a"):
                    automode = True
                    logger.info(help_functions()[chr(key)])
                    frame_number = (frame_number + 1) % len(scene_info["list_color_images"])
                    break
                if key == ord("n"):
                    logger.info(help_functions()[chr(key)])
                    frame_number = (frame_number + 1) % len(scene_info["list_color_images"])
                    break
                if key == ord("j"):
                    logger.info(help_functions()[chr(key)])
                    frame_number = (frame_number + 10) % len(
                        scene_info["list_color_images"]
                    )
                    break
                elif key == ord("p"):
                    logger.info(help_functions()[chr(key)])
                    frame_number = frame_number - 1
                    if frame_number < 0:
                        frame_number = len(scene_info["list_color_images"]) - 1
                    break
                elif key == ord(" "):
                    logger.info(help_functions()[chr(key)])
                    scene_number = (scene_number + 1) % len(list_scenes)
                    pe = PoseEstimation()
                    frame_number = 0
                    scene_info = None
                    break
                elif key == ord("\x08"):
                    logger.info(help_functions()[chr(key)])
                    scene_number = scene_number - 1
                    pe = PoseEstimation()
                    if scene_number < 0:
                        scene_number = len(list_scenes) - 1
                    frame_number = 0
                    scene_info = None
                    break
                elif key == ord("h"):
                    logger.info(help_functions()[chr(key)])
                    print_help()
                elif key == ord("q"):
                    logger.info(help_functions()[chr(key)])
                    done = True
                    if not no_vis_3d:
                        vis_3d.destroy_window()
                        vis_depth.destroy_window()
                    break
                elif key != 255:
                    logger.info(f"Unkown command, {chr(key)}")
                    print("Use 'h' to print help and 'q' to quit.")

def load_aesculap_frame(scene_info: Dict[str, List[str]], frame_number: int):
    """Function to load the scene and frame from the Hamlyn dataset,

    Args:
        scene_info ([type]): [description]
        frame_number (int): [description]

    Returns:
        [type]: [description]
    """

    # -- Open the rgb image.
    rgb_path = scene_info["list_color_images"][frame_number]
    rgb_image = pilimage_rgb_to_bgr(Image.open(rgb_path))
    print(type(rgb_image), rgb_image.size, rgb_image.getextrema())

    ## Grey out borders/hide daVinci UI:
    #rgb_image[:, :border_top, :] = -0.458
    #rgb_image[:, border_bottom:, :] = -0.458
    #rgb_image[:, :, :border_left] = -0.458
    #rgb_image[:, :, border_right:] = -0.458
 
    rgb_image_np = pilimage_to_numpy_array(rgb_image)
    print(type(rgb_image_np), rgb_image_np.shape, rgb_image_np.min(), rgb_image_np.max(), rgb_image_np.dtype)
    # Grey out borders/hide daVinci UI:
    rgb_image_np[:aesculap.border_top, :, :] = 0
    rgb_image_np[aesculap.border_bottom:, :, :] = 0
    rgb_image_np[:, :aesculap.border_left, :] = 0
    rgb_image_np[:, aesculap.border_right:, :] = 0

    print(type(rgb_image_np), rgb_image_np.shape, rgb_image_np.min(), rgb_image_np.max(), rgb_image_np.dtype)

    # BGR to RGB
    #rgb_image_np = np.flip( rgb_image_np, 2 )

    rgb_image = numpy_array_to_pilimage( rgb_image_np, mode="RGB" )
    print(type(rgb_image), rgb_image.size, rgb_image.getextrema())

    # -- Open the depth image.
    depth_path = scene_info["list_depth_images"][frame_number]
    #depth_np = cv.imread(depth_path, cv.IMREAD_ANYDEPTH).astype(np.float32) / 1000
    with open( depth_path, 'rb' ) as f:
        depth_np = np.load( f )
    depth_np = depth_np.astype( np.float32 )
    depth_np = depth_np / 1000
    print("Depth range (m):", depth_np.min(), depth_np.max(), depth_np.dtype )

    # Avoid zeros:
    depth_np[depth_np < 2e-2] = 9999
    #depth_np[depth_np < 3e-2] = 9999
    #depth_np[depth_np > 8e-2] = 9999  # transfer-depth becomes unreliable at around 10 cm?
    #

    #edge_mask = depth_edge_mask_from_angles( depth_np, scene_info["camera_parameters"] )
    #depth_np[edge_mask] = 9999

    # -- Get the camera matrices.
    k_depth: np.ndarray = scene_info["intrinsics"][:3, :3]  # type: ignore

    # -- Transform into tensors.
    k_depth = kornia.utils.image_to_tensor(k_depth, keepdim=False).squeeze(1)  # Bx3x3

    h_d, w_d = depth_np.shape

    print("depth shape:", h_d, w_d)

    return (
        rgb_image,
        depth_np,
        k_depth,
        h_d,
        w_d,
    )



def load_davinci_frame(scene_info: Dict[str, List[str]], frame_number: int):
    """Function to load the scene and frame from the Hamlyn dataset,

    Args:
        scene_info ([type]): [description]
        frame_number (int): [description]

    Returns:
        [type]: [description]
    """

    # -- Open the rgb image.
    rgb_path = scene_info["list_color_images"][frame_number]
    rgb_image = pilimage_rgb_to_bgr(Image.open(rgb_path))
    print(type(rgb_image), rgb_image.size, rgb_image.getextrema())

    ## Grey out borders/hide daVinci UI:
    #rgb_image[:, :border_top, :] = -0.458
    #rgb_image[:, border_bottom:, :] = -0.458
    #rgb_image[:, :, :border_left] = -0.458
    #rgb_image[:, :, border_right:] = -0.458
 
    rgb_image_np = pilimage_to_numpy_array(rgb_image)
    print(type(rgb_image_np), rgb_image_np.shape, rgb_image_np.min(), rgb_image_np.max(), rgb_image_np.dtype)
    # Grey out borders/hide daVinci UI:
    rgb_image_np[:daVinci.border_top, :, :] = 0
    rgb_image_np[daVinci.border_bottom:, :, :] = 0
    rgb_image_np[:, :daVinci.border_left, :] = 0
    rgb_image_np[:, daVinci.border_right:, :] = 0

    print(type(rgb_image_np), rgb_image_np.shape, rgb_image_np.min(), rgb_image_np.max(), rgb_image_np.dtype)

    # BGR to RGB
    #rgb_image_np = np.flip( rgb_image_np, 2 )

    rgb_image = numpy_array_to_pilimage( rgb_image_np, mode="RGB" )
    print(type(rgb_image), rgb_image.size, rgb_image.getextrema())

    # -- Open the depth image.
    depth_path = scene_info["list_depth_images"][frame_number]
    with open( depth_path, 'rb' ) as f:
        depth_np = np.load( f )
    depth_np = depth_np / 1000
    depth_np = depth_np.astype( np.float32 )
    #depth_np = cv.imread(depth_path, cv.IMREAD_ANYDEPTH).astype(np.float32) / 1000
    print("Depth range (m):", depth_np.min(), depth_np.max())

    # -- Get the camera matrices.
    k_depth: np.ndarray = scene_info["intrinsics"][:3, :3]  # type: ignore

    # -- Transform into tensors.
    k_depth = kornia.utils.image_to_tensor(k_depth, keepdim=False).squeeze(1)  # Bx3x3

    h_d, w_d = depth_np.shape

    return (
        rgb_image,
        depth_np,
        k_depth,
        h_d,
        w_d,
    )


def help_functions():
    helper = {}
    helper["a"] = "Autoplay"
    helper["s"] = "Stop Autoplay"
    helper["n"] = "Next Image"
    helper["j"] = "Next Image x10"
    helper["p"] = "Previous Image"
    helper["\x08"] = "(Backslash) Previous Scene"
    helper[" "] = "(Space bar) Next Scene"
    helper["h"] = "Print help"
    helper["q"] = "Quit program"
    return helper


def print_help():
    logger.info("Printing help:")
    for k, v in help_functions().items():
        print(k, ":\t", v)


if __name__ == "__main__":
    main()
