import cv2 as cv
import open3d as o3d
import pickle5 as pickle
import numpy as np
import kornia
from pathlib import Path
import argparse
import scipy
import time
import os
import sys
import yaml
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
import pytransform3d.camera as pyt_camera

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from edam.utils.parser import txt_to_nparray
from edam.utils.LineMesh import LineMesh
#from edam.utils.depth import depth_edge_mask_from_angles


def parse_args() -> argparse.Namespace:
    """Returns the ArgumentParser of this app.

    Returns:
        argparse.Namespace -- Arguments
    """
    parser = argparse.ArgumentParser(
        description="Shows the images from Hamlyn Dataset"
    )
    parser.add_argument(
        "-o",
        "--output_root_directory",
        type=str,
        required=True,
        help="Root directory where mesh is saved. E.g. path/to/test1",
    )
    parser.add_argument(
        "-i",
        "--input_odometry_file",
        type=str,
        required=True,
        help="Input odometry file. E.g. apps/tracking_ours/results/test1.pkl",
    )
    parser.add_argument(
        "--depth_folder",
        type=str,
        default="depth_anything_v2")
    parser.add_argument(
        "--use_poses_from_oneslam",
        action="store_true" )
    parser.add_argument(
        "--use_poses_from_endovo",
        action="store_true" )

    return parser.parse_args()

class Loader:

    def __init__( self, root_path, depth_folder ):

        list_color_images = {}
        list_depth_images = {}
        intrinsics = None
        camera_parameters = None
        color_path = os.path.join( root_path, "color" )
        depth_path = os.path.join( root_path, depth_folder )
        filenames = [filename for filename in os.listdir( color_path )]
        
        self.valid_frame_numbers = []
   
        # We count the number of "left*.png" files we encounter and use that as index for each frame.
        # This is because the tested tracking algorithms also use each left frame and thus their frame ids
        # correspond to those frames.
        # However, because we filter out some bad frame names, we only take a subset of these frames and store
        # those "valid" frame ids in self.valid_frame_numbers. The ids in valid_frame_numbers can then be
        # used to retrieve the corresponding frame via get_frame() and those ids will then correspond to the
        # correct poses in the trajectory.
        filenames = [f for f in filenames if ("left" in f and ".png" in f)]
        for i, filename in enumerate( sorted( filenames ) ):
            if "left" in filename and "png" in filename:
                print("filename", filename)
                left_filename = os.path.join( color_path, filename )
                #depth_filename = os.path.join( depth_path, filename.replace("left", "depth") )
                depth_filename = os.path.join( depth_path, filename.replace("png", "npy") )

                # Also try other format:
                #if not os.path.exists( depth_filename ):
                #    depth_filename = os.path.join( depth_path, filename )
                left_config_filename = left_filename.replace( "png", "yml" )
                print("depth", depth_filename, os.path.exists( depth_filename ))
                print("config", left_config_filename, os.path.exists( left_config_filename ))
                if os.path.exists( depth_filename ) and os.path.exists( left_config_filename ):
                    print(left_filename, depth_filename)
                    list_color_images[i] = left_filename
                    list_depth_images[i] = depth_filename
                    self.valid_frame_numbers.append( i )
                    if intrinsics is None:
                        with open( left_config_filename ) as stream:
                            left_config = yaml.safe_load(stream)
                            intrinsic_params = left_config["projection_matrix"]["data"]
                            intrinsics = np.array( intrinsic_params ).reshape( (3,4) )
                            camera_parameters = {
                                        "fx": intrinsics[0,0],
                                        "fy": intrinsics[1,1],
                                        "cx": intrinsics[0,2],
                                        "cy": intrinsics[1,2],
                                        }
                            print("intrinsics", intrinsics)


        print("intrinsics", intrinsics)
        
        #depth = cv.imread( list_depth_images[0], cv.IMREAD_ANYDEPTH)\
        #            .astype(np.float32) / 1000  # meters

        with open( list(list_depth_images.values())[0], 'rb' ) as f:
            depth = np.load( f )
        depth = depth.astype( np.float32 )
        depth = depth / 1000

        # Avoid zeros:
        #depth[depth < 1e-3] = 9999
        depth[depth < 2e-2] = 9999
        #depth[depth < 1e-2] = 9999
        #depth[depth > 8e-2] = 9999  # transfer-depth becomes unreliable at around 10 cm?

        h, w = depth.shape

        print("depth", np.min(depth), np.max(depth), h, w )
        print("depth w/o outliers", np.min(depth[depth < 9990]), np.max(depth[depth < 9990]), h, w )

        #edge_mask = depth_edge_mask_from_angles( depth, camera_parameters )
        #depth[edge_mask] = 9999

        self.img_size = (h,w)

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            w,
            h,
            intrinsics[0, 0],  # fx
            intrinsics[1, 1],  # fy
            intrinsics[0, 2],  # cx
            intrinsics[1, 2],  # cy
        )
        self.intrinsics_matrix = intrinsics
        self.list_color_images = list_color_images
        self.list_depth_images = list_depth_images

    def get_frame( self, frame_number ):

        assert frame_number in self.valid_frame_numbers, f"{frame_number} not in {self.valid_frame_numbers}"

        color = o3d.io.read_image( self.list_color_images[frame_number] )
        #depth = cv.imread( self.list_depth_images[frame_number], cv.IMREAD_ANYDEPTH)\
        #            .astype(np.float32) / 1000  # meters

        with open( self.list_depth_images[frame_number], 'rb' ) as f:
            depth = np.load( f )
        depth = depth.astype( np.float32 )
        depth = depth / 1000

        depth[depth < 2e-2] = 9999
        #depth[depth < 1e-2] = 9999
        #depth[depth > 8e-2] = 9999  # transfer-depth becomes unreliable at around 10 cm?
        #depth = depth / 1000 * 20

        print("depth", np.min(depth), np.max(depth), depth.shape)
        print("depth w/o outliers", np.min(depth[depth < 9990]), np.max(depth[depth < 9990]), depth.shape)

        return color, depth, self.intrinsics, self.list_color_images[frame_number]



def main():
    args = parse_args()
    root = args.output_root_directory
    odometry_file = args.input_odometry_file

    name_of_folder_to_save_results = os.path.splitext(os.path.basename(os.path.normpath(odometry_file)))[0]
    parent_folder_to_save_results = Path(os.path.join(root, "map"))
    if not (parent_folder_to_save_results.exists()):
        parent_folder_to_save_results.mkdir(parents=True, exist_ok=True)
    folder_to_save_results = Path(os.path.join(parent_folder_to_save_results, name_of_folder_to_save_results))
    if not (folder_to_save_results.exists()):
        folder_to_save_results.mkdir(parents=True, exist_ok=True)

    print("Reading poses from:", odometry_file)

    if args.use_poses_from_endovo:

        with open(odometry_file, 'rb') as f:
            lines = f.readlines()
            poses = []
            poses_frame_numbers = []
            for l in lines:
                frame, x, y, z, qx, qy, qz, qw = l.split()
                poses_frame_numbers.append( int(frame) )
                rot = scipy.spatial.transform.Rotation.from_quat(
                        (float(qx), float(qy), float(qz), float(qw)) )
                pose = np.eye( 4 )
                pose[:3,:3] = rot.as_matrix()
                pose[:3,3] = (float(x),float(y),float(z))
                poses.append( np.linalg.inv(pose) )

    else:
        with open(odometry_file, 'rb') as f:
            poses_register = pickle.load(f)
        if args.use_poses_from_oneslam:
            # Oneslam only saves one dictionary of poses:
            #frame_numbers = poses_register.keys()
            tmp_poses = poses_register.values()
            poses = [p[0] for p in tmp_poses] # remove intrinsics values that are stored with each pose

            path, filename = os.path.split( odometry_file )
            keyframes_file = os.path.join( path, "keyframes.pickle" )
            with open(keyframes_file, 'rb') as f:
                poses_frame_numbers = pickle.load(f)
            print(poses_frame_numbers)

        else:
            # EndoDepthAndMotion by default uses a more nested structure, so extract the poses from that:
            poses = poses_register["estimated_pose"]
            poses_frame_numbers = poses_register["frame_number"]

    # Create poses.log
    with open(os.path.join(folder_to_save_results, 'poses.log'), 'w') as traj:
        print("WRITING:", os.path.join(folder_to_save_results, 'poses.log') )
        for fn in poses_frame_numbers:
        #for i, pose in enumerate(poses):
            pose = poses[fn]
            print(pose)
            traj.write(f"0 0 {fn}\n"
                       f"{pose[0, 0]} {pose[0, 1]} {pose[0, 2]} {pose[0, 3]}\n"
                       f"{pose[1, 0]} {pose[1, 1]} {pose[1, 2]} {pose[1, 3]}\n"
                       f"{pose[2, 0]} {pose[2, 1]} {pose[2, 2]} {pose[2, 3]}\n"
                       f"{pose[3, 0]} {pose[3, 1]} {pose[3, 2]} {pose[3, 3]}\n"
                       )
    trajectory = o3d.io.read_pinhole_camera_trajectory(os.path.join(folder_to_save_results, 'poses.log'))
    os.remove(os.path.join(folder_to_save_results, 'poses.log'))

    loader = Loader( root, args.depth_folder )
    intrinsics = loader.intrinsics
    for i in range(len(trajectory.parameters)):
        print("Changing intrinsics of the {:d}-th image.".format(i))
        trajectory.parameters[i].intrinsic = intrinsics
    o3d.io.write_pinhole_camera_trajectory(os.path.join(folder_to_save_results, 'trajectory.log'), trajectory)

    print("poses:", poses_frame_numbers)
    print("valid image frames:", loader.valid_frame_numbers)

    # Adjust the frame numbers to only include those frames which were also found to be valid by the
    # image loader:
    frame_numbers = [fn for fn in poses_frame_numbers if fn in loader.valid_frame_numbers]

    print("frame_numbers:", frame_numbers)

    # DEBUG: REMOVE THIS!!
    #if args.use_poses_from_endovo:
    #    frame_numbers = frame_numbers[0:int(len(frame_numbers)*0.5)]

    # DEBUG:
    #frame_numbers = [frame_numbers[-1]]

    # DEBUG WRITE TRAJECTORY:
    # Store whole trajectory:
    with open(os.path.join(folder_to_save_results, 'all_frame_poses.csv'), 'w') as f:
        f.write( f"x y z nx ny nz\n" )
        for i in range(len(trajectory.parameters)):
            ex = trajectory.parameters[i].extrinsic
            x, y, z = ex[0,3], ex[1,3], ex[2,3]
            rot = ex[0:3,0:3]
            n = rot.dot( np.array([0,0,1]) )  # Rotate the unit vector in z direction (forward) by cam rotation
            f.write( f"{x} {y} {z} {n[0]} {n[1]} {n[2]}\n" )
    # Stores only those poses of the frames that are valid (have depth) and are used later on:
    with open(os.path.join(folder_to_save_results, 'used_frame_poses.csv'), 'w') as f:
        f.write( f"x y z nx ny nz\n" )
        for fn in frame_numbers:
            ex = trajectory.parameters[fn].extrinsic
            x, y, z = ex[0,3], ex[1,3], ex[2,3]
            rot = ex[0:3,0:3]
            n = rot.dot( np.array([0,0,1]) )  # Rotate the unit vector in z direction (forward) by cam rotation
            f.write( f"{x} {y} {z} {n[0]} {n[1]} {n[2]}\n" )

    #fig = pv.figure()
    #fig = plt.figure()
    ##fig.plot_mesh(mesh_filename)
    #for pose in poses:
    #    #fig.plot_transform(A2B=pose, s=0.1)
    #    pyt_camera.plot_camera(M=loader.intrinsics_matrix, cam2world=pose, virtual_image_distance=0.1, sensor_size=loader.img_size)
    #
    ##fig.save_image( "trajectory.png" )
    #fig.show()

    #exit()

    rgbd_images = compute_rgbd_images(loader, frame_numbers)

    print( "len(rgbd_images)", len(rgbd_images) )
    print( "len(poses)", len(poses) )
    print( "len(frame_numbers)", len(frame_numbers) )
    print( "max(frame_numbers)", max(frame_numbers) )
    print( "len(trajectory.parameters)", len(trajectory.parameters) )
    print( "len(loader.valid_frame_numbers)", len(loader.valid_frame_numbers) )
    print( "max(loader.valid_frame_numbers)", max(loader.valid_frame_numbers) )

    max_frame_number = min( len(rgbd_images), len(poses) ) - 1

    export_pointclouds = True
    if export_pointclouds:
        compute_point_clouds(loader, frame_numbers, rgbd_images, folder_to_save_results)

    # Volumetric fusion with TSDF
    begin = time.time()
    mesh = tsdf_optimize(loader, frame_numbers, rgbd_images, trajectory)
    end = time.time()
    time_tsdf = round((end - begin) * 1000)
    print("   Computing time of volumetric fusion {}ms".format(time_tsdf))
    mesh_filename = os.path.join(folder_to_save_results, 'mesh.ply')
    o3d.io.write_triangle_mesh( mesh_filename,
                               mesh,
                               write_ascii=True)
    print("Wrote:", mesh_filename)

    # Draw trajectory
    points = []
    for i, pose in enumerate(poses):
        pose = np.linalg.inv(pose)
        position = pose[:, 3]
        points.append(position[:3])
        print(i, position)
    line_mesh = LineMesh(points, radius=0.0005)
    line_mesh_geoms = line_mesh.cylinder_segments

    o3d.visualization.draw_geometries(line_mesh_geoms + [mesh])

    print("\n-> Done!")


# Cut off UI and borders:
border_top = 6
border_bottom = 540
border_left = 108
border_right = 960

def compute_rgbd_images(loader, frame_numbers):
    rgbd_images = {}
    for fn in frame_numbers:
        print("Store {:d}-th image into rgbd_images.".format(fn))
        color, depth, _, _ = loader.get_frame(fn)

        print("depth", depth.shape, depth.min(), depth.max())

        depth[:border_top, :] = 9999
        depth[border_bottom:, :] = 9999
        depth[:, :border_left] = 9999
        depth[:, border_right:] = 9999

        print("depth", depth.shape, depth.min(), depth.max())

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            o3d.geometry.Image(depth),
            depth_trunc=0.3,
            depth_scale=1.0,
            convert_rgb_to_intensity=False
        )
        rgbd_images[fn] = rgbd

    return rgbd_images

def compute_point_clouds(loader, frame_numbers, rgbd_images, output_path):
    
    for fn in frame_numbers:
        #if i % 5 != 0:
        #    continue
        print("Point cloud for {:d}-th image".format(fn))
        color, depth, intrinsics, color_filename = loader.get_frame(fn)
        rgbd_image = rgbd_images[fn]
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics )
        print(pc)

        #pc = o3d.geometry.voxel_down_sample( pc, 0.002 )
        pc = pc.voxel_down_sample( 0.002 )

        print(pc)

        pc_filename = color_filename.replace( "left", "pc" )
        pc_filename = pc_filename.replace( ".png", ".ply" )

        basename = os.path.basename( pc_filename )

        out_path = os.path.join( output_path, basename )
        print("\tWriting:", out_path)

        o3d.io.write_point_cloud(out_path, pc)



def tsdf_optimize(loader, frame_numbers, rgbd_images, poses):
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.001,
        sdf_trunc=0.005,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for fn in frame_numbers:
        print("Integrate {:d}-th image into the volume.".format( fn ) )
        _, _, intrinsics, _ = loader.get_frame( fn )
        tsdf.integrate(rgbd_images[fn], intrinsics, np.linalg.inv(poses.parameters[fn].extrinsic))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = tsdf.extract_triangle_mesh()

    return mesh


#def load_frame(root, frame_number: int):
#    color = o3d.io.read_image(os.path.join(root, "color", "{:010d}.jpg".format(frame_number)))
#    depth = cv.imread(os.path.join(root, "depth", "{:010d}.png".format(frame_number)), cv.IMREAD_ANYDEPTH)\
#                .astype(np.float32) / 1000  # meters
#    intrinsics = txt_to_nparray(os.path.join(root, "intrinsics.txt"))
#    k: np.ndarray = intrinsics[:3, :3]  # type: ignore
#
#    # -- Transform into tensors.
#    k = kornia.utils.image_to_tensor(k, keepdim=False).squeeze(1)  # Bx3x3
#
#    h, w = depth.shape
#
#    intrinsic = o3d.camera.PinholeCameraIntrinsic(
#        w,
#        h,
#        k[0, 0, 0],  # fx
#        k[0, 1, 1],  # fy
#        k[0, 0, 2],  # cx
#        k[0, 1, 2],  # cy
#    )
#
#    return color, depth, intrinsic


if __name__ == "__main__":
    main()
