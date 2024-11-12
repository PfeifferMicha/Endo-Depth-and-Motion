import os
import yaml
import numpy as np

## Cut off daVinci UI:
#border_top = 50
#border_bottom = 480
#border_left = 31
#border_right = 607
# Cut off UI and borders:
border_top = 6
border_bottom = 540
border_left = 108
border_right = 960

def load_scene_files( color_path, depth_path ):

    list_color_images = []
    list_depth_images = []
    intrinsics = None
    filenames = [filename for filename in os.listdir( color_path )]

    for filename in sorted( filenames ):
        if "left" in filename and "png" in filename:
            print("filename", filename)
            left_filename = os.path.join( color_path, filename )
            #depth_filename = os.path.join( depth_path, filename.replace("left", "depth") )
            depth_filename = os.path.join( depth_path, filename.replace("png", "npy") )
            left_config_filename = left_filename.replace( "png", "yml" )
            # Also try without replacing depth:
            #if not os.path.exists( depth_filename ):
            #    depth_filename = os.path.join( depth_path, filename )
            print("depth", depth_filename, os.path.exists( depth_filename ))
            print("config", left_config_filename, os.path.exists( left_config_filename ))
            if os.path.exists( depth_filename ) and os.path.exists( left_config_filename ):
                print(left_filename, depth_filename)
                list_color_images.append( left_filename )
                list_depth_images.append( depth_filename )
                if intrinsics is None:
                    with open( left_config_filename ) as stream:
                        left_config = yaml.safe_load(stream)
                        intrinsic_params = left_config["projection_matrix"]["data"]
                        intrinsics = np.array( intrinsic_params ).reshape( (3,4) )

    print("intrinsics", intrinsics)

    # Also store intrinsics as dictionary:
    camera_parameters = {
            "fx": intrinsics[0,0],
            "fy": intrinsics[1,1],
            "cx": intrinsics[0,2],
            "cy": intrinsics[1,2],
            }
    print("camera_parameters", camera_parameters)

    return {
            "list_color_images": list_color_images,
            "list_depth_images": list_depth_images,
            "intrinsics": intrinsics,
            "camera_parameters": camera_parameters,
            }
    
     

