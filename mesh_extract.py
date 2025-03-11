import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import math
import numpy as np
from scene.cameras import Camera
from gaussian_renderer import render
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json

from PIL import Image
from utils.vis_utils import apply_depth_colormap

def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')



def extract_mesh(dataset, pipe, checkpoint_iterations=None):
    gaussians = GaussianModel(dataset.sh_degree)
    output_path = os.path.join(dataset.model_path,"point_cloud")
    iteration = 0
    if checkpoint_iterations is None:
        for folder_name in os.listdir(output_path):
            iteration= max(iteration,int(folder_name.split('_')[1]))
    else:
        iteration = checkpoint_iterations
    output_path = os.path.join(output_path,"iteration_"+str(iteration),"point_cloud.ply")
    assert os.path.exists(output_path), "{} does not exist".format(output_path)


    gaussians.load_ply(output_path)
    print(f'Loaded gaussians from {output_path}')
    
    kernel_size = dataset.kernel_size
    
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = load_camera(dataset)

    print(f'Loaded cameras from colmap')

    render_image_folder = os.path.join(dataset.model_path, "train", "ours_{}".format(iteration), "renders")
    os.makedirs(render_image_folder, exist_ok=True)

    depth_list = []
    color_list = []
    alpha_thres = 0.5
    for idx, viewpoint_cam in enumerate(viewpoint_cam_list):
        # Rendering offscreen from that camera 
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1,2,0)
        color_list.append(np.ascontiguousarray(rendered_img))
        depth = render_pkg["median_depth"].clone()
        if viewpoint_cam.gt_mask is not None:
            depth[(viewpoint_cam.gt_mask < 0.5)] = 0
        depth[render_pkg["mask"]<alpha_thres] = 0
        depth_list.append(depth[0].cpu().numpy())

        rendered_depth = render_pkg["depth"]
        rendered_mask = render_pkg["mask"]
        plt.imshow(apply_depth_colormap(rendered_depth[0, ..., None], rendered_mask[0, ..., None]).detach().cpu().numpy())

        # 保存图片
        save_img_u8(rendered_img.transpose((1,2,0)), os.path.join(render_image_folder, '{0:05d}'.format(idx) + ".png"))

    print("rendered image and depth from 3DGS")

    torch.cuda.empty_cache()
    voxel_size = 0.02   # 原值为0.002，当场景太大，可尝试增大该值
    o3d_device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'),
                                            attr_dtypes=(o3c.float32,
                                                         o3c.float32,
                                                         o3c.float32),
                                            attr_channels=((1), (1), (3)),
                                            voxel_size=voxel_size,
                                            block_resolution=16,
                                            block_count=50000,
                                            device=o3d_device)
    for color, depth, viewpoint_cam in zip(color_list, depth_list, viewpoint_cam_list):
        depth = o3d.t.geometry.Image(depth)
        depth = depth.to(o3d_device)
        color = o3d.t.geometry.Image(color)
        color = color.to(o3d_device)
        W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
        fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
        fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
        intrinsic = np.array([[fx,0,float(W)/2],[0,fy,float(H)/2],[0,0,1]],dtype=np.float64)
        intrinsic = o3d.core.Tensor(intrinsic)
        extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        frustum_block_coords = vbg.compute_unique_block_coordinates(
                                                                        depth, 
                                                                        intrinsic,
                                                                        extrinsic, 
                                                                        1.0, 8.0
                                                                    )
        vbg.integrate(
                        frustum_block_coords, 
                        depth, 
                        color,
                        intrinsic,
                        extrinsic,  
                        1.0, 8.0
                    )

    mesh = vbg.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon.ply"),mesh.to_legacy())
    print("done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
        
        
    
    