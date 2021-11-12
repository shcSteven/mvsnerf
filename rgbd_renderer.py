import os,sys, trimesh, json, pyrender, glob
import numpy as np
import random
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.networks import read_pfm
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def save_camera(filename, pose, K):
    file = open(filename, "wb")
    
    file.write('extrinsic\n'.encode('utf-8'))
    posetxt = '%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n0 0 0 1\n' % (pose[0,0],pose[0,1],pose[0,2],pose[0,3],
                                                                  pose[1,0],pose[1,1],pose[1,2],pose[1,3],
                                                                  pose[2,0],pose[2,1],pose[2,2],pose[2,3])
    file.write(posetxt.encode('utf-8'))
    file.write('\n'.encode('utf-8'))
    file.write('intrinsic\n'.encode('utf-8'))
    Ktxt = '%f 0 %f\n0 %f %f\n0 0 1\n'%(K[0,0],K[0,2],K[1,1],K[1,2])
    file.write(Ktxt.encode('utf-8'))

    file.close()

def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

def save_minmax(filename, min, max):
    file = open(filename, "wb")
    file.write("{} {}\n".format(min, max).encode('utf-8'))
    file.close()

def toGLmat(camera_pose):
    yz_flip = np.eye(4, dtype=np.float32)
    #yz_flip[2, 2] = -1 # , yz_flip[2, 2], -1
    camera_pose = yz_flip.dot(camera_pose.T)
    
    return camera_pose.T

def render_depth(scene, camera_pose, intrinsic, light_pose, size=(256, 256)):
    # campose: cam to world
    camera_pose = toGLmat(camera_pose)
    camera = pyrender.IntrinsicsCamera(intrinsic[0,0], intrinsic[1,1], intrinsic[0, 2], 
                                       intrinsic[1, 2], znear=0.05, zfar=100.0)
    scene.add(camera, pose=camera_pose)

    r = pyrender.OffscreenRenderer(size[0], size[1])
    color, depth = r.render(scene)
    r.delete()
    
    camera_node = list(scene.get_nodes(obj=camera))[0]
    
    scene.remove_node(camera_node)
    
    # to opencv format
#     camera_pose[:3,[1,2]] = -camera_pose[:3,[1,2]]
#     camera_pose = np.linalg.inv(camera_pose)
    return color, depth

def mesh_flip_zy(mesh):
    mesh.vertices[:,[1,2]] = mesh.vertices[:,[2,1]] 
    mesh.vertices[:,[2]]   = -mesh.vertices[:,[2]]
    return mesh

def find_nearestCam(position, N=10):
    nearest = []
    for p in position:
        nearest.append(np.argsort(np.linalg.norm(position - p, axis=1))[:N+1])
    return np.stack(nearest, axis=0)

def _lookat(eye, center):
    direction = eye - center
    unit_direction = direction / np.linalg.norm(direction)
    phi = np.arccos(unit_direction[2])
    theta = np.arccos(unit_direction[0] / np.sqrt(1 - unit_direction[2] ** 2))
    T = np.array([
        [np.sin(theta), np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), eye[0]],
        [-np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), eye[1]],
        [0, -np.sin(phi), np.cos(phi), eye[2]],
        [0, 0, 0, 1]
    ])
    return T

from math import sin, cos, pi
def _getSphericalPosition(r, theta, phi, arg=True):
    if arg:
        theta = theta / 180 * pi
        phi = phi / 180 * pi
    x = r * cos(theta) * sin(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(phi)
    return np.array([x,y,z])

def main():
    dataset_dir = '../portraits_data/mesh'
    output_dir = './rendered_data'

    mesh_num = 100
    image_num = 100
    W, H = 256, 256
    N = 10 # neighbor number
    center = np.array([0,0,0])
    scale = 0.2
    radius = 2
    focal = 400 #1111.111
    intrinsic = np.array([[focal,0,W/2],[0,focal,H/2],[0,0,1]])
    #------------------------------------------------------
    
    os.makedirs(f'{output_dir}/cams', exist_ok=True)
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/pairs', exist_ok=True)
    os.makedirs(f'{output_dir}/depths', exist_ok=True)

    for k in range(1, mesh_num + 1):
        data_path = os.path.join(dataset_dir, str(k // 1000),'%05d.obj' % k)
        if not os.path.isfile(data_path): # Some of the orders are missing/deleted
            continue
        mesh = trimesh.load(data_path)
        mesh = mesh_flip_zy(mesh)
        
        # Normalize model points to to [-1,1] with maximun range of xyz
        length = max(mesh.extents)
        center = np.mean(mesh.vertices,axis=0) - np.array([0, 0, 0.1])
        mesh.apply_translation(-center)
        mesh.apply_scale(2 / length)

        Mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        scene.add(Mesh)
        

        poses = []
        for _ in range(image_num):
            pos = _getSphericalPosition(radius, theta=random.random() * 180, phi=60 + random.random() * 70)
            center_noise = np.array([random.uniform(-scale, scale),random.uniform(-scale, scale),random.uniform(-scale, scale)])
            new_center = center + center_noise
            # camera pose in openGL
            T = _lookat(pos, new_center)
            poses.append(T)
        poses = np.stack(poses)

        # add light
    #     for l in range(5):
    #         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.2)
    #         scene.add(light, pose=poses[l])
            
        opengltocv = np.eye(4)
        opengltocv[[1,2],[1,2]] = -1

        # Save poses, images and depths
        for i in tqdm(range(image_num), desc='Progress for Mesh #%05d' % k):
            pose = poses[i]
            color, depth = render_depth(scene=scene, camera_pose=pose, intrinsic=intrinsic, light_pose=poses[0], size=[W,H])
            depth_min, depth_max = np.min([x for x in depth.flatten() if x > 0]), np.max(depth)
            relative_depth = (depth - depth_min) / (depth_max - depth_min)
            pose_cv = np.linalg.inv(pose @ opengltocv)

            # Save camera prameters
            camera_file_path = os.path.join(output_dir, 'cams', 'scan%05d' % k)
            os.makedirs(camera_file_path, exist_ok=True)
            camera_file_name = os.path.join(camera_file_path,'%03d_cam.txt' % i)
            save_camera(camera_file_name, pose_cv, intrinsic)

            # Save relative depth
            depth_file_path = os.path.join(output_dir, 'depths', 'scan%05d' % k)
            os.makedirs(depth_file_path, exist_ok=True)
            depth_file_name = os.path.join(depth_file_path,'depth_map_%03d.pfm' % i)
            save_pfm(depth_file_name, relative_depth)

            # Save minmax depth
            minmax_file_path = os.path.join(output_dir, 'minmaxs', 'scan%05d' % k)
            os.makedirs(minmax_file_path, exist_ok=True)
            minmax_file_name = os.path.join(minmax_file_path,'minmax_map_%03d.txt' % i)
            save_minmax(minmax_file_name, depth_min, depth_max)

            # Save rgba
            color_file_path = os.path.join(output_dir, 'images','scan%05d' % k)
            os.makedirs(color_file_path, exist_ok=True)
            color_file_name = os.path.join(color_file_path,'%03d.png' % i)
            mask = (depth > 0).reshape(H,W,1) * 255
            rgba = np.concatenate([color,mask],axis=2)
            rgba = rgba.astype(np.uint8)
            img = PIL.Image.fromarray(rgba, 'RGBA')
            img.save(color_file_name)
            

        neighbors = find_nearestCam(poses[:,:3,3], N)
        pairs_path = os.path.join(output_dir, 'pairs', 'scan%05d' % k)
        #np.savetxt(f'{output_dir}/pairs/scan{}.txt'.format(k),neighbors,fmt='%d')
        np.savetxt(pairs_path, neighbors, fmt='%d')

if __name__ == '__main__':
    main()
