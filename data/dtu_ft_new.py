
from torch.utils.data import Dataset
from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *

class DTU_ft_new(Dataset):
    def __init__(self, args, split='train', load_ref=False):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.args = args
        self.root_dir = os.path.dirname(args.datadir)
        self.scan = os.path.basename(args.datadir)
        self.split = split

        downsample = args.imgScale_train if split=='train' else args.imgScale_test
        #assert int(640*downsample)%32 == 0, \
            #f'image width is {int(640*downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        #self.img_wh = (int(640*downsample),int(512*downsample))
        self.img_wh = (int(256*downsample),int(256*downsample))
        self.downsample = downsample
        print(f'==> image down scale: {self.downsample}')

        self.scale_factor = 1.0 / 200
        self.define_transforms()

        #self.pair_idx = torch.load('configs/pairs.th')
        #print("pair_idx; ", self.pair_idx)
        #self.pair_idx = [self.pair_idx['dtu_train'],self.pair_idx['dtu_test']]
        idx1 = []
        idx2 = []
        #for i in range(32):
            #idx1.append(i)
        #for i in range(40,48):
            #idx2.append(i)
        #self.pair_idx = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [16,17,18,19]]
        #self.pair_idx = [idx1, idx2]
        self.pair_idx = [[0,49,24,76,1,65,69,89,2,24,49,40,3,47,79,60,4,22,57,9,5,28,94,41,6,40,41,45,7,26,51,27],
                         [8,87,59,93,9,4,42,57]]
        self.bbox_3d = torch.tensor([[-1.0, -1.0, 2.2], [1.0, 1.0, 4.2]])
        self.near_far = [2.125, 4.525]
        

        if not load_ref:
            self.read_meta()

    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_cam_file(self, filename, maxminname):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        #print("ext: ", extrinsics)
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        #print("int: ", intrinsics)
        # depth_min & depth_interval: line 11
        
        with open(maxminname) as ff:
            lines = [line.rstrip() for line in ff.readlines()]
            
        #depth_min = float(lines[11].split()[0]) * self.scale_factor
        #depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        #self.depth_interval = float(lines[11].split()[1])
        
        depth_min = float(lines[0].split()[0]) * self.scale_factor
        depth_max = float(lines[0].split()[1]) * self.scale_factor
        self.depth_interval = depth_max - depth_min
        

        # scaling
        extrinsics[:3, 3] *= self.scale_factor
        intrinsics[0:2] *= self.downsample

        return intrinsics, extrinsics, [depth_min, depth_max]

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (256, 256)
        #print("the size of depth_h is: ", depth_h.size)
        """
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        """
        depth_h = cv2.resize(depth_h, None, fx=self.downsample, fy=self.downsample,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        
        return depth_h

    def read_source_views(self, pair_idx=None, device=torch.device("cpu")):

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # if do not specify source views, load index from pairing file
        #if pair_idx is None:
            #pair_idx = self.pair_idx[0][:3]
            #print(f'====> ref idx: {pair_idx}')
            
        self.img_idx = self.pair_idx[0] if 'train'==self.split else self.pair_idx[1]
        #print(f'===> {self.split}ing index: {self.img_idx}')

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        near_far_source = []
        for i,idx in enumerate(pair_idx):
            proj_mat_filename = os.path.join(self.root_dir, f'rendered_temp/cams/scan00001/{idx:03d}_cam.txt')
            #print(i)
            maxmin_filename = os.path.join(self.root_dir, f'rendered_temp/minmaxs/scan00001/minmax_map_{idx:03d}.txt')
            
            intrinsic, w2c, near_far_source = self.read_cam_file(proj_mat_filename, maxmin_filename)
            c2w = np.linalg.inv(w2c)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]
            intrinsic[:2] = intrinsic[:2]*4  # 4 times downscale in the feature space
            intrinsics.append(intrinsic.copy())

            image_path = os.path.join(self.root_dir, f'rendered_temp/images/scan00001/{idx:03d}.png')
            print("image_path: ", image_path)
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            img = img[0:3, :, :] #(3, h, w)
            imgs.append(src_transform(img))

        pose_source = {}
        #pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        #pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        #pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        print("imgs: ", imgs)
        print("proj_mats: ", proj_mats)
        #imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        #proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, near_far_source, pose_source

    def load_poses_all(self):
        c2ws = []
        List = sorted(os.listdir(os.path.join(self.root_dir, f'Cameras/train/')))
        for item in List:
            proj_mat_filename = os.path.join(self.root_dir, f'Cameras/train/{item}')
            intrinsic, w2c, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            c2ws.append(np.linalg.inv(w2c))
        self.focal = [intrinsic[0, 0], intrinsic[1, 1]]
        return np.stack(c2ws)

    def read_meta(self):

        # sub select training views from pairing file
        """
        if os.path.exists('configs/pairs.th'):
            self.img_idx = self.pair_idx[0] if 'train'==self.split else self.pair_idx[1]
            print(f'===> {self.split}ing index: {self.img_idx}')
        """
        
        self.img_idx = self.pair_idx[0] if 'train'==self.split else self.pair_idx[1]
        print(f'===> {self.split}ing index: {self.img_idx}')

        # name = os.path.basename(self.root_dir)
        # test_idx = torch.load('configs/pairs.th')[f'{name}_test']
        # self.img_idx = test_idx if self.split!='train' else np.delete(np.arange(0,49),test_idx)

        w, h = self.img_wh

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_depth = []
        self.w2cs = []
        self.c2ws = []
        self.intrinsics = []
        self.near_fars = []
        
        
        imgs = [] 
        proj_mats = []
        near_far_source = []
        pose_source = []
        imgs, proj_mats, near_far_source, pose_source = self.read_source_views(self)
        self.imgs = imgs
        self.proj_mats = proj_mats
        
        for idx in self.img_idx:
            #proj_mat_filename = os.path.join(self.root_dir, f'Cameras/train/{idx:08d}_cam.txt')
            #intrinsic, w2c, near_far = self.read_cam_file(proj_mat_filename)
            proj_mat_filename = os.path.join(self.root_dir, f'rendered_temp/cams/scan00001/{idx:03d}_cam.txt')
            
            maxmin_filename = os.path.join(self.root_dir, f'rendered_temp/minmaxs/scan00001/minmax_map_{idx:03d}.txt')
            
            intrinsic, w2c, near_far = self.read_cam_file(proj_mat_filename, maxmin_filename)
            c2w = np.linalg.inv(w2c)
            self.poses += [c2w]
            c2w = torch.FloatTensor(c2w)

            #image_path = os.path.join(self.root_dir,
            #                            f'Rectified/{self.scan}_train/rect_{idx + 1:03d}_3_r5000.png')
            #depth_filename = os.path.join(self.root_dir,
            #                              f'Depths/{self.scan}/depth_map_{idx:04d}.pfm')
            image_path = os.path.join(self.root_dir, f'rendered_temp/images/scan00001/{idx:03d}.png')
            depth_filename = os.path.join(self.root_dir, f'rendered_temp/depths/scan00001/depth_map_{idx:03d}.pfm')
            
            
            #depth = self.read_depth(depth_filename)
            
            
            self.image_paths += [image_path]
            img = Image.open(image_path)
            #width, height = img.size
            #print(width, height)
            #print(self.img_wh)
            #print(Image.LANCZOS)
            img = img.resize(self.img_wh, Image.LANCZOS)
            #width, height = img.size
            #print(width, height)
            #print("pixel value is: ", img.getpixel((99,100)))
            img = self.transform(img)  # (4, h, w)?
            img = img[0:3, :, :] #(3, h, w)
            #print(img.shape)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
            self.all_rgbs += [img]

            if os.path.exists(depth_filename) and self.split!='train':
                depth = self.read_depth(depth_filename)
                depth *= self.scale_factor
                self.all_depth += [torch.from_numpy(depth).float().view(-1,1)]

            # ray directions for all pixels, same for all images (same H, W, focal)
            intrinsic[:2] *= 4
            center = [intrinsic[0,2], intrinsic[1,2]]
            self.focal = [intrinsic[0,0], intrinsic[1,1]]
            self.directions = get_ray_directions(h, w, self.focal, center)  # (h, w, 3)
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         near_far[0] * torch.ones_like(rays_o[:, :1]),
                                         near_far[1] * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)
            self.w2cs.append(torch.FloatTensor(w2c))
            self.c2ws.append(torch.FloatTensor(c2w))
            self.intrinsics.append(torch.FloatTensor(intrinsic))
            self.near_fars.append(torch.FloatTensor(near_far))
            

        self.poses = np.stack(self.poses)
        if 'train' == self.split:
            #print(self.all_rays)
            #print(self.all_rays.shape)
            #self.w2cs = torch.tensor(np.array(self.w2cs))
            #print(self.w2cs)
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_depth = torch.stack(self.all_depth, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
            #print(self.all_rays)
            #print(self.all_rays.shape)
            self.w2cs = torch.cat(self.w2cs, 0)
            self.c2ws = torch.cat(self.c2ws, 0)
            self.intrinsics = torch.cat(self.intrinsics, 0)
            self.near_fars = torch.cat(self.near_fars, 0)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_depth = torch.stack(self.all_depth, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
            self.w2cs = torch.cat(self.w2cs, 0)
            self.c2ws = torch.cat(self.c2ws, 0)
            self.intrinsics = torch.cat(self.intrinsics, 0)
            self.near_fars = torch.cat(self.near_fars, 0)


    def __len__(self):
        #if self.split == 'train':
            #return len(self.all_rays)
            #return len(self.w2cs)
            #return len(self.c2ws)
            #return len(self.intrinsics)
            #return len(self.near_fars)
        return len(self.all_rays)
        return len(self.all_rgbs)
        return len(self.all_depth)
        return len(self.w2cs)
        return len(self.c2ws)
        return len(self.intrinsics)
        return len(self.near_fars)
        return len(self.imgs)
        return len(self.proj_mats)
        

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
            # sample = {'rays': self.all_rays[view,ray_idx],
            #           'rgbs': self.all_rgbs[view,ray_idx]}


            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'depth': self.all_depth[idx],
                      'w2cs': self.w2cs[idx],
                      'c2ws': self.c2ws[idx],
                      'intrinsics': self.intrinsics[idx],
                      'near_fars': self.near_fars[idx],
                      'imgs': self.imgs[idx],
                      'proj_mats': self.proj_mats[idx]
                      }

        else:  # create data for each image separately

            #img = self.all_rgbs[idx]
            #rays = self.all_rays[idx]
            #depth = self.all_depth[idx] # for quantity evaluation

            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'depth': self.all_depth[idx],
                      'w2cs': self.w2cs[idx],
                      'c2ws': self.c2ws[idx],
                      'intrinsics': self.intrinsics[idx],
                      'near_fars': self.near_fars[idx],
                      'imgs': self.imgs[idx],
                      'proj_mats': self.proj_mats[idx]
                      }
        sample['idx'] = idx
        return sample

