
from torch.utils.data import Dataset
from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R

class MVSDatasetPortrait(Dataset):
    def __init__(self, root_dir, split, n_views=3, levels=1, img_wh=None, downSample=1.0, max_len=-1):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        print("root_dir: ", root_dir)
        self.split = split

        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0
        self.max_len = max_len
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'
        self.build_metas()
        self.n_views = n_views
        self.levels = levels  # FPN levels
        self.build_proj_mats()
        self.define_transforms()
        print(f'==> image down scale: {self.downSample}')

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    #T.Normalize(mean=[0.485, 0.456, 0.406],
                                                #std=[0.229, 0.224, 0.225]),
                                    ])

    def build_metas(self):
        list = sorted(os.listdir(f'{self.root_dir}/rendered_sample75_512/pairs/'))
        if self.split=='train':
            self.scans = [l[:-4] for l in list[:60]]
        elif self.split=='val':
            self.scans = [l[:-4] for l in list[60:65]]
        else:
            self.scans = [l[:-4] for l in list[65:75]]

        self.metas = []
        for scan in self.scans:
            pairs = np.loadtxt(f'{self.root_dir}/rendered_sample75_512/pairs/{scan}.txt').astype('int')
            # viewpoints (49)
            for pair in pairs:
                ref_view = int(pair[0])
                src_views = pair[1:]
                self.metas += [(scan, ref_view, src_views)]


    def build_proj_mats(self):
        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = {}, {}, {}, {}
        for meta in self.metas:
            scan, ref_view, _ = meta

            proj_mat_filename = os.path.join(self.root_dir,f'rendered_sample75_512/cams/{scan}/{ref_view:03d}_cam.txt')
            intrinsic, extrinsic = self.read_cam_file(proj_mat_filename)
            near_far = np.loadtxt(f'{self.root_dir}/rendered_sample75_512/minmaxs/{scan}/minmax_map_{ref_view:03d}.txt').tolist()
            # intrinsic[:2] *= 4
            # extrinsic[:3, 3] *= self.scale_factor

            intrinsic[:2] = intrinsic[:2] * self.downSample
            self.intrinsics[f'{scan}_{ref_view:03d}'] = intrinsic.copy()

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]

            self.proj_mats[f'{scan}_{ref_view:03d}']  = (proj_mat_l, near_far)
            self.world2cams[f'{scan}_{ref_view:03d}'], self.cam2worlds[f'{scan}_{ref_view:03d}'] = extrinsic, np.linalg.inv(extrinsic)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        return intrinsics, extrinsics

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        mask = depth_h > 0

        return  mask, depth_h


    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}
        scan, target_view, src_views = self.metas[idx]
        if self.split=='train':
            ids = torch.randperm(6)[:3]
            view_ids = [src_views[i] for i in ids] + [target_view]
        else:
            view_ids = [src_views[i] for i in range(3)] + [target_view]

        affine_mat, affine_mat_inv = [], []
        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i, vid in enumerate(view_ids):

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'rendered_sample75_512/images/{scan}/{vid:03d}.png')
            depth_filename = os.path.join(self.root_dir,
                                          f'rendered_sample75_512/depths/{scan}/depth_map_{vid:03d}.pfm')

            img = Image.open(img_filename).convert('RGB')
            img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
            img = img.resize(img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            proj_mat_ls, near_far = self.proj_mats[f'{scan}_{vid:03d}']
            intrinsics.append(self.intrinsics[f'{scan}_{vid:03d}'])
            w2cs.append(self.world2cams[f'{scan}_{vid:03d}'])
            c2ws.append(self.cam2worlds[f'{scan}_{vid:03d}'])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

            if os.path.exists(depth_filename):
                mask, depth_h = self.read_depth(depth_filename)
                depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                depths_h.append(np.zeros((1, 1)))

            near_fars.append(near_far)

        imgs = torch.stack(imgs).float()
        depths_h = np.stack(depths_h)
        proj_mats = np.stack(proj_mats)[:, :3]
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)

        sample['images'] = imgs  # (V, H, W, 3)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        sample['scan'] = scan

        return sample


