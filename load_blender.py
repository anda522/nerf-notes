import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 绕y轴顺时针旋转
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

# angle, -30, 4
def pose_spherical(theta, phi, radius):
    # 平移矩阵，相机沿z轴平移radius位置
    c2w = trans_t(radius)
    # 旋转矩阵，绕x轴逆时针旋转phi角度，控制上下反向
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # 绕y轴顺时针旋转theta角度，控制左右方向
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # 坐标变换矩阵，将相机坐标系调整到所需要的方向（？）
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    # 存储训练、测试和验证的相机角度和每帧的旋转矩阵
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    # [0, 100, 113, 138] 验证100/8=13张，测试200/8=25张
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # 可以自定义非train时的skip,默认为8
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        # imgs: (100, 800, 800, 4) 将数据标准化
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    # [0,...,99], [100,...,112], [113,...,137]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    # imgs: (138, 800, 800, 4)
    imgs = np.concatenate(all_imgs, 0)
    # poses: (138, 4, 4)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    # 求相机焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # 将360度分成40等份，每个角度生成一个相机的姿态用4*4内参矩阵表示
    # pose_spherical相机位姿参数
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            # 该插值方法适合缩小图像，可以保留细节
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


