"""Example code for crop word patch from images"""
import os
import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np

def get_gt_info(gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    labels = []
    polys = []
    for line in lines:
        polys.append(list(map(int,line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 0 if '###' in line else 1
        labels.append(label)
    polys = np.array(polys, np.float32)
    labels = np.array(labels)
    if len(polys) > 0:
        polys = polys.reshape(-1, 4, 2)
    return polys, labels


def get_perspective_transform(src, dst):
    """Calculates a perspective transform from four pairs of the corresponding points.
    Args:
        src(tensor): Coordinates of quadrangle vertices in the source image, with shape (4,2)
        dst(tensor): Coordinates of the corresponding quadrangle vertices in the destination image
    Return:
        tensor: perspective transform matrix with shape (3,3)
    """
    
    device = dst.device
    assert src.shape[0] == dst.shape[0] and src.shape[0] == 4
    A = torch.zeros((8, 8), dtype=torch.float32, device=device)
    B = torch.zeros((8, 1), dtype=torch.float32, device=device)
    M = torch.ones((9, 1), dtype=torch.float32, device=device)
    for i in range(0, 4):
        x, y = src[i,:]
        u, v = dst[i,:]
        A[2*i] = torch.tensor([x, y, 1, 0, 0, 0, -u*x, -u*y], dtype=torch.float32, device=device)
        A[2*i+1] = torch.tensor([0, 0, 0, x, y, 1, -v*x, -v*y], dtype=torch.float32, device=device)     
        B[2*i] = u
        B[2*i+1] = v
    A_inverse = torch.inverse(A)
    M[:8, 0] = torch.matmul(A_inverse, B).reshape(-1)
    M = M.reshape((3, 3))
    return M


def grid_sample(img, sample_points, mode='bilinear', border_value=0):
    """Execute interpolation for given sample_points and input img.
    Args:
        img(tensor): input img, can be gray img or color img.
        sample_points(tensor): sample coordinates of output img, on input img.
        mode(str): interpolation mode.
        border_value(scalar): padding value for sample_points which cross boundary.
    return:
        tensor: output img after interpolation.
    """
    h, w = sample_points.shape[:2]
    assert img.ndim >= 2, 'invalid input img shape for interpolation'
    if img.ndim == 2:
        src_h, src_w = img.shape # gray image
        dst = torch.zeros((h, w), dtype=img.dtype)
    else:
        src_h, src_w, c = img.shape # color image
        dst = torch.zeros((h, w, c), dtype=img.dtype)

    if mode == 'bilinear':
        xs = sample_points[:, :, 0]
        ys = sample_points[:, :, 1]
        x1 = torch.floor(xs).to(torch.int64).clamp(0, src_w - 1)
        y1 = torch.floor(ys).to(torch.int64).clamp(0, src_h - 1)
        x2 = (x1 + 1).clamp(0, src_w - 1)
        y2 = (y1 + 1).clamp(0, src_h - 1)
        
        out_boundary = (xs < 0) | (xs > src_w - 1) | (ys < 0) | (ys > src_h - 1)
        
        if img.ndim == 2:
            points_tl = img[y1, x1].reshape(h, w)   # top_left
            points_tr = img[y1, x2].reshape(h, w)   # top_right
            points_bl = img[y2, x1].reshape(h, w)   # bottom_left
            points_br = img[y2, x2].reshape(h, w)   # bottom_right
            dst[:,:] = (x2 - xs) * (y2 - ys) * points_tl + \
                        (xs - x1) * (y2 - ys) * points_tr + \
                        (x2 - xs) * (ys - y1) * points_bl + \
                        (xs - x1) * (ys - y1) * points_br
            dst[out_boundary] = border_value
        else:
            for i in range(c):
                points_tl = img[y1, x1, i].reshape(h, w)   # top_left
                points_tr = img[y1, x2, i].reshape(h, w)   # top_right
                points_bl = img[y2, x1, i].reshape(h, w)   # bottom_left
                points_br = img[y2, x2, i].reshape(h, w)   # bottom_right
                dst[:, :, i] = (x2 - xs) * (y2 - ys) * points_tl + \
                                (xs - x1) * (y2 - ys) * points_tr + \
                                (x2 - xs) * (ys - y1) * points_bl + \
                                (xs - x1) * (ys - y1) * points_br
                dst[:, :, i][out_boundary] = border_value
                
    elif mode == 'nearest':
        sample_points = torch.floor(sample_points + 0.5).to(torch.int64)
        xs = sample_points[:, :, 0]
        ys = sample_points[:, :, 1]
        out_boundary = (xs < 0) | (xs > src_w - 1) | (ys < 0) | (ys > src_h - 1)
        
        xs = xs.reshape(-1).clamp(0, src_w - 1)
        ys = ys.reshape(-1).clamp(0, src_h - 1)
        if img.ndim == 2:
            dst[:,:] = img[ys, xs].reshape(h, w)
            dst[out_boundary] = border_value
        else:
            for i in range(c):
                dst[:,:,i] = img[ys, xs, i].reshape(h, w)
                dst[:, :, i][out_boundary] = border_value
    else:
        raise ValueError('unsupported interpolation mode.')

    return dst


if __name__ == '__main__':
    img_dir = 'D:\yanshuang\data\icdar2015_test\image'
    gt_dir = 'D:\yanshuang\data\icdar2015_test\meta'
    fix_h = 64
    total_num = len(os.listdir(img_dir))
    start = time.time()
    for idx, img_name in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, 'gt_' + img_name.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        src_h, src_w = img.shape[:2]
        polys, labels = get_gt_info(gt_path)
        
        # viz
        viz_img = img.copy()
        cv2.polylines(viz_img, [poly.astype(np.int32).reshape(4,2) for poly in polys], True, (0,0,255), 2)
        cv2.imshow('polys', cv2.resize(viz_img, dsize=None, fx=0.6, fy=0.6))
        cv2.waitKey(0)
        
        img = torch.from_numpy(img)
        polys = torch.from_numpy(polys)
        for poly in polys:
            cx = poly[:, 0].mean()
            cy = poly[:, 1].mean()
            
            w = (((poly[[1,3],:] - poly[[0,2],:]) ** 2).sum(dim=-1) ** 0.5).mean()
            h = (((poly[[2,0],:] - poly[[1,3],:]) ** 2).sum(dim=-1) ** 0.5).mean()
            
            if w < h/3:
                poly = torch.roll(poly, shift=1, dim=0)
                w, h = h, w
            x1 = cx - w/2
            x2 = cx + w/2
            y1 = cy - h/2
            y2 = cy + h/2
            dst_points = torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).to(torch.float32)
            perspective_M = get_perspective_transform(dst_points, poly)
            
            fy = h / fix_h
            dst_w = int(w / fy)
            fx = w / dst_w
            
            # translate + scale
            affine_M = torch.tensor([[fx, 0, x1], [0, fy, y1], [0,  0,  1]]).to(torch.float32)
            
            # Attention that, useage of torch.meshgrid() is different from np.meshgrid()
            ys, xs = torch.meshgrid(torch.arange(fix_h, dtype=torch.float32), torch.arange(dst_w, dtype=torch.float32))
            xys = torch.stack([xs, ys, torch.ones(xs.shape, dtype=xs.dtype)], dim=0).reshape(3, -1)
            
            M = torch.matmul(perspective_M, affine_M)
            sample_points = torch.matmul(M, xys)
            sample_points = sample_points[[0, 1], :] / sample_points[[2], :]
            sample_points = sample_points.permute(1, 0)
            
             # method1: do interpolation manually.
            ############################################################
            # sample_img = img.clone().to(torch.float32)
            # sample_points = sample_points.reshape(fix_h, dst_w, 2)
            # word_img = grid_sample(sample_img, sample_points, 'nearest')

            
            # method2: apply torch.nn.functional.grid_sample() function.
            ############################################################
            sample_points = sample_points.reshape(1, fix_h, dst_w, 2) # (1, dst_h, dst_w, 2)
            sample_img = img.clone().permute(2,0,1).unsqueeze(0).to(torch.float32) # (1, c, src_h, src_w)
            # normalize sample point into [-1, 1]
            sample_points[:,:,:,0] = sample_points[:,:,:,0] / src_w * 2 - 1
            sample_points[:,:,:,1] = sample_points[:,:,:,1] / src_h * 2 - 1
            
            word_img = F.grid_sample(sample_img, sample_points, mode='nearest', align_corners=False).squeeze(0).permute(1,2,0)
            
        
            
            word_img = word_img.numpy().astype(np.uint8)

            # viz crop result
            cv2.imshow('word_img', word_img)
            cv2.waitKey(0)
        print('{}/{}==>processed.'.format(idx+1, total_num))
    total_time = time.time() - start
    print('total_time = {}, fps = {:4f}'.format(total_time, total_num/total_time))
            
            
            
            
        
        
        