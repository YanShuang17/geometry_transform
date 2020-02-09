"""Example code for crop word patch from images"""
import os
import cv2
import time
import numpy as np
from geometry_transform import interpolation

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
        this function is equivalent to cv2.getPerspectiveTransform().
    Args:
        src(ndarray): Coordinates of quadrangle vertices in the source image, with shape (4,2)
        dst(ndarray): Coordinates of the corresponding quadrangle vertices in the destination image
    Return:
        ndarray: perspective transform matrix with shape (3,3)
    """
    assert src.shape[0] == dst.shape[0] and src.shape[0] == 4
    A = np.zeros((8, 8), np.float32)
    B = np.zeros((8, 1), np.float32)
    M = np.ones((9, 1), np.float32)
    for i in range(0, 4):
        x, y = src[i,:]
        u, v = dst[i,:]
        A[2*i] = np.array([x, y, 1, 0, 0, 0, -u*x, -u*y], np.float32)
        A[2*i+1] = np.array([0, 0, 0, x, y, 1, -v*x, -v*y], np.float32)     
        B[2*i] = u
        B[2*i+1] = v
    A_inverse = np.matrix(A).I
    M[:8, 0] = np.matmul(A_inverse, B).reshape(-1)
    M = M.reshape((3, 3))
    return M


def find_min_rect_angle(poly):
    poly = poly.reshape(4, 2).transpose(1, 0)
    angles = np.linspace(-90, 89.5, num=360) / 180 * np.pi
    rotate_matrix = np.stack([np.cos(angles), np.sin(angles) * -1, np.sin(angles), np.cos(angles)], axis=-1)
    rotate_matrix = rotate_matrix.reshape(-1, 2, 2)
    
    rotated_polys = np.matmul(rotate_matrix, angles)
    min_xs = rotated_polys[:,0,:].min(axis=-1)
    max_xs = rotated_polys[:,0,:].max(axis=-1)
    min_ys = rotated_polys[:,1,:].min(axis=-1)
    max_ys = rotated_polys[:,1,:].max(axis=-1)
    
    areas = (max_ys - min_ys) * (max_xs - min_xs)
    min_idx = np.argsort(areas)[0]

    return angles[min_idx]
    

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
        polys, labels = get_gt_info(gt_path)
        
        # viz
        viz_img = img.copy()
        cv2.polylines(viz_img, [poly.astype(np.int32).reshape(4,2) for poly in polys], True, (0,0,255), 2)
        cv2.imshow('polys', cv2.resize(viz_img, dsize=None, fx=0.6, fy=0.6))
        cv2.waitKey(0)
        
        for poly in polys:
            cx = poly[:, 0].mean()
            cy = poly[:, 1].mean()
            
            w = np.sqrt(((poly[[1,3],:] - poly[[0,2],:]) ** 2).sum(axis=-1)).mean()
            h = np.sqrt(((poly[[2,0],:] - poly[[1,3],:]) ** 2).sum(axis=-1)).mean()
            
            if w < h/3:
                poly = np.roll(poly, shift=1, axis=0)
                w, h = h, w
            x1 = cx - w/2
            x2 = cx + w/2
            y1 = cy - h/2
            y2 = cy + h/2
            dst_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            
            # perspective_M = get_perspective_transform(poly, dst_points)
            # # perspective_M = cv2.getPerspectiveTransform(poly, dst_points)
            # tmp_img = cv2.warpPerspective(img, perspective_M, dsize=(img.shape[1], img.shape[0]))
            # cv2.polylines(tmp_img, [dst_points.astype(np.int32)], True, (0,255,0), 1)
            # cv2.imshow('tmp_img', cv2.resize(tmp_img, dsize=None, fx=0.6, fy=0.6))
            # cv2.waitKey(0)
            
            perspective_M = get_perspective_transform(dst_points, poly)
            # perspective_M = cv2.getPerspectiveTransform(dst_points, poly)
            
            fy = h / fix_h
            dst_w = int(w / fy)
            fx = w / dst_w
            
            # translate + scale
            affine_M = np.array([[fx, 0, x1],
                                 [0, fy, y1],
                                 [0,  0,  1]], np.float32)
            # Attention that, useage of torch.meshgrid() is different from np.meshgrid()
            xs, ys = np.meshgrid(np.arange(dst_w).astype(np.float32), np.arange(fix_h).astype(np.float32))
            xys = np.stack([xs, ys, np.ones(xs.shape, xs.dtype)], axis=0).reshape(3, -1)
            
            M = np.matmul(perspective_M, affine_M)
            sample_points = np.matmul(M, xys)
            sample_points = sample_points[[0, 1], :] / sample_points[[2], :]
            sample_points = sample_points.transpose(1, 0).reshape(fix_h, dst_w, 2)
            
            # viz sample points
            # viz_sample = img.copy()
            # points = sample_points.astype(np.int32).reshape(-1,2)
            # for (x,y) in points:
            #     cv2.circle(viz_sample, (x,y), 1, (0, 0, 255), 1)
            # cv2.imshow('viz_sample', cv2.resize(viz_sample, dsize=None, fx=0.6, fy=0.6))
            # cv2.waitKey(0)

            word_img = interpolation(img.copy(), sample_points, mode='bilinear')
            
            # viz crop result
            cv2.imshow('word_img', word_img)
            cv2.waitKey(0)
        print('{}/{}==>processed.'.format(idx+1, total_num))
    total_time = time.time() - start
    print('total_time = {}, fps = {:4f}'.format(total_time, total_num/total_time))
            
            
            
            
            
            
        
        
        