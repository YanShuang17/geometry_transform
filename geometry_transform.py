import cv2
import math
import numpy as np

def Kronecker_delta(x, y):
    return 0 if x!=y else 1

def bilinear_interpolation(img, sample_points, border_value=0):
    """
    Execute bilinear interpolation for given sample_points and input img.
    Args:
        img(ndarray): input img, can be gray img or color img.
        sample_points(ndarray): sample coordinates of output img, on input img.
        border_value(scalar): padding value for sample_points which cross boundary.
    return:
        ndarray: output img after interpolation.
    """
    
    h, w = sample_points.shape[:2]
    assert img.ndim >= 2, 'invalid input img shape for interpolation'
    if img.ndim == 2:
        src_h, src_w = img.shape # gray image
        dst = np.zeros((h, w), img.dtype)
    else:
        src_h, src_w, c = img.shape # color image
        dst = np.zeros((h, w, c), img.dtype)
    
    # fast mode
    #########################################
    xs = sample_points[:, :, 0]
    ys = sample_points[:, :, 1]
    x1 = np.floor(xs).astype(np.int32).clip(0, src_w - 1)
    y1 = np.floor(ys).astype(np.int32).clip(0, src_h - 1)
    x2 = (x1 + 1).clip(0, src_w - 1)
    y2 = (y1 + 1).clip(0, src_h - 1)
    
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
    # source mode, very slow...
    #########################################
    # if img.ndim == 2:
    #     for i in range(h):
    #         for j in range(w):
    #             dst_x, dst_y = sample_points[i, j]
    #             for y in range(src_h):
    #                 for x in range(src_w):
    #                     dst[i, j] += img[y, x] * max(0, 1-abs(x-dst_x)) * max(0, 1-abs(y-dst_y))
    # else:
    #     for k in range(c):
    #         for i in range(h):
    #             for j in range(w):
    #                 dst_x, dst_y = sample_points[i, j]
    #                 for y in range(src_h):
    #                     for x in range(src_w):
    #                         dst[i, j, k] += img[y, x, k] * max(0, 1-abs(x-dst_x)) * max(0, 1-abs(y-dst_y))
    
    return dst


def nearest_interpolation(img, sample_points, border_value=0):
    """
    Execute nearest interpolation for given sample_points and input img.
    Args:
        img(ndarray): input img, can be gray img or color img.
        sample_points(ndarray): sample coordinates of output img, on input img.
        border_value(scalar): padding value for sample_points which cross boundary.
    return:
        ndarray: output img after interpolation.
    """
    
    h, w = sample_points.shape[:2]
    assert img.ndim >= 2, 'invalid input img shape for interpolation'
    if img.ndim == 2:
        src_h, src_w = img.shape # gray image
        dst = np.zeros((h, w), img.dtype)
    else:
        src_h, src_w, c = img.shape # color image
        dst = np.zeros((h, w, c), img.dtype)
    
    # fast mode
    ######################################
    sample_points = np.floor(sample_points + 0.5).astype(np.int32)
    xs = sample_points[:, :, 0]
    ys = sample_points[:, :, 1]
    out_boundary = (xs < 0) | (xs > src_w - 1) | (ys < 0) | (ys > src_h - 1)
    
    xs = xs.reshape(-1).clip(0, src_w - 1)
    ys = ys.reshape(-1).clip(0, src_h - 1)
    if img.ndim == 2:
        dst[:,:] = img[ys, xs].reshape(h, w)
        dst[out_boundary] = border_value
    else:
        for i in range(c):
            dst[:,:,i] = img[ys, xs, i].reshape(h, w)
            dst[:, :, i][out_boundary] = border_value
            
    # source mode, very slow...
    ######################################
    # if img.ndim == 2:
    #     for i in range(h):
    #         for j in range(w):
    #             dst_x, dst_y = sample_points[i, j]
    #             for y in range(src_h):
    #                 for x in range(src_w):
    #                     dst[i, j] += img[y, x] * \
    #                                     Kronecker_delta(np.floor(dst_x + 0.5), x) * \
    #                                     Kronecker_delta(np.floor(dst_y + 0.5), y)
    # else:
    #     for k in range(c):
    #         for i in range(h):
    #             for j in range(w):
    #                 dst_x, dst_y = sample_points[i, j]
    #                 for y in range(src_h):
    #                     for x in range(src_w):
    #                         dst[i, j, k] += img[y, x, k] * \
    #                                         Kronecker_delta(np.floor(dst_x + 0.5), x) * \
    #                                         Kronecker_delta(np.floor(dst_y + 0.5), y)
    
    return dst

def interpolation(img, sample_points, mode='bilinear', border_value=0):
    """
    Execute interpolation for given sample_points and input img.
    Args:
        img(ndarray): input img, can be gray img or color img.
        sample_points(ndarray): sample coordinates of output img, on input img.
        mode(str): interpolation mode.
        border_value(scalar): padding value for sample_points which cross boundary.
    return:
        ndarray: output img after interpolation.
    """
    
    if mode == 'bilinear':
        return bilinear_interpolation(img, sample_points)
    elif mode == 'nearest':
        return nearest_interpolation(img, sample_points)
    else:
        raise ValueError('unsupported interpolation mode.')

    
def resize(img, dsize, fx=None, fy=None, mode='bilinear'):
    """Resize img for given dsize(w, h) or fx/fy.
    """
    
    if (dsize is not None) and (fx is not None or fy is not None):
        raise KeyError("should specify one of keys: [dsize, fx/fy]")
    elif (dsize is None) and (fx is None or fy is None):
        raise KeyError("should specify one of keys: [dsize, fx/fy]")
    if dsize is not None:
        w, h = dsize
    else:
        src_h, src_w = img.shape[:2]
        w = int(src_w * fx)
        h = int(src_h * fy)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    
    # shape = (3, h*w)
    xys = np.stack([xs, ys, np.ones(xs.shape, xs.dtype)], axis=0).reshape(3, -1).astype(np.float32)
    fx = src_w / w
    fy = src_h / h
    # resize/scale is a kind of Affine Transform, just applying 2*3 transform matrix.
    scale_mat = np.array([[fx, 0,  0],
                          [0,  fy, 0]], np.float32)
    
    sample_points = np.matmul(scale_mat, xys).transpose(1, 0).reshape(h, w, 2)
    return interpolation(img, sample_points, mode)

    
def crop(img, start, end, mode='bilinear'):
    """Crop img"""
    x1, y1 = start
    x2, y2 = end
    h, w = img.shape[:2]
    assert 0 <= x1 <= w, 'invalid crop value(x1)'
    assert 0 <= x2 <= w, 'invalid crop value(x2)'
    assert 0 <= y1 <= h, 'invalid crop value(y1)'
    assert 0 <= y2 <= h, 'invalid crop value(y2)'
    
    xs, ys = np.meshgrid(np.arange(x2-x1).astype(np.float32), np.arange(y2-y1).astype(np.float32))
    xys = np.stack([xs, ys, np.ones(xs.shape, xs.dtype)], axis=0).reshape(3, -1)
    
    # crop is a kind of Affine Transform, just like translation.
    translate_mat = np.array([[1, 0, x1],
                              [0, 1, y1]], np.float32)
    sample_points = np.matmul(translate_mat, xys).transpose(1, 0).reshape(xs.shape[0], xs.shape[1], 2)
    return interpolation(img, sample_points, mode)


def translate(img, offset, mode='bilinear'):
    """Translate img"""
    delta_x, delta_y = offset
    h, w = img.shape[:2]
    
    xs, ys = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32))
    xys = np.stack([xs, ys, np.ones(xs.shape, xs.dtype)], axis=0).reshape(3, -1)
    
    # translation is a kind of Affine Transform.
    translate_mat = np.array([[1, 0, -delta_x],
                              [0, 1, -delta_y]], np.float32)
    sample_points = np.matmul(translate_mat, xys).transpose(1, 0).reshape(xs.shape[0], xs.shape[1], 2)
    return interpolation(img, sample_points, mode)


def rotate(img, center, angle, dsize=None, mode='bilinear'):
    """
    Rotate img for given rotation center(cx, cy) and angle.
    Args:
        img(ndarray): input img, can be gray img or color img.
        center(tuple): rotation center
        angle(float): rotateion angle, positive for anticlockwise
        mode(str): interpolation mode
    return:
        ndarray: result img after ratation.
    """
    theta = angle / 180 * math.pi
    # rotation center of rotate_mat is origin/(0,0)
    rotate_mat = np.array([[math.cos(theta), -math.sin(theta), 0],
                           [math.sin(theta), math.cos(theta),  0]], np.float32)
    if dsize == None:
        h, w = img.shape[:2]
    else:
        w, h = dsize
    
    xs, ys = np.meshgrid(np.arange(w).astype(np.float32), np.arange(h).astype(np.float32))
    # if rotate img at specified rotation center(center_x, center_y), you should:
    # 1. translate img first, so (center_x, center_y) becomes origin.
    # 2. do standard rotation.
    # 3. translate img back.
    center_x, center_y = center
    xs -= center_x
    ys -= center_y
    xys = np.stack([xs, ys, np.ones(xs.shape, xs.dtype)], axis=0).reshape(3, -1)
    sample_points = np.matmul(rotate_mat, xys).transpose(1, 0).reshape(h, w, 2)
    sample_points[:, :, 0] += center_x
    sample_points[:, :, 1] += center_y
    
    return interpolation(img, sample_points, mode)

if __name__ == '__main__':
    image_path = 'sample.jpg'
    img = cv2.imread(image_path)
    cv2.imshow('src_img', img)
    cv2.waitKey(0)

    # resize
    ####################################
    # fx = 1.2
    # fy = 1.2
    # resize_img = resize(img, dsize=None, fx=fx, fy=fy, mode='bilinear')
    # cv2_resize_img = cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    
    # cv2.imshow('resize_img', resize_img)
    # cv2.imshow('cv2_resize_img',  cv2_resize_img)
    # cv2.waitKey(0)
    #####################################


    # crop
    #####################################
    h, w = img.shape[:2]
    start = (int(w * 0.1), int(h * 0.1))
    end = (int(w * 0.8), int(h * 0.8))
    crop_img = crop(img, start, end)
    cv2.imshow('crop_img', crop_img)
    cv2.waitKey(0)
    #####################################


    # translation
    #####################################
    h, w = img.shape[:2]
    offset = (100, 200)
    translate_img = translate(img, offset, mode='nearest')
    cv2.imshow('translate_img', translate_img)
    cv2.waitKey(0)
    #####################################


    # rotate
    #####################################
    angle = 30
    center_x = 50
    center_y = 300
    # center_x = (img.shape[1] - 1) / 2
    # center_y = (img.shape[0] - 1) / 2
    rotate_img = rotate(img, center=(center_x, center_y), angle=angle, dsize=None, mode='bilinear')
    cv2.imshow('rotate_img', rotate_img)
    
    rotation_M = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=angle, scale=1)
    cv2_rotate_img = cv2.warpAffine(img, rotation_M, dsize=(img.shape[1], img.shape[0]))
    cv2.imshow('cv2_rotate_img', cv2_rotate_img)
    cv2.waitKey(0)
    


    

    