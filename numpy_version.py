import os
import re
import json
#from functools import lru_cache

import cv2
import numpy as np
from shapely.geometry import Polygon


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring



def transform_points(points, mat):
    points = np.array(points)
    mat = np.array(mat)
    if points.ravel().shape[0] == 0:
        return points
    if mat.shape[0] == 2:
        mat = np.concatenate([mat, np.array([[0, 0, 1]])], axis=0)

    shape = points.shape
    if shape[-1] == 2:
        points = np.concatenate([points, np.ones(shape[:-1] + (1, ))], axis=len(shape) - 1)
    points = points.dot(mat.T)
    if shape[-1] == 2:
        points = points[..., :2] / (points[..., 2:3] + (points[..., 2:3] == 0) * 1e-6)

    return points


def rect_to_corner_points(rect):
    rect = np.array(rect)
    assert(rect.shape[-1] == 4)
    rect = np.array([rect[..., 0], rect[..., 1], rect[..., 2], rect[..., 1], rect[..., 2], rect[..., 3], rect[..., 0], rect[..., 3]])
    rect = rect.reshape((8, -1)).transpose(1, 0).reshape(rect.shape[2:] + (4, 2))
    return rect


def distance(p0, p1):
    return ((p0 - p1) ** 2).sum() ** 0.5


def euclidian_distance(p0, p1):
    return distance(p0, p1)


def vec_length(x):
    return np.sqrt((x[0] ** 2 + x[1] ** 2).sum())


def unit_vector(v):
    return v / np.linalg.norm(v)


def unit_dir(x):
    return unit_vector(x)


def get_polygon_iou(rect0, rect1):
    rect0 = Polygon(rect0)
    rect1 = Polygon(rect1)
    i = rect0.intersection(rect1).area
    u = rect0.union(rect1).area
    return i / u


def get_angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotate_image(img, angle):
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
    cos, sin = abs(mat[0, 0]), abs(mat[1, 0])
    nh = int(h * cos + w * sin + 1)
    nw = int(w * cos + h * sin + 1)
    mat[0, 2] += (nw - w) // 2
    mat[1, 2] += (nh - h) // 2
    assert nh > 0 and nw > 0, '{},{}'.format(nh, nw)
    dst = cv2.warpAffine(img, mat, (nw, nh)) if angle != 0 else img

    return dst, mat


def get_rotated_bounding_box(pts, rad):
    c = np.cos(rad)
    s = np.sin(rad)
    xs = c * pts[:, 0] + s * pts[:, 1]
    ys = -s * pts[:, 0] + c * pts[:, 1]

    minx = xs.min()
    maxx = xs.max()
    miny = ys.min()
    maxy = ys.max()

    bbox = np.array([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)], 'float32')
    rect = np.stack([c * bbox[:, 0] - s * bbox[:, 1], s * bbox[:, 0] + c * bbox[:, 1]], axis=1)
    return rect


# median of numbers, distance(x, y) = min(max(x, y) - min(x, y), 1 - max(x, y) + min(x, y))
def cyclic_median(a, buckets=1000):
    count = np.bincount((np.array(a, 'float32') * buckets).astype('int32'), minlength=buckets)
    idx1 = np.tile(np.arange(buckets)[:, None], (1, buckets))
    idx2 = np.tile(np.arange(buckets)[None], (buckets, 1))
    d = np.minimum(np.maximum(idx1, idx2) - np.minimum(idx1, idx2), buckets - np.maximum(idx1, idx2) + np.minimum(idx1, idx2))
    best_bucket = (d * count[:, None]).sum(axis=0).argmin(axis=0)
    return (best_bucket + 0.5) / buckets


def expand_quad(rect, expand_ratio):
    '''
    rect: 4,2  x,y
    expand_ratio, x,y
    shift, x,y
    '''
    expand_ratio = (expand_ratio[0] / 2, expand_ratio[1] / 2)
    new_rect = rect.copy()
    new_rect[1] = (rect[1] + expand_ratio[0] * (rect[1] - rect[0]) + expand_ratio[1] * (rect[1] - rect[2]))
    new_rect[2] = (rect[2] + expand_ratio[0] * (rect[2] - rect[3]) + expand_ratio[1] * (rect[2] - rect[1]))
    new_rect[3] = (rect[3] + expand_ratio[0] * (rect[3] - rect[2]) + expand_ratio[1] * (rect[3] - rect[0]))
    new_rect[0] = (rect[0] + expand_ratio[0] * (rect[0] - rect[1]) + expand_ratio[1] * (rect[0] - rect[3]))

    return new_rect

def resize_ensure_shortest_edge_with_limit(img, edge_length, max_length, return_ratio=False):
    assert isinstance(edge_length, int) and isinstance(max_length, int) and \
            max_length >= edge_length and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, min(max(1, int(ratio*w)), max_length)
        ratio = tw / th
    else:
        ratio = float(edge_length) / w
        th, tw = min(max(1, int(ratio*h)), max_length), edge_length
        ratio = th / tw

    ret = cv2.resize(img, (tw, th))
    if return_ratio:
        return ret, ratio
    return ret

def cal_phi(a_points, b_points):
    N = a_points.shape[0]
    M = b_points.shape[0]

    pair_diff = a_points.reshape(N, 1, 2) - b_points.reshape(1, M, 2)
    #import pdb;pdb.set_trace()
    pair_diff_square = pair_diff * pair_diff
    pair_dis_square = pair_diff_square.sum(axis = 2)
    phi_r = 0.5 * pair_dis_square * np.ma.log(pair_dis_square).filled(0)
    return phi_r

def calc_delta_c(dst_points):
    N = dst_points.shape[0]
    ret = np.zeros((N + 3, N + 3), dtype = np.float32)
    ret[0:1, :N] = 1
    ret[1:3, :N] = np.transpose(dst_points)
    ret[3:, :N] = cal_phi(dst_points, dst_points)
    ret[3:, N:N+1] = 1
    ret[3:, N+1:] = dst_points
    return ret

def calc_transformation(src_points, dst_points):
    N = src_points.shape[0]
    left_matrix = np.zeros((2, N+3), dtype = np.float32)
    left_matrix[:, :N] = np.transpose(src_points)
    delta_c = calc_delta_c(dst_points)
    ret = np.matmul(left_matrix, np.linalg.pinv(delta_c))
    return ret

def cal_projection_matrix(dst_control_points, dst_points):
    N = dst_control_points.shape[0]
    M = dst_points.shape[0]
    assert N >= 3, N
    ret = np.ones((N + 3, M), dtype = np.float32)
    phi = cal_phi(dst_control_points, dst_points.reshape(-1, 2)) #N, M
    ret[3:, :] = phi
    ret[1:3, :] = np.transpose(dst_points).reshape(2, M)
    return ret

def tpsPoints(src_points, dst_points, points):
    '''
    src_points: np.array:n,2
    return: k,2
    '''
    assert src_points.shape[0] == dst_points.shape[0]
    T = calc_transformation(src_points, dst_points)
    proj = cal_projection_matrix(dst_points, points)
    ret = np.matmul(T, proj)
    return np.transpose(ret)

def tpsImg(src_points, dst_points, img, target_size):
    assert src_points.shape[0] == dst_points.shape[0]
    if src_points.shape[0] < 3: return img
    #assert src_points.shape[0] >= 3, src_points.shape[0]#actrually, <3 is also okay, but not enough to determine a plane
    #tps
    w, h = target_size
    x = np.repeat(np.arange(w).reshape(1, -1), h, axis = 0).reshape(-1, 1)
    y = np.repeat(np.arange(h).reshape(-1, 1), w, axis = 1).reshape(-1, 1)
    coords = np.concatenate([x, y], axis=1)
    src_coords = tpsPoints(src_points, dst_points, coords)
    src_coords = (src_coords + 0.5).astype('int32')#N, 2

    #crop
    src_coords = src_coords + 1
    sh, sw = img.shape[:2]
    img = np.concatenate([np.zeros([1, sw, 3]), img, np.zeros([1, sw, 3])], axis = 0)
    img = np.concatenate([np.zeros([sh + 2, 1, 3]), img, np.zeros([sh + 2, 1, 3])], axis = 1)
    x = np.clip(src_coords[:, 0], 0, sw + 1, ).reshape(-1, 1)
    y = np.clip(src_coords[:, 1], 0, sh + 1, ).reshape(-1, 1)
    dst = img[y, x].reshape(h, w, 3).astype('uint8')
    return dst


# util function for image drawing
class ImageDraw(object):
    def __init__(self, img, longer_side=768):
        self.scale = longer_side / max(img.shape[0], img.shape[1])
        self.img = cv2.resize(img, (int(self.scale * img.shape[1] + 1), int(self.scale * img.shape[0] + 1)))

    def polylines(self, pts, *args, **kwargs):
        pts = np.array(np.array(pts, 'float32') * self.scale, 'int32')
        cv2.polylines(self.img, pts, *args, **kwargs)

    def circle(self, center, *args, **kwargs):
        cv2.circle(self.img, (int(center[0] * self.scale), int(center[1] * self.scale)), *args, **kwargs)

    def result(self):
        return self.img

if __name__ == '__main__':
    #tps test
    src_points = np.array([100, 100, 500, 100, 400, 400, 0, 400]).reshape(-1, 2)
    dst_points = np.array([0, 0, 400, 0, 400, 400, 0, 400]).reshape(-1, 2)
    points = np.array([10, 0, 10, 10, 30, 30, 0, 20, 50, 50]).reshape(-1, 2)
    ret = tpsPoints(src_points, dst_points, points)
    print(ret)
    img = cv2.imread('./test_image.jpg')
    h, w = img.shape[:2]
    dst = tpsImg(src_points, dst_points, img, (w,h))
    cv2.imshow('src', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
