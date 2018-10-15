#!/usr/bin/env python2

import argh
import yaml
import numpy as np
import pickle
from collections import defaultdict, OrderedDict
import cv2
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import marker_ba as mba
import sys
import logging as logger
logger.basicConfig(level=logger.INFO)

def transform_points(world, rvec, tvec):
    angle = np.linalg.norm(rvec)
    if angle > 1e-6:
        axis = (rvec/angle).reshape(3, 1)
        x, y, z = axis.reshape(-1)
        R = np.array((((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))))
        R = np.cos(angle)*np.identity(3) + np.sin(angle)*R + (1 - np.cos(angle))*np.dot(axis, axis.T)
        world = np.inner(R, world).T
    tvec = tvec.reshape(-1)
    return world + tvec

def get_marker_points(geom, pose):
    return transform_points(geom*(np.exp(pose[-1])), *pose[:-1])

def get_marker_pose(geometry, points):
    centers = np.mean(geometry, axis=0), np.mean(points, axis=0)
    scales = (
        np.mean(np.sqrt(np.sum((geometry - centers[0])**2, axis=1))),
        np.mean(np.sqrt(np.sum((points - centers[1])**2, axis=1)))
        )
    scales = (1.0, 1.0)
    normed = (geometry - centers[0])/scales[0], (points - centers[1])/scales[1]
    
    cormat = np.sum([np.dot(p[0].reshape(-1, 1), p[1].reshape(-1, 1).T) for p in zip(*normed)], axis=0)
    U, _, V = np.linalg.svd(cormat)
    R = np.dot(U, V).T
    if np.linalg.det(R) < 0:
        V[:,-1] *= -1
        R = np.dot(U, V).T
    
    scale = scales[1]/scales[0]
    rvec = cv2.Rodrigues(R)[0]
    geom_est = transform_points(normed[0]*scales[1], rvec, np.zeros(3))
    tvec = np.mean(points - geom_est, axis=0)
    return rvec.reshape(1, -1), tvec.reshape(1, -1), np.log(scale)

    

def frame_subsample(frames, bandwidth=20.0):
    if len(frames) == 0: return frames
    msubsets = defaultdict(list)
    infids = []
    
    fids = list(frames)
    #random.shuffle(fids)
    
    def add_frame(fid, frame):
        for mid, marker in frame.iteritems():
            msubsets[mid].append(marker)
        infids.append(fid)

    for fid in fids:
        frame = frames[fid]
        if len(frame) < 2: continue
        for mid, marker in frame.iteritems():
            if mid not in msubsets:
                add_frame(fid, frame)
                break
            inframes = msubsets[mid]
            dists = np.sqrt(np.sum(np.subtract(frame[mid], np.array(inframes))**2, axis=(1, 2)))
            mindist = np.min(dists)
            if mindist > bandwidth:
                add_frame(fid, frame)
                break
    
    infids.sort()
    return OrderedDict((fid, frames[fid]) for fid in infids)

def get_reference_frame(frames):
    counts = defaultdict(int)
    for frame in frames.itervalues():
        for marker_id in frame.keys():
            counts[marker_id] += 1
    
    return max(counts.items(), key=lambda item: item[1])[0]

class PoseEstimateError(Exception): pass
def rough_marker_pose_estimate(cm, cd, marker_points, frame):
    worlds = []
    screens = []
    for mid, world in marker_points.iteritems():
        if mid not in frame: continue
        worlds.extend(world)
        screens.extend(frame[mid])
    if len(worlds) < 4:
        raise PoseEstimateError("Too few points")
    
    if len(worlds) < 5:
        ret, rvec, tvec = cv2.solvePnP(np.array(worlds), np.array(screens), cm, cd)
        if not ret:
            raise PoseEstimateError("SolvePnP failed")
    else:
        rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(worlds).astype(np.float32),
                np.array(screens).astype(np.float32), cm, cd)
        if inliers is None or len(inliers) != len(worlds):
            raise PoseEstimateError("Outliers in frame")
    return rvec.reshape(-1), tvec.reshape(-1)

def is_in_front_of_camera(points, pose):
    depth = transform_points(points, *pose)[:,-1]
    return np.all(depth > 0.0)

def to_homogenous(x):
    return np.column_stack((x, np.ones(len(x))))
def from_homogenous(x):
    return x[:,:-1]/x[:,-1:]

def random_pairs(a, include_max=100):
    idx = np.random.choice(len(a), include_max)
    combinations = list(itertools.combinations(idx, 2))
    while len(combinations) > 0:
        i1, i2 = combinations.pop(np.random.choice(len(combinations)))
        yield a[i1], a[i2]

def getProjectionMatrix(cm, rvec, tvec):
    T = np.empty((3,4))
    T[:,:3] = cv2.Rodrigues(rvec)[0]
    T[:,-1] = tvec.reshape(-1)
    return np.dot(cm, T)

class TriangulationError(Exception): pass
def multiviewTriangulation(cm, poses, points, max_frames=1000):
    if len(poses) < 10:
        raise TriangulationError("Not enough poses")
    ests = []
    for (p1, x1), (p2, x2) in random_pairs(zip(poses, points)):
        # TODO: This is wrong! Should be angle of the ray
        # to the pixel
        #if np.linalg.norm(x1 - x2) < 20.0: continue
        #if np.degrees(relative_rotation(p1, p2)) < 10.0: continue
        P1 = getProjectionMatrix(cm, *p1)
        P2 = getProjectionMatrix(cm, *p2)
        #est = triangulate_points(P1, P2, x1, x2)
        est = cv2.triangulatePoints(P1, P2, x1.T, x2.T).T
        est = est[:,:-1]/est[:,-1:]
        if not is_in_front_of_camera(est, p1): continue
        if not is_in_front_of_camera(est, p2): continue
        ests.append(est)
        if len(ests) > max_frames:
            break
    
    if len(ests) < 10:
        raise TriangulationError("Not enough poses")
    ests = np.array(ests)
    mest = np.median(ests, axis=0)
    """
    plt.plot(mest[:,0], mest[:,1])
    plt.plot(ests[:,0,0], ests[:,0,1], '.')
    plt.plot(ests[:,1,0], ests[:,1,1], '.')
    plt.plot(ests[:,2,0], ests[:,2,1], '.')
    plt.plot(ests[:,3,0], ests[:,3,1], '.')
    plt.show()
    """
    return mest

def multiviewMarkerTriangulation(cm, poses, marker_frames):
    poselist = []
    pointslist = []
    for i, pose in poses.iteritems():
        try:
            marker = marker_frames[i]
        except KeyError:
            continue
        poselist.append(pose)
        pointslist.append(marker.reshape(-1, 2))
    return multiviewTriangulation(cm, poselist, pointslist)

def multiview_marker_pose_triangulation(cm, geometry, poses, marker_frames):
    points = multiviewMarkerTriangulation(cm, poses, marker_frames)
    centers = np.mean(geometry, axis=0), np.mean(points, axis=0)
    #scales = np.mean(np.std(geometry, axis=0)), np.mean(np.std(points, axis=0))
    scales = (1.0, 1.0)
    normed = (geometry - centers[0])/scales[0], (points - centers[1])/scales[1]
    if len(points) < 4:
        raise TriangulationError("Not enough points for triangulation")
    
    cormat = np.sum([np.dot(p[0].reshape(-1, 1), p[1].reshape(-1, 1).T) for p in zip(*normed)], axis=0)
    U, _, V = np.linalg.svd(cormat)
    R = np.dot(U, V).T
    if np.linalg.det(R) < 0:
        V[:,-1] *= -1
        R = np.dot(U, V).T
    
    scale = scales[1]/scales[0]
    rvec = cv2.Rodrigues(R)[0]
    geom_est = transform_points(normed[0]*scales[1], rvec, np.zeros(3))
    tvec = np.mean(points - geom_est, axis=0)
    return rvec.reshape(1, -1), tvec.reshape(1, -1), np.log(scale)

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_markers(marker_points, color='red', ax=None):
    if ax is None:
        ax = plt.gcf().add_subplot(111, projection='3d')
    for marker_id, res in marker_points.iteritems():
        ax.text(res[0,0], res[0,1], res[0,2], str(marker_id))
        ax.plot(res[:,0], res[:,1], res[:,2], color=color)
    axisEqual3D(ax)
    return ax

def plot_camera_poses(poses, fmt='.'):
    acolors = ['red', 'green', 'blue']
    for i, color in enumerate(acolors):
        plt.subplot(2,1,1)
        plt.plot(poses.keys(), [k[1].reshape(-1)[i] for k in poses.values()], fmt, color=acolors[i])
        #plt.subplot(2,1,2)
        #plt.plot(poses.keys(), [rvec_to_euler(k[0])[i] for k in poses.values()], fmt, color=acolors[i])

def locate_markers(camera, marker_sizes, frame_times, frames, reference_id=None, known_markers=None):
    cm, cd = camera['camera_matrix'], camera['dist_coefs']

    all_frames = frames
    logger.info("Getting frame subsample")
    frames = frame_subsample(frames)
    logger.info("Using %i of %i frames"%(len(frames), len(all_frames)))
    if reference_id is None:
        reference_id = get_reference_frame(frames)
    else:
        reference_id = int(reference_id)
    
    marker_geometry = np.array([
        [-1,1,0],
        [1,1,0],
        [1,-1,0],
        [-1,-1,0],
        ]).astype(np.float)*0.5
    
    
    marker_geometries = {}
    for marker_id, size in marker_sizes.iteritems():
        marker_geometries[marker_id] = marker_geometry*size

    initial_marker_pose = lambda: (np.zeros(3), np.zeros(3), 0.0)
    def pointify(marker_poses):
        return {k: get_marker_points(marker_geometries[k], pose) for k, pose in marker_poses.iteritems()}
    
    """
    if known_markers is not None:
        # Transform the "known" markers so that the reference id is the
        # origin.
        reference_points = known_markers[reference_id]
        reference_transformation = get_marker_pose(reference_points, marker_geometries[reference_id])
        assert reference_transformation[-1] == 0
        reference_pose = reference_transformation[:-1]
        for mid in known_markers.keys():
            known_markers[mid] = transform_points(known_markers[mid], *reference_pose)
        known_marker_poses = {
                mid: get_marker_pose(marker_geometries[mid], known_markers[mid]) for mid in known_markers.keys()}
    else:
        known_marker_poses = {reference_id: initial_marker_pose()}
    """
    
    known_marker_poses = {reference_id: initial_marker_pose()}
    known_camera_poses = {}
    marker_blacklist = set()
    
    #plot_markers(pointify(known_marker_poses), color='blue'); plt.show()
    while True:
        # Get an initial estimate for new camera poses based
        # on markers found previously
        logger.info('Getting initial camera pose estimates')
        for fid, frame in frames.iteritems():
            if fid in known_camera_poses: continue
            n_markers = len(set(known_marker_poses).intersection(set(frame)))
            if n_markers < min(len(known_marker_poses), 2): continue
            try:
                known_camera_poses[fid] = rough_marker_pose_estimate(cm, cd, pointify(known_marker_poses), frame)
            except PoseEstimateError as e:
                continue
        
        #plot_camera_poses(known_camera_poses); plt.show()

        # Optimize new camera positions further
        logger.info('Optimizing camera poses')
        _, known_camera_poses, _ = mba.marker_ba(cm, frames, frame_times,
                known_camera_poses, known_marker_poses,
                marker_geometries, set([reference_id]),
                fix_features=True)
        
        # Find next marker to estimate
        subsamples = defaultdict(dict)
        for fid, frame in frames.iteritems():
            for mid, marker in frame.iteritems():
                if mid in known_marker_poses: continue
                if mid in marker_blacklist: continue
                if fid not in known_camera_poses: continue
                subsamples[mid][fid] = marker
        if len(subsamples) == 0:
            break

        winner = max(subsamples.iteritems(), key=lambda kv: len(kv[1]))[0]
        subsample = subsamples[winner]
        
        
        # Get initial estimates for new marker poses
        # based on new camera poses
        logger.info('Triangulating marker %i with %i poses'%(winner, len(subsample)))
        try:
            known_marker_poses[winner] = multiview_marker_pose_triangulation(cm, marker_geometry,
                known_camera_poses, subsample)
        except TriangulationError as e:
            logger.warning("Triangulation error: %s"%(e,))
            marker_blacklist.add(winner)
            continue
        
        # Do a bundle adjustment
        logger.info('Bundle adjustment for %i markers and %i camera poses'%(len(known_marker_poses), len(known_camera_poses)))
        _, known_camera_poses, known_marker_poses = mba.marker_ba(cm, frames, frame_times,
                known_camera_poses, known_marker_poses,
                marker_geometries, set([reference_id]))
        #print known_marker_poses
        
    #plot_markers(pointify(known_marker_poses), color='blue'); plt.show()
    return pointify(known_marker_poses)

    

def main(camera_spec, marker_sizes, marker_data, out_file, reference_id=None, known_markers=None):
    camera = pickle.load(open(camera_spec))
    marker_sizes = yaml.load(open(marker_sizes))
    
    if reference_id is not None:
        reference_id = int(reference_id)
    if known_markers is not None:
        known_markers = pickle.load(open(known_markers))

    marker_data_raw = np.load(marker_data)
    #marker_data_raw = marker_data_raw[:1000] # REMOVEME!
    frames = OrderedDict()
    frame_times = OrderedDict()
    for i, raw_frame in enumerate(marker_data_raw):
        frame = frames[i] = {}
        frame_times[i] = raw_frame['ts']
        for marker in raw_frame['markers']:
            if marker['id'] not in marker_sizes:
                continue
            if marker['loc_confidence'] < 0.99 or marker['id_confidence'] < 0.99: continue
            frame[marker['id']] = marker['verts'].reshape(-1, 2).astype(np.float64)
    
    markers = locate_markers(camera, marker_sizes, frame_times, frames, reference_id=reference_id, known_markers=known_markers)
    pickle.dump(markers, open(out_file, 'w'))
    

if __name__ == '__main__':
    argh.dispatch_command(main)
