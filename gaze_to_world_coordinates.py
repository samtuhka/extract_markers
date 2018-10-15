#!/usr/bin/env python2

#import cPickle as pickle
#import smoothpose
#import yaml
import numpy as np
import cv2
import scipy.interpolate
from pprint import pprint

def pixel_to_angle(pixel, cm):
    f = np.array([cm[0,0], cm[1,1]])
    c = np.array([cm[0,2], cm[1,2]])
    return np.arctan((pixel - c)/f)

def pixel_to_direction(pixel, cm):
    f = np.array([cm[0,0], cm[1,1]])
    c = np.array([cm[0,2], cm[1,2]])
    n = (pixel - c)/f
    z = 1.0/np.sqrt(np.sum(n**2) + 1)
    x, y = n*z
    return np.array([x, y, z])

def relative_pose(pose_src, pose_dst):
    src_r, src_t = pose_src
    dst_r, dst_t = pose_dst
    
    dst_t = dst_t.reshape(3, 1)
    src_t = src_t.reshape(3, 1)
    
    src_R = cv2.Rodrigues(src_r)[0]
    dst_R = cv2.Rodrigues(dst_r)[0]
    
    rel_R = np.dot(src_R, dst_R.T)
    rel_t = -np.dot(rel_R, dst_t) + src_t
    rel_r = cv2.Rodrigues(rel_R)[0]

    return rel_r.reshape(3), rel_t.reshape(3)

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

def invert_pose(rvec, tvec):
    R = cv2.Rodrigues(rvec)[0].T
    return cv2.Rodrigues(R)[0], np.dot(R, -tvec)

def pose_interpolator(ts, rvecs, tvecs):
    quats = smoothpose.Q.from_rvec(rvecs)
    pinterp = smoothpose.interpolate_pose(ts, quats, tvecs)
    def interp(ts):
        q, t = pinterp(ts)
        return smoothpose.Q.to_rvec(q), t
    return interp

# Reimplementation of OpenCV 3's fisheye undistortion.
def undistort_points_fisheye(distorted, K, D, P):
    undistorted = np.ones((len(distorted), 3), dtype=distorted.dtype)
    K = np.asarray(K)
    f = np.array([K[0,0], K[1,1]])
    c = np.array([K[0,2], K[1,2]])
    D = np.asarray(D)
    P = np.asarray(P)
    
    for i in range(len(distorted)):
        pw = (distorted[i] - c)/f
        theta_d = np.linalg.norm(pw)
        # This is in the original implementation, although
        # theta_d is clearly always positive
        theta_d = np.clip(theta_d, -np.pi/2.0, np.pi/2.0)
        scale = 1.0
        if theta_d > 1e-8:
            theta = theta_d
            for j in range(10):
                theta = theta_d/(1 + np.sum(D*np.power(theta, [2,4,6,8])))
            scale = np.tan(theta)/theta_d
        undistorted[i,:2] = pw*scale
    
    undistorted = np.inner(undistorted, P)
    undistorted /= undistorted[:,-1]
    return undistorted[:,:2]

def main(camera_mapping, origin_pose, poses, video, frame_times, pupil_data, out_pupil_data):
    camera = pickle.load(open(camera_mapping))
    K, D, resolution, cm = camera['camera_matrix'], camera['dist_coefs'], camera['resolution'], camera['rect_camera_matrix']
    assert camera['distortion_model'] == 'fisheye'
    
    origin_pose = yaml.load(open(origin_pose))
    ref_rvec = np.array(origin_pose['rvec'])
    ref_tvec = np.array(origin_pose['tvec'])
    origin_pose = (ref_rvec, ref_tvec)
    
    frame_times = np.load(frame_times)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    video_to_pupil = scipy.interpolate.interp1d(
            np.arange(len(frame_times))/fps,
            frame_times, bounds_error=False)
    poses = pickle.load(open(poses))
    poses = zip(*((p['time'], p['rvec'], p['tvec']) for p in poses))
    pose_ts, rvecs, tvecs = map(np.array, poses)
    pose_ts = video_to_pupil(pose_ts)
    poses = pose_interpolator(pose_ts, rvecs, tvecs)

    pupil_data = pickle.load(open(pupil_data))
    pupil_data['world_origin'] = origin_pose
    
    test_point_depth = 10.0

    for frame in pupil_data['gaze_positions']:
        try:
            pose = relative_pose(poses(frame['timestamp']), origin_pose)
        except ValueError:
            continue
        gaze = np.array(frame['norm_pos'])
        gaze[1] = 1 - gaze[1]
        gaze *= resolution
        gaze = undistort_points_fisheye(gaze.reshape(1, 2), K, D, cm).reshape(2)
        eye_in_head = pixel_to_direction(gaze, cm)
        
        
        virtual_target = transform_points((eye_in_head*10.0).reshape(1, -1), *invert_pose(*pose)).reshape(-1)
        heading = np.arctan2(virtual_target[0], virtual_target[2])
        pitch = np.arctan2(virtual_target[1], virtual_target[2])
        w = {
            'camera_pose': pose,
            'eye_in_head': eye_in_head,
            'gaze_angle_at_10m_target': np.array((heading, pitch))
            }
        frame['world'] = w

    pickle.dump(pupil_data, open(out_pupil_data, 'w'))

if __name__ == '__main__':
    import argh
    argh.dispatch_command(main)
