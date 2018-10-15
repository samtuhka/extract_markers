#!/usr/bin/env python2

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cPickle as pickle
import argh

def estimate_poses(camera, marker_positions, marker_data):
    cm, cd = camera['camera_matrix'], camera['dist_coefs']
    poses = []
    for frame in marker_data:
        world = []
        screen = []
        marker_ids = set()
        for marker in frame['markers']:
            if marker['id'] not in marker_positions: continue
            marker_ids.add(marker['id'])
            world.extend(marker_positions[marker['id']])
            screen.extend(marker['verts'])
        if len(world) < 4: continue
        if len(marker_ids) < 2: continue
        rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(world).astype(np.float32),
                np.array(screen).astype(np.float32), cm, cd)
        poses.append({
            'time': frame['ts'],
            'rvec': rvec,
            'tvec': tvec
            })
    return poses 

def main(camera_spec, marker_positions, marker_data, out_file):
    camera = pickle.load(open(camera_spec))
    marker_positions = pickle.load(open(marker_positions))
    marker_data = np.load(marker_data)

    poses = estimate_poses(camera, marker_positions, marker_data)
    pickle.dump(poses, open(out_file, 'w'))

if __name__ == '__main__':
    argh.dispatch_command(main)
