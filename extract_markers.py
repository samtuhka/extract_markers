#!/usr/bin/env python3

import sys, os
import cv2
import argh
import pickle
import square_marker_detect as markerdetect
import numpy as np
from datetime import timedelta
from shutil import copyfile
from file_methods import *
from gaze_to_world_coordinates import undistort_points_fisheye


def normalize(pos, width, height,flip_y=False):
    """
    normalize return as float
    """
    x = pos[0]
    y = pos[1]
    x /=float(width)
    y /=float(height)
    if flip_y:
        return x,1-y
    return x,y

def rectify_gaze_data(path, K, D, rect_camera_matrix):

    #if not os.path.exists(path + 'pupil_data_original'):
    #    data = load_object(path + 'pupil_data')
    #    save_object(data, path + 'pupil_data_original')
    #else:
    #    data = load_object(path + 'pupil_data_original')

    data = load_object(path + 'pupil_data')

    if not 'gaze_positions' in data:
        print("no gaze_positions", data.keys())
        return

    gazes = data['gaze_positions']
    for g in gazes:
            gaze = denormalize(g['norm_pos'],1280, 720)
            gaze = np.float32(gaze).reshape(-1,1,2)
            gaze = cv2.fisheye.undistortPoints(gaze,K,D, P=rect_camera_matrix).reshape(2)
            gaze = normalize(gaze, 1280, 720)
            g['norm_pos'] = gaze

    save_object(data, path + 'pupil_data_corrected')

def denormalize(pos, width, height, flip_y=False):
    """
    denormalize
    """
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return x,y

def marker_positions(camera_spec, videofile, outfile, path, new_camera="camera", start_time=0.0, end_time=float("inf"), visualize=False,
        output_camera=None, write = False):
    camera = pickle.load(open(camera_spec, 'rb'), encoding='bytes')
    image_resolution = camera[b'resolution']
    
    if b'rect_map' not in camera:
        camera_matrix = camera[b'camera_matrix']
        camera_distortion = camera[b'dist_coefs']
        rect_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, image_resolution, 0.0)
        rmap = cv2.initUndistortRectifyMap(
            camera_matrix, camera_distortion, None, rect_camera_matrix, image_resolution,
            cv2.CV_32FC1)
    else:
        rmap = camera[b'rect_map']
        rect_camera_matrix = camera[b'rect_camera_matrix']


    K, D, resolution, cm = camera[b'camera_matrix'], camera[b'dist_coefs'], camera[b'resolution'], rect_camera_matrix

    camera = {}
    camera['camera_matrix'] = rect_camera_matrix
    camera['dist_coefs'] = None
    camera['resolution'] = image_resolution
    save_object(camera, new_camera)

    rectify_gaze_data(path, K, D, rect_camera_matrix)

    #if new_camera is not None:
    #    pickle.dump(camera, open(new_camera, 'w'))

    video = cv2.VideoCapture(videofile)
    video.set(cv2.CAP_PROP_POS_MSEC, start_time*1000)
    frames = []
    #marker_tracker = markerdetect.MarkerTracker()
    
    prev_minute = 0.0
    marker_cache = []

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')

    #if write == True:
    #    out = cv2.VideoWriter(path + "world.mp4",fourcc, 30.0, (1280,720))
    
    while True:
        ret, oframe = video.read()
        if not ret:
            break

        frame = cv2.remap(oframe, rmap[0], rmap[1], cv2.INTER_LINEAR)

        #if write == True:
        #    out.write(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        msecs = video.get(cv2.CAP_PROP_POS_MSEC)
        time = msecs/1000.0
        if time > end_time:
            break

        markers = markerdetect.detect_markers(frame, 5, min_marker_perimeter=15)
        marker_cache.append(markers)

        frame_d = {
                'ts': time,
                'markers': markers,
                }
        frames.append(frame_d)
        
        if not visualize: continue
        markerdetect.draw_markers(frame, frame_d['markers'])

        cv2.imshow('frameBG', frame)
        cv2.waitKey(1)

    np.save(outfile, frames)

    if write == True:
        out.release()

    return marker_cache


if __name__ == '__main__':
    rootdir = sys.argv[1]

    #try:
    #    writeVideo = sys.argv[2]
    #except:
    #    writeVideo = False
            
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) + "/"
        if os.path.exists(path + "world.mp4"):
            
            #if not os.path.exists(path + "world_old.mp4") and writeVideo:
            #    os.rename(vid_path, path + "world_old.mp4")
            #    vid_path = path + "world_old.mp4"
            #elif os.path.exists(path + "world_old.mp4"):
            #    vid_path = path + "world_old.mp4" 
            #else:
            #    vid_path = path + "world.mp4"

            marker_cache = Persistent_Dict(path + 'square_marker_cache')
            marker_cache['version'] = 2
            markers = marker_positions('sdcalib.rmap.full.camera.pickle', vid_path, path + 'markers_new.npy', path, visualize = True)
            marker_cache['marker_cache'] = markers
            marker_cache['inverted_markers'] = False
            marker_cache.close()
