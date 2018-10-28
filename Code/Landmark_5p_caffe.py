# coding: utf-8
import numpy as np
import math
import cv2
import os
from os import listdir
import json
import time
import sys
from MtcnnDetector import FaceDetector
#import Enhance_image

TRAINING = 0
TESTING = 1

CPU = 0
GPU = 1

CASIA = 0
REPLAYATTACK = 1
OULU = 2

def list2colmatrix(pts_list):
    """
        convert list to column matrix
    Parameters:
    ----------
        pts_list:
            input list
    Retures:
    -------
        colMat:

    """
    assert len(pts_list) > 0
    colMat = []
    for i in range(len(pts_list)):
        colMat.append(pts_list[i][0])
        colMat.append(pts_list[i][1])
    colMat = np.matrix(colMat).transpose()
    return colMat

def find_tfrom_between_shapes(from_shape, to_shape):
    """
        find transform between shapes
    Parameters:
    ----------
        from_shape:
        to_shape:
    Retures:
    -------
        tran_m:
        tran_b:
    """
    assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

    sigma_from = 0.0
    sigma_to = 0.0
    cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

    # compute the mean and cov
    from_shape_points = from_shape.reshape(from_shape.shape[0]/2, 2)
    to_shape_points = to_shape.reshape(to_shape.shape[0]/2, 2)
    mean_from = from_shape_points.mean(axis=0)
    mean_to = to_shape_points.mean(axis=0)

    for i in range(from_shape_points.shape[0]):
        temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
        sigma_from += temp_dis * temp_dis
        temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
        sigma_to += temp_dis * temp_dis
        cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

    sigma_from = sigma_from / to_shape_points.shape[0]
    sigma_to = sigma_to / to_shape_points.shape[0]
    cov = cov / to_shape_points.shape[0]

    # compute the affine matrix
    s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    u, d, vt = np.linalg.svd(cov)

    if np.linalg.det(cov) < 0:
        if d[1] < d[0]:
            s[1, 1] = -1
        else:
            s[0, 0] = -1
    r = u * s * vt
    c = 1.0
    if sigma_from != 0:
        c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

    tran_b = mean_to.transpose() - c * r * mean_from.transpose()
    tran_m = c * r

    return tran_m, tran_b
def extract_image_chips(img, points, desired_size=256, padding=0):
    """
        crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256
        padding: default 0
    Retures:
    -------
        crop_imgs: list, n
            cropped and aligned faces
    """
    crop_imgs = []
    for p in points:
        shape  =[]
        for k in range(len(p)/2):
            shape.append(p[2*k])
            shape.append(p[2*k+1])

        if padding > 0:
            padding = padding
        else:
            padding = 0
        # average positions of face points
        mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
        mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]
        #mean_face_shape_x = [0.3405, 0.6751, 0.5009, 0.3718, 0.6452]
        #mean_face_shape_y = [0.3203, 0.3203, 0.5059, 0.6942, 0.6962]
        from_points = []
        to_points = []

        for i in range(len(shape)/2):
            x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
            y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
            to_points.append([x, y])
            from_points.append([shape[2*i], shape[2*i+1]])

        # convert the points to Mat
        from_mat = list2colmatrix(from_points)
        to_mat = list2colmatrix(to_points)

        # compute the similar transfrom
        tran_m, tran_b = find_tfrom_between_shapes(from_mat, to_mat)

        probe_vec = np.matrix([1.0, 0.0]).transpose()
        probe_vec = tran_m * probe_vec

        scale = np.linalg.norm(probe_vec)
        angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

        from_center = [(shape[0]+shape[2])/2.0, (shape[1]+shape[3])/2.0]
        to_center = [0, 0]
        to_center[1] = desired_size * 0.4
        to_center[0] = desired_size * 0.5

        ex = to_center[0] - from_center[0]
        ey = to_center[1] - from_center[1]

        rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1*angle, scale)
        rot_mat[0][2] += ex
        rot_mat[1][2] += ey

        chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
        crop_imgs.append(chips)

    return crop_imgs

def Genface_chip(img, detector, devConfig):
    detect_time_start = time.time()
    total_boxes, face_points, numbox = detector.detectface(img) # face detection
    detect_time_end = time.time()
    face_chip = []
    detect_time = detect_time_end - detect_time_start
    print "detect face cost time is " + str(detect_time)+ 's'

    if numbox >= 1:
        # for data argumentation during training, we get the 144*144 faces and then randomly crop to 128*128
        # padding = 0.37, target_img_size = 144
        # for testing or no data augmentationi training, padding = 0.273, target_img_size = 128
        padding = 0.06
        target_img_size = 224
        points = []
        for i in range(numbox):
	    p = []
            for j in range(5):
                p.append(face_points[j,i])
                p.append(face_points[j+5, i])
	    points.append(p)
        print points
        facechip_time_start = time.time()
        face_chip = extract_image_chips(img, points, target_img_size, padding)
        facechip_time_end = time.time()
	print "face chip cost time is " + str(facechip_time_end - facechip_time_start) + 's'
    return face_chip,detect_time

def ProcessLandmark_5p(PROCESSFLAG,DataBaseOpt,mtcnnParam,devConfig):
    true_img_dir_CASIA = ['1.avi', '2.avi', 'HR_1.avi', 'HR_4.avi']
    if DataBaseOpt == REPLAYATTACK:
        Head_train = "../../../Data/ReplayAttack/Train/"
        Head_S2_train = Head_train + "Train_frames/"  # from unzip the train/test.rar
        Head_T_train = Head_train + "Train_aligned_REPLAY/"

        Head_test = "../../../Data/ReplayAttack/Test/"
        Head_S2_test = Head_test + "Test_frames/"  # from unzip the train/test.rar
        Head_T_test = Head_test + "Test_aligned_REPLAY/"
        count_content_Path = "../../Data/Caffe_Train/Val/test_count.json"
    elif DataBaseOpt == CASIA:
        Head_train = "../../Data/CBSR-Antispoofing/Train/"
        Head_S2_train = Head_train + "Train_frames_ALL/"  # from unzip the train/test.rar
        Head_T_train = Head_train + "Train_aligned_ALL/"

        Head_test = "../../Data/CBSR-Antispoofing/Test/"
        Head_S2_test = Head_test + "Test_frames_ALL/"  # from unzip the train/test.rar
        Head_T_test = Head_test + "Test_aligned_ALL/"
        count_content_Path = "../../../Data/CASIA/Test/test_count_5p.json"
    elif DataBaseOpt == OULU:
        Head_train = "../../Data/Caffe_Train/Train/"
        Head_S2_train = Head_train + "Dev_frames/"  # from unzip the train/test.rar
        Head_T_train = Head_train + "Dev_aligned/"

        Head_test = "../../Data/Caffe_Train/Val/"
        Head_S2_test = Head_test + "Test_frames_10/"  # from unzip the train/test.rar
        Head_T_test = Head_test + "Test_aligned_10/"
        count_content_Path = "../../Data/Caffe_Train/Val/test_count_10.json"
    else:
        print "DataBase Unsupport!!"
        return

    ratio = 2;
    threshold = [0.6, 0.6, 0.7]

    detector = FaceDetector(minsize=mtcnnParam[0], factor = mtcnnParam[1],
                            gpuid=1, fastresize=True,threshold = mtcnnParam[2],
                            nms_thresh = mtcnnParam[3])

    if PROCESSFLAG == TRAINING:
        Head_S2 = Head_S2_train
        Head_T = Head_T_train
    else:
        Head_S2 = Head_S2_test
        Head_T = Head_T_test

    filedir_index = []

    sum_count = 0
    fail_count = 0
    detect_time_sum = 0.0
    os.makedirs(Head_T + '/' + 'Attack')
    os.makedirs(Head_T + '/' + 'Real')

    for d in listdir(Head_S2):  # d is file folder
        #if not os.path.exists(Head_T + d):  # not exist
        #else:
            #pass

        #for f in listdir(Head_S2 + d):
            #print(f)
            #if not os.path.exists(Head_T + d + "/" + f):
                #if f in true_img_dir_CASIA:
                    #os.makedirs(Head_T + '/'+ 'Real' + '/' + d + "/" + f)
                #else:
                    #os.makedirs(Head_T + '/'+ 'Attack' + '/' + d + "/" + f)
            # for g in listdir(Head_S2 + d + "/" + f):
            #     if not os.path.exists(Head_T + d + "/" + f + "/" + g):
            #         os.makedirs(Head_T + '/'+ 'Attack' + d + "/" + f + "/" + g)

        video_frame_count_attack = 0
        for f in listdir(Head_S2 + d):
            for f2f in listdir(Head_S2 + d + "/" + f):
                video_frame_count_attack = 0
                print("read" + Head_S2 + d + "/" + f + "/" + f2f + '\n')

                img = cv2.imread(Head_S2 + d + "/" + f + "/" + f2f)

                if img is None:
                    print "Read Image Failed!"
                    return
                img_size = (img.shape[1] / ratio, img.shape[0] / ratio)
                img_new = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)

                (face_chips, detect_time) = Genface_chip(img_new, detector, devConfig)
                detect_time_sum += detect_time
                if face_chips:
                    for i, chip in enumerate(face_chips):
		                #chip = Enhance_image.Image_Enhance(chip,1.35,35)
                        if f in true_img_dir_CASIA:
                            cv2.imwrite(Head_T  + '/'+ 'Real' + '/' + d+f+f2f, chip)
                            print('write' + Head_T  + '/'+ 'Real' + '/' + d+f+f2f)
                        else:
                            cv2.imwrite(Head_T  + '/'+ 'Attack' + '/' + d+f+f2f, chip)
                            print('write' + Head_T  + '/'+ 'Attack' + '/' + d + f + f2f)
                        #path = Head_T + d + "/" + f

                    
                    video_frame_count_attack += 1
                    sum_count += 1
                else:
                    fail_count += 1
                    print (Head_S2 + d + "/" + f  + "face detection failed\n")
                    pass
                #if PROCESSFLAG == TESTING:
            if sum_count - video_frame_count_attack == sum_count:
                e_attack = sum_count - video_frame_count_attack
            else:
                e_attack = sum_count - video_frame_count_attack + 1
            #filedir_index.append([f, f2f, e_attack, sum_count])
            #print [f, f2f, e_attack, sum_count]
    #average_time = float(detect_time_sum)/(float(sum_count)+float(fail_count))
#    if PROCESSFLAG == TESTING:
#        print(filedir_index)
#        with open(count_content_Path, 'w') as asdf:
#            json.dump(filedir_index, asdf, indent=4)
#            #return filedir_index
    return
    print("Processing ProcessLandmark Success!")

mtcnnParam1 = [40, 0.709, [0.6, 0.7, 0.7], [0.5, 0.7, 0.7, 0.7]]
ProcessLandmark_5p(TRAINING, CASIA, mtcnnParam1, GPU)
ProcessLandmark_5p(TESTING, CASIA, mtcnnParam1, GPU)

