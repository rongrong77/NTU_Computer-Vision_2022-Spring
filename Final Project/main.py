import os
import sys
import math
import copy
from PIL import Image
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from utils import *
np.set_printoptions(suppress=True)

def write_config(opt):
    with open(f'./result/TRAIN_ST_{opt.exp_no}/config.txt', 'w') as f:
        f.write(f'''
                intensity_ratio: {opt.intensity_ratio}
                OpenIter: {opt.OpenIter}
                FiltDist: {opt.FiltDist}
                AssistTh: {opt.AssistTh}
                maxCorners: {opt.maxCorners}
                qualityLevel: {opt.qualityLevel}
                minDistance: {opt.minDistance}
                ''')

# O F A 2000 0.01 10
# python3 main.py --mode train --OpenIter 1 --FiltDist 15 --AssistTh 0.2 --> 0.43750 0.43541 0.42780
# python3 main.py --mode train --OpenIter 1 --FiltDist 20 --AssistTh 0.2 --> 0.43371 0.43448 0.42649
# python3 main.py --mode train --OpenIter 1 --FiltDist 25 --AssistTh 0.2 --> 0.43401 0.43501 0.42575
# python3 main.py --mode train --OpenIter 1 --FiltDist 21 --AssistTh 0.2 --> 0.43324 0.43413 0.42530
# python3 main.py --mode train --OpenIter 1 --FiltDist 22 --AssistTh 0.2 --> 0.43485 0.43538 0.42604
# python3 main.py --mode train --OpenIter 1 --FiltDist 19 --AssistTh 0.2 --> 0.43358 0.43373 0.42752
# python3 main.py --mode train --OpenIter 3 --FiltDist 19 --AssistTh 0.2 --> 0.45393 0.44273 0.43700
# 1 19 0.2 m q m
# python3 main.py --mode train --maxCorners 2000 --qualityLevel 0.05 --minDistance 10 --> 0.43516 0.43500 0.42937
# python3 main.py --mode train --maxCorners 2000 --qualityLevel 0.1 --minDistance 10 --> 0.42982 0.42760 0.40546
# python3 main.py --mode train --maxCorners 2000 --qualityLevel 0.01 --minDistance 20 --> 0.44078 0.42992 0.42385
# python3 main.py --mode train --maxCorners 2000 --qualityLevel 0.1 --minDistance 20 --> 0.43129 0.42590 0.40353 * 0.42181 v3
# I 1 19 0.2 2000 0.1 20 
# python3 main.py --intensity_ratio 0.1 --> 0.45478 0.41594 0.40978
# python3 main.py --intensity_ratio 0.05 --> 0.62965 0.43389 0.42220
# python3 main.py --intensity_ratio 0.15 --> 0.43263 0.41814 0.41728
# python3 main.py --intensity_ratio 0.11 --> 0.43073 0.41534 0.41325
# python3 main.py --intensity_ratio 0.12 --> 0.42807 0.41750 0.41692
# 10 python3 main.py --intensity_ratio 0.13 --> 0.42941 0.41524 0.41416 * 0.40042 v11
# python3 main.py --intensity_ratio 0.14 --> 0.43022 0.41593 0.41438
# python3 main.py --intensity_ratio 0.125 --> 0.42344 0.41819 0.41095 * 0.39941 v2

# No adaptive mask
# 13 python3 main.py --OpenIter 2 --maxCorners 400 --> 0.44548 0.43472 0.41744
# 14 python3 main.py --minDistance 15 --> 0.43278 0.43024 0.41438
# 15 python3 main.py --minDistance 25 --> 0.43299 0.43000 0.41152
# 16 python3 main.py --minDistance 19 --> 0.43120 0.42862 0.40477
# 17 python3 main.py --minDistance 18 --> 0.43361 0.42899 0.40917
# 18 python3 main.py --minDistance 17 --> 0.43220 0.42929 0.41335
# 19 python3 main.py --minDistance 21 --> 0.43277 0.42617 0.40578
# 20 python3 main.py --minDistance 18 --> 0.43004 0.42909 0.40613
# 21 python3 main.py --AssistTh 0.3 --> 0.43129 0.42590 0.40353   X

# adaptive mask
# 22 python3 main.py --intensity_ratio 0.121 --> 0.42939 0.41484 0.40928 
# 23 python3 main.py --intensity_ratio 0.122 --> 0.42997 0.41963 0.40473 * 0.40227 v4
# 24 python3 main.py --intensity_ratio 0.123 --> 0.42863 0.41871 0.41026 
# 25 python3 main.py --intensity_ratio 0.124 --> 0.42317 0.42194 0.41127 * 0.39902 v5
# 26 python3 main.py --minDistance 19 --> 0.42661 0.41733 0.41293
# 27 python3 main.py --minDistance 18 --> 0.42551 0.41701 0.41424

# 28 python3 main.py --intensity_ratio 0.124 --minDistance 19 --> 0.42661 0.41733 0.41293
# 29 python3 main.py --intensity_ratio 0.124 --minDistance 18 --> 0.42551 0.41701 0.41424 * 0.40055 v6
# 30 python3 main.py --intensity_ratio 0.124 --minDistance 21 --> 0.42600 0.42144 0.41074
# 31 python3 main.py --intensity_ratio 0.124 --minDistance 22 --> 0.42385 0.41765 0.40615 * 0.40416 v7

# 32 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.05 --> 0.42655 0.41979 0.41029
# 33 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.09 --> 0.42228 0.42188 0.41158
# 34 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.08 --> 0.42241 0.42179 0.41158 * 0.39968 v13
# 35 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.01 --> 0.42651 0.41933 0.41144
# 36 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.12 --> 0.42454 0.42159 0.41086 * 0.39871 v8
# 37 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.13 --> 0.42463 0.42158 0.41084 * 0.39871 v10
# 38 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.15 --> 0.42367 0.41856 0.41245 * 0.40052 v12
# 39 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.20 --> 0.47305 0.42486 0.40970 * 0.40606 v9
# 40 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.14 --> 0.42483 0.42142 0.41084
# 41 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.11 --> 0.42320 0.42194 0.41132
# 42 python3 main.py --intensity_ratio 0.124 --qualityLevel 0.115 --> 0.42317 0.42194 0.41132

# 0.124 1 19 0.2 mC 0.12 20
# 43 python3 main.py --maxCorners 200 --> 0.42265 0.42079 0.41087
# 44 python3 main.py --maxCorners 300 --> 0.42454 0.42159 0.41086
# 45 python3 main.py --maxCorners 400 --> 0.41454 0.42159 0.41086
# 46 python3 main.py --maxCorners 1000 --> 0.41454 0.42159 0.41086

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--exp_no', type=int, default=0)
parser.add_argument('--intensity_ratio', type=float, default=0.124)
parser.add_argument('--OpenIter', type=int, default=1)
parser.add_argument('--FiltDist', type=int, default=19)
parser.add_argument('--AssistTh', type=float, default=0.2)
parser.add_argument('--maxCorners', type=int, default=2000)
parser.add_argument('--qualityLevel', type=float, default=0.12)
parser.add_argument('--minDistance', type=int, default=20)
opt = parser.parse_args()
print(opt)

# Training material
train_camera_dict = pickle_load('./pkl/camera_dict.pkl')
# Testing material
test_camera_dict = pickle_load('./pkl/camera_dict_test.pkl')

if not os.path.exists('./result'):
    os.mkdir('./result')

if opt.mode == 'train':
    os.mkdir(f'./result/TRAIN_ST_{opt.exp_no}')
    os.mkdir(f'./result/TRAIN_ST_{opt.exp_no}/solution')
    os.mkdir(f'./result/TRAIN_ST_{opt.exp_no}/solution/seq1')
    os.mkdir(f'./result/TRAIN_ST_{opt.exp_no}/solution/seq2')
    os.mkdir(f'./result/TRAIN_ST_{opt.exp_no}/solution/seq3')
    write_config(opt)
    re = {}
    for seq in ['seq1', 'seq2', 'seq3']:
        re[seq] = {}
        # seq_time = txt_load(f'./ITRI_DLC2/{seq}/localization_timestamp.txt')
        seq_time = os.listdir(f'./ITRI_dataset/{seq}/dataset/')
        for time in tqdm(seq_time):
            img = np.array(Image.open(f'./ITRI_dataset/{seq}/dataset/{time}/raw_image.jpg'))
            camera = train_camera_dict[seq][time].split('_')[-2]
            _, _, _, corner_point = preprocess(img, camera, intensity_ratio=opt.intensity_ratio, 
                                               open_iter=opt.OpenIter, method='shi_tomasi', 
                                               maxCorners=opt.maxCorners, 
                                               qualityLevel=opt.qualityLevel, 
                                               minDistance=opt.minDistance)
            world_point_camera = pixel_to_world(corner_point, camera)
            # base_point_cloud = world_coor_transform(world_point_camera, camera+'2base')
            base_point_cloud = world_coor_transform_high_eff(world_point_camera, camera+'2base')
            land_base_point_cloud = correct_to_land(base_point_cloud, filt_dist=opt.FiltDist)
            re[seq][time] = land_base_point_cloud
    aug_re = {}
    for seq in ['seq1', 'seq2', 'seq3']:
        aug_re[seq] = {}
        seq_time = os.listdir(f'./ITRI_dataset/{seq}/dataset/')
        seq_camera = []
        for time in seq_time:
            seq_camera.append(  train_camera_dict[seq][time].split('_')[-2]  )
        seq_camera = np.array(seq_camera)
        for time in tqdm(seq_time):
            camera = train_camera_dict[seq][time].split('_')[-2]
            assist = get_nearst_assist_time(time, camera, seq_time, seq_camera, opt.AssistTh)
            temp = re[seq][time].copy()
            for a in assist:
                temp = np.concatenate([temp, re[seq][a]], axis=0)
            aug_re[seq][time] = temp.copy()

    pickle_save(re, f'./result/TRAIN_ST_{opt.exp_no}/base_point_cloud.pkl')
    pickle_save(aug_re, f'./result/TRAIN_ST_{opt.exp_no}/aug_point_cloud.pkl')
    

if opt.mode == 'test':
    os.mkdir(f'./result/TEST_ST_{opt.exp_no}')
    os.mkdir(f'./result/TEST_ST_{opt.exp_no}/solution')
    os.mkdir(f'./result/TEST_ST_{opt.exp_no}/solution/test1')
    os.mkdir(f'./result/TEST_ST_{opt.exp_no}/solution/test2')
    re = {}
    for test in ['test1', 'test2']:
        re[test] = {}
        test_time = os.listdir(f'./ITRI_DLC/{test}/dataset/')
        for time in tqdm(test_time):
            img = np.array(Image.open(f'./ITRI_DLC/{test}/dataset/{time}/raw_image.jpg'))
            camera = test_camera_dict[test][time].split('_')[-2]
            _, _, _, corner_point = preprocess(img, camera, intensity_ratio=opt.intensity_ratio, 
                                               open_iter=opt.OpenIter, method='shi_tomasi', 
                                               maxCorners=opt.maxCorners, 
                                               qualityLevel=opt.qualityLevel, 
                                               minDistance=opt.minDistance)
            world_point_camera = pixel_to_world(corner_point, camera)
            # base_point_cloud = world_coor_transform(world_point_camera, camera+'2base')
            base_point_cloud = world_coor_transform_high_eff(world_point_camera, camera+'2base')
            land_base_point_cloud = correct_to_land(base_point_cloud, filt_dist=opt.FiltDist)
            re[test][time] = land_base_point_cloud
    aug_re = {}
    for test in ['test1', 'test2']:
        aug_re[test] = {}
        test_time = os.listdir(f'./ITRI_DLC/{test}/dataset/')
        test_camera = []
        for time in test_time:
            test_camera.append(  test_camera_dict[test][time].split('_')[-2]  )
        test_camera = np.array(test_camera)
        for time in tqdm(test_time):
            camera = test_camera_dict[test][time].split('_')[-2]
            assist = get_nearst_assist_time(time, camera, test_time, test_camera, opt.AssistTh)
            temp = re[test][time].copy()
            for a in assist:
                temp = np.concatenate([temp, re[test][a]], axis=0)
            aug_re[test][time] = temp.copy()

    pickle_save(re, f'./result/TEST_ST_{opt.exp_no}/base_point_cloud.pkl')
    pickle_save(aug_re, f'./result/TEST_ST_{opt.exp_no}/aug_point_cloud.pkl')