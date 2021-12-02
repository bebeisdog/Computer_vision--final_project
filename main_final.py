from featurepoint import *
import cv2 
from prospective import *
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time
import copy
import pygame

def image_preprocess(path):
    """
    read pic is Gray and do image preprocess: GaussianBlur
    """
    pic = cv2.imread(path,0)
    guass_pic = cv2.GaussianBlur(pic, (3, 3), 0)
    return guass_pic

# 於弦相似度，算矩陣是否相似
@jit(nopython=True)
def cosine_similarity(matrix_a, matrix_b):
    top = np.sum(np.multiply(matrix_a, matrix_b))
    bottom = np.sqrt(np.sum((matrix_a**2)))*(np.sqrt(np.sum((matrix_b**2))))
    return (top/bottom)


def find_pic_1(target, target_name, video, video_pic):
    global fucking_count
    count = 0   
    times = 1
    average_frame_kp = []
    while True:
        if times >= 60: # 定次 reset  
            print('鏡頭中一直沒有目標照片，重製 temp ')
            times = 1
            average_frame_kp = []
            average_sum = 0
            time.sleep(3)
            
        ## 找蒂法特徵圖，貼上統神端火鍋影片
        _, frame = cap.read()
        cv2.imshow('origin_frame', frame)
        cv2.waitKey(1)
        
        four_point_index, background_goodkeypoint, target_goodkeypoint, average_sum = find_keypoint(sift, bf, frame, target, average_frame_kp)
        print((average_sum / times))
        if (average_sum / times) <= 90:
            times += 1
            continue
        else:
            ## 查看找到兩張圖的四個點如何
            try:
                look_two_pic_keypoint_correspond(frame, target_name, four_point_index, background_goodkeypoint, target_goodkeypoint)
            except TypeError:
                print('目前影像擷取屏幕中，找不到特徵圖片')
                continue
            ## 根據這找到的點貼上照片試試
            # look_feature_point_wrap_pic(name1, name2, four_point_index, target_goodkeypoint, background_goodkeypoint)
            times += 1
            target_point = []
            background_point = []
            for i in four_point_index:
                target_point.append([target_goodkeypoint[i,0],target_goodkeypoint[i,1]])
                background_point.append([background_goodkeypoint[i,0], background_goodkeypoint[i,1]])
            
            ## feature_pic_H 找特徵圖背景圖的轉換矩陣
            feature_pic_H = find_H_matrix(background_point, target_point)  
            feature_pic_match_matrix = find_perspective_matrix(feature_pic_H)
            feature_pic_edge = get_feature_edge(frame, target, feature_pic_match_matrix)

            ## new 方法，先將影片存成 npy 再進行使用(影片用小更快)
            video_point = get_pic_edge_point(video_pic)
            ## final_pic_H 最終要貼上的圖跟背景圖的轉換矩陣
            final_pic_H = find_H_matrix(feature_pic_edge, video_point)  
            final_pic_match_matrix = find_perspective_matrix(final_pic_H)
            if count >= video.shape[0]:
                count = 0
                fucking_count += 1
                break
            video_pic = video[count,:,:,:]

            ## warp new pic 上去          
            _, new_frame = cap.read()
            try:
                final = warp_pic(video_pic, new_frame, final_pic_match_matrix)
            except  ZeroDivisionError :
                continue
            print('showing.....')
            cv2.imshow('final', final)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(1)
            count += 3
            cv2.waitKey(1)
                
            times = 1
            average_frame_kp = []
            average_sum = 0
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ### init 區域
    fucking_count = 0
    cap = cv2.VideoCapture(0) ## 1 --> OBS vir camera、 0 --> real camera
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    pygame.mixer.init()
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.load('test.mp3')

    ## 1~3 Video， pre-load to npy 
    video_1 = np.load('preprocess_video_1.npy')
    video_1_pic = video_1[0,:,:,:]
    video_2 = np.load('preprocess_video_2.npy')
    video_2_pic = video_2[0,:,:,:]
    video_3 = np.load('preprocess_video_3.npy')
    video_3_pic = video_3[0,:,:,:]

    ## target is our Feature Pic，we wrap video to this pic
    target_1_name = 'woman_target2.jpeg'
    target_1 = cv2.imread(target_1_name, 0)
    target_2_name = 'godasia2.jpg'
    target_2 = cv2.imread(target_2_name, 0)
    target_3_name = 'meme22.jpg'
    target_3 = cv2.imread(target_3_name, 0)


    ### First find Pic
    find_pic_1(target_1, target_1_name, video_1, video_1_pic)
    pygame.mixer.music.pause( )
    ### Second find Pic
    find_pic_1(target_2, target_2_name, video_2, video_2_pic)
    pygame.mixer.music.pause( )
    ### Third find Pic
    find_pic_1(target_3, target_3_name, video_3, video_3_pic)
    pygame.mixer.music.pause( )

    if fucking_count == 3:
        final_work(video_1, video_2, video_3)