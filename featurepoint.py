import cv2
from matplotlib import pyplot as plt
import numpy as np
from prospective import *
import pygame

## 1修改過(設門檻)
## 2修改過(sift、bf等，希望再程式開始就初始化，不用一直初始化)
def find_keypoint(sift, bf, background, target, average_frame_kp): 
    """
    find two pic keypoint 對應
    and compute inlier four index
    """
    kp_1, descriptor_1 = sift.detectAndCompute(background, None)
    kp_2, descriptor_2 = sift.detectAndCompute(target, None)
    # print('背景圖描述子大小 {:}'.format(descriptor_1.shape))
    # print('目標圖描述子大小 {:}'.format(descriptor_2.shape))
    ### 這行是指，左圖關鍵點在右圖關鍵點找出 k 相似的
    matches = bf.knnMatch(descriptor_2, descriptor_1, k=2)  ### 要在 background 裡找像 target 的，因此 background 要放後面
    ### Apply ratio test，see document [ https://docs.opencv.org/4.4.0/dc/dc3/tutorial_py_matcher.html ]
    good = []
    for i,j in matches:   ### matches return k objects and object have 4 thing (distance, trainIdx, queryIdx, imgIdx) 
        if i.distance < 0.7 * j.distance:  ### 把第1個跟第2個相似點差距大的抓出來( 都是 descriptor_2)
            good.append([i])

    if len(good) < 77:
        average_frame_kp.append(len(good))
        return None, None, None, sum(average_frame_kp)
    else:
        # 看api找出全部特徵點對應如何    
        # combine_pic = cv2.drawMatchesKnn(target, kp_2, background, kp_1, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.figure(figsize = (12, 8))
        # plt.imshow(cv2.cvtColor(combine_pic, cv2.COLOR_BGR2RGB))
        # plt.show()

        average_frame_kp.append(len(good))
        # 把兩張圖的 keypoint 取出
        background_goodkeypoint = []
        target_goodkeypoint = []
        four_point_index = []
        for i in good:
            background_goodkeypoint.append(kp_1[i[0].trainIdx].pt)
            target_goodkeypoint.append(kp_2[i[0].queryIdx].pt)
        background_goodkeypoint = np.array(background_goodkeypoint).astype('uint16')
        target_goodkeypoint = np.array(target_goodkeypoint).astype('uint16')


        background_mean = np.mean(background_goodkeypoint,axis = 0)
        background_std = np.std(background_goodkeypoint,axis = 0)
        background_x_mean, background_y_mean = background_mean[:]
        background_x_std, background_y_std = background_std[:]

        # 2 or 3
        std_distance = 2
        # x 對應的 index 是由小排到大，所以可以直接這樣用
        x_min_index = 0
        x_max_index = (len(background_goodkeypoint[:,0]) - 1)
        while ((background_goodkeypoint[x_min_index,0] < (background_x_mean - (std_distance * background_x_std))) or 
            (background_goodkeypoint[x_min_index,0] > (background_x_mean + (std_distance * background_x_std)))):
            x_min_index += 1
        # print(x_min_index)
            
        while ((background_goodkeypoint[x_max_index,0] < (background_x_mean - (std_distance * background_x_std))) or 
            (background_goodkeypoint[x_max_index,0] > (background_x_mean + (std_distance * background_x_std)))):
            x_max_index -= 1
        # print(x_max_index)

        ## y 就沒有排序好，要改變寫法
        y_index_sorted = np.argsort(background_goodkeypoint[:,1])  # 返回數字由小排到大對應的 index
        first_index = 0
        tail_index = -1
        y_min_index = y_index_sorted[first_index]
        y_max_index = y_index_sorted[tail_index]

        while ((background_goodkeypoint[y_min_index,1] < (background_y_mean - (std_distance * background_y_std))) or 
            ((background_goodkeypoint[y_min_index,1]) > (background_y_mean + (std_distance * background_y_std))) or 
            (y_min_index <= x_min_index) or (y_min_index >= x_max_index)):
            first_index += 1
            y_min_index = y_index_sorted[first_index]

        while ((background_goodkeypoint[y_max_index,1] < (background_y_mean - (std_distance * background_y_std))) or 
            ((background_goodkeypoint[y_max_index,1]) > (background_y_mean + (std_distance * background_y_std))) or 
            (y_max_index <= x_min_index) or (y_max_index >= x_max_index)):
            tail_index -= 1   
            y_max_index = y_index_sorted[tail_index]

        four_point_index = [x_min_index, x_max_index, y_min_index, y_max_index]
        return four_point_index, background_goodkeypoint, target_goodkeypoint, sum(average_frame_kp)

def look_two_pic_keypoint_correspond(frame, name2, four_point_index, background_goodkeypoint, target_goodkeypoint):
    """
    Look at two pic keypoint in where, is it match ?
    """

    img_2 = cv2.imread(name2)
    count = 1
    for i in four_point_index:
        x1,y1 = background_goodkeypoint[i]
        cv2.circle(frame, (x1,y1), 3, (0,255,0), 5)
        cv2.putText(frame,  str(count) + ':'+ str((x1, y1)), (x1, y1), 2, cv2.FONT_HERSHEY_PLAIN, (255,0,0), 2, cv2.LINE_AA)
        x2,y2 = target_goodkeypoint[i]
        # print('第{:}組對應，{:} --> {:}, {:} --> {:}'.format(count,x2,x1,y2,y1))
        cv2.circle(img_2, (x2,y2), 3, (0,255,0), 5)
        cv2.putText(img_2, str(count) + ':' + str((x2, y2)), (x2, y2), 2, cv2.FONT_HERSHEY_PLAIN, (255,0,0), 2, cv2.LINE_AA)    
        count += 1
    
    # plt.figure(figsize = (12, 8))
    # plt.subplot(1,2,1)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.subplot(1,2,2)
    # plt.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imshow('1', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow('2', cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

def look_feature_point_wrap_pic(name1, name2, four_point_index, target_goodkeypoint, background_goodkeypoint):
    """
    according to 'look_two_pic_keypoint_correspond' function find four point
    we use this point warp two pic
    """
    ## 貼圖片上去
    background_pic = cv2.imread(name1)
    target_pic = cv2.imread(name2)

    target_point = []
    background_point = []
    for i in four_point_index:
        target_point.append([target_goodkeypoint[i,0],target_goodkeypoint[i,1]])
        background_point.append([background_goodkeypoint[i,0], background_goodkeypoint[i,1]])

    H = find_H_matrix(background_point, target_point) 
    match_matrix = find_perspective_matrix(H)
    test_img = warp_pic(target_pic, background_pic, match_matrix)

    plt.figure(figsize = (12, 8))
    test_img = test_img.astype('uint8') 
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.show()

def get_feature_edge(background_pic, target_pic, match_matrix): # 更改第一項
    """
    need input background, target_pic, match_matrix
    get feature pic edge in background,this function return point
    """
    # background_pic = cv2.imread(background_pic_name)
    ## 取得大圖中特證照片的四個角落(用母圖四個角落找)
    target_point = get_pic_edge_point(target_pic)
    # print(target_point)
    feature_pic_edge = []
    ### 原本找座標是用背景圖乘上轉移矩陣去求得在目標圖上的座標，我們現在要用目標圖上的角落找到在背景圖上的特徵圖角落
    match_matrix_inv = np.linalg.inv(match_matrix)
    for i in target_point:
        ans = match_matrix_inv.dot([i[0],i[1],1])
        ans = np.around(ans / ans[-1]).astype('int16')
        if ans[0] < 0:
            ans[0] = 0
        elif ans[0] > background_pic.shape[1]:
            ans[0] = background_pic.shape[1]
        if ans[1] < 0:
            ans[1] = 0
        elif ans[1] > background_pic.shape[0]:
            ans[1] = background_pic.shape[0]
        feature_pic_edge.append([ans[0],ans[1]])
    # print(feature_pic_edge)

    ## 看一下點是不是正確
    # count = 1
    # for j in range(len(feature_pic_edge)):
    #     x1, y1 = feature_pic_edge[j]
    #     cv2.circle(background_pic, (x1,y1), 3, (0,255,0), 5)
    #     cv2.putText(background_pic,  str(count) + ':'+ str((x1, y1)), (x1, y1), 2, cv2.FONT_HERSHEY_PLAIN, (255,0,0), 2, cv2.LINE_AA)
    #     count += 1
    # plt.figure(figsize = (12, 8))
    # plt.imshow(cv2.cvtColor(background_pic, cv2.COLOR_BGR2RGB))
    # plt.show()
    return feature_pic_edge
    
## 最後一部，貼上最終影片
def final_work(video_1, video_2, video_3):  
    count = 3
    x = video_1.shape[0]
    y = video_1.shape[1]
    new_x = int(x / 2)
    new_y = int(y / 2)

    img = np.ones((x,y,3), dtype = 'uint8')
    cv2.putText(img, str(count), (new_x, new_y), 5, cv2.FONT_HERSHEY_TRIPLEX, (0,0,255), 2, cv2.LINE_AA)   
    cv2.imshow('',img)
    cv2.waitKey(1000)

    count -= 1
    img = np.ones((x,y,3), dtype = 'uint8')
    cv2.putText(img, str(count), (new_x, new_y), 5, cv2.FONT_HERSHEY_TRIPLEX, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('',img)
    cv2.waitKey(1000)

    count -= 1
    img = np.ones((x,y,3), dtype = 'uint8')
    cv2.putText(img, str(count), (new_x, new_y), 5, cv2.FONT_HERSHEY_TRIPLEX, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('',img)
    cv2.waitKey(1000)

    temp_count = 0
    while True:
        # print(temp_count)
        video_1_pic = video_1[temp_count,:,:,:]
        cv2.imshow('', video_1_pic)
        temp_count += 1
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(1)
        if cv2.waitKey(30) & 0xFF == ord('q') or temp_count == video_1.shape[0]:
            break

    temp_count = 0
    while True:
        # print(temp_count)
        video_2_pic = video_2[temp_count,:,:,:]
        cv2.imshow('', video_2_pic)
        temp_count += 1
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(1)
        if cv2.waitKey(30) & 0xFF == ord('q') or temp_count == video_2.shape[0]:
            break

    temp_count = 0
    while True:
        # print(temp_count)
        video_3_pic = video_3[temp_count,:,:,:]
        cv2.imshow('', video_3_pic)
        temp_count += 1
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(1)
        if cv2.waitKey(30) & 0xFF == ord('q') or temp_count == video_3.shape[0]:
            break

    cv2.destroyAllWindows()

    text = 'How many godofasia3:0 appear ^^ ?'
    img = np.ones((500,1500,3), dtype = 'uint8')
    cv2.putText(img, str(text), (100, 250), 1, cv2.FONT_HERSHEY_TRIPLEX, (0,144,255), 2, cv2.LINE_AA)
    cv2.imshow('',img)
    cv2.waitKey(0)