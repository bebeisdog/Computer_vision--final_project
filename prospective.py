from numba import jit
import numpy as np
from matplotlib import pyplot as plt
import cv2


def find_H_matrix(Pre_conversion_point, conversion_complete_point):
    """
    Pre_conversion_point 為要轉換的點
    conversion_complete_point 為轉換過去的點
    so you will get Pre_conversion_point ==> conversion_complete_point, 
    Big A matrix not H get H in find_perspective_matrix
    """

    init = np.zeros((8,9))
    x = 0
    count = 0
    for i in range(init.shape[0]):
        if (i%2) == 0:
            init[i][0:2] = Pre_conversion_point[x][0:2]
            init[i][2] = 1
            init[i][-3:-1] = Pre_conversion_point[x][0:2]
            init[i][-1] = 1 
            init[i][-3:] = init[i][-3:] * (conversion_complete_point[x][count]) * -1
            count += 1
        else:
            init[i][3:5] = Pre_conversion_point[x][0:2]
            init[i][5] = 1 
            init[i][-3:-1] = Pre_conversion_point[x][0:2]
            init[i][-1] = 1 
            init[i][-3:] = init[i][-3:] * (conversion_complete_point[x][count]) * -1
            count += 1  
        if count == 2:
            x += 1
            count = 0
    return init

def find_perspective_matrix(matrix):
    '''  
    SVD分解
    '''
    svd_matrix = np.transpose(matrix).dot(matrix)
    U, S, V = np.linalg.svd(svd_matrix)  # nx9 9x9 9x9
    # print(S) # eigenvector
    min_eigenvalue = min(S)
    new_S = np.zeros(9)
    ### S 的最小 eigenvalue 對應到 V 的 column 是我們的答案
    new_S[8] = min_eigenvalue
    calculate = (new_S).dot(V)
    calculate = calculate.reshape(3,3)
    calculate_answer = calculate / calculate[-1][-1]
    return calculate_answer

## 將兩張圖片疊合，順便過濾
## 修改了 warp
@jit(nopython=True)  
def warp_pic(target, background_pic, calculate_answer_matrix):
    """
    Put target pic on the background pic
    extend can decide to extend output pic size
    """
    # if extend:
    #     new_img = np.zeros((background_pic.shape[0], (target.shape[1]+background_pic.shape[1]), 3), dtype = np.uint8)       
    # else:  
    #     new_img = np.zeros((background_pic.shape[0], background_pic.shape[1], 3), dtype = np.uint8)  
    for u in range(background_pic.shape[0]):
        for v in range(background_pic.shape[1]):
            # np.dot 在 jit 不支援，用手刻
            ans_0 = calculate_answer_matrix[0][0] * v + calculate_answer_matrix[0][1] * u + calculate_answer_matrix[0][2] * 1
            ans_1 = calculate_answer_matrix[1][0] * v + calculate_answer_matrix[1][1] * u + calculate_answer_matrix[1][2] * 1
            ans_2 = calculate_answer_matrix[2][0] * v + calculate_answer_matrix[2][1] * u + calculate_answer_matrix[2][2] * 1
            ans_0 = ans_0 / ans_2
            ans_1 = ans_1 / ans_2
            ans_0 = int(ans_0)
            ans_1 = int(ans_1)
            # # 先把一張貼上去
            # if  v < background_pic.shape[1] and u < background_pic.shape[0]:
            #     new_img[u,v:] = background_pic[u,v,:]
            # ans[0] = 橫坐標，ans[1] = 縱座標
            if (ans_0 > 0 and ans_1 > 0 and  target.shape[0] > ans_1 and  target.shape[1] > ans_0):
                background_pic[u,v,:] = target[ans_1, ans_0, :] 
    return background_pic

def get_pic_edge_point(pic):
    edge_point = [[0,0], [0,pic.shape[0]], [pic.shape[1], pic.shape[0]], [pic.shape[1],0]]
    return edge_point

def warp_anthoer_pic(name, background_pic, feature_pic_edge):
    new_pic = cv2.imread(name)
    new_pic_point = get_pic_edge_point(new_pic)
    print(new_pic_point)
    H = find_H_matrix(feature_pic_edge, new_pic_point)  
    match_matrix = find_perspective_matrix(H)
    new_img = warp_pic(new_pic, background_pic, match_matrix)

    cv2.imshow('test', new_img)
    cv2.waitKey(1)