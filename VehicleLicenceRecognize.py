#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
from skimage import io
import sys
import json
import numpy as np
import base64
import urllib2
import Image
import cStringIO

minPlateRatio = 2.5 # 车牌最小比例
maxPlateRatio = 5   # 车牌最大比例

none_img = cv2.imread("nonelicence.jpg")

# 图像处理
def imageProcess(gray):
    # 高斯平滑
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    
    #cv2.imwrite("gaussian_1.jpg", gaussian)

    # Sobel算子，X方向求梯度
    sobel = cv2.convertScaleAbs(cv2.Sobel(gaussian, cv2.CV_16S, 1, 0, ksize=3))
    #cv2.imwrite("a_1_sobel.jpg", sobel) 

    # 二值化
    ret, binary = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)
    #cv2.imwrite("a_2_binary.jpg", binary) 

    # 对二值化后的图像进行闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    #cv2.imwrite("a_3_closed.jpg", closed) 
    
    # 再通过腐蚀->膨胀 去掉比较小的噪点
    erosion = cv2.erode(closed, None, iterations=2)
    #cv2.imwrite("a_4_erosion.jpg", erosion) 

    dilation = cv2.dilate(erosion, None, iterations=2)

    #cv2.imwrite("a_5_dilation.jpg", dilation) 
    # 返回最终图像
    return dilation

# 找到符合车牌形状的矩形
def findPlateNumberRegion(img):

    region = []
    # 查找外框轮廓
    cvimage, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print("contours lenth is :%s" % (len(contours)))
    # 筛选面积小的


    for i in range(len(contours)):
        cnt = contours[i]
        # 计算轮廓面积
        area = cv2.contourArea(cnt)

        # 面积小的忽略
        if area < 2000:
            continue

        # 转换成对应的矩形（最小）
        rect = cv2.minAreaRect(cnt)
        #print("rect is:%s" % {rect})

        # 根据矩形转成box类型，并int化
        box = np.int32(cv2.boxPoints(rect))

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 正常情况车牌长高比在2.7-5之间,那种两行的有可能小于2.5，这里不考虑
        ratio = float(width) / float(height)
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        # 符合条件，加入到轮廓集合
        region.append(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)  
    #cv2.imwrite("aaaa.jpg", img)
    return region

def detect(img):

    #去掉；图像的上上下左右的1/4，突出查找图片的中间区域;
    height1,width1 = img.shape[:2]  #获取原图像的水平方向尺寸和垂直方向尺寸。
    width2 = width1*(720.0/height1)
    res_img0 = cv2.resize(img, (int(width2), 720),interpolation=cv2.INTER_CUBIC)
    res_img1 = res_img0[int(720.0/4):650, int(width2/4):int((width2/4)*3)]  
    # 转化成灰度图
    gray = cv2.cvtColor(res_img1, cv2.COLOR_BGR2GRAY)
    # 形态学变换的处理
    dilation = imageProcess(gray)
    # 查找车牌区域
    region = findPlateNumberRegion(dilation)

    #如果没找到相应的区域。可能是因为角度问题，调整顺时针10度角度
    h, w = dilation.shape[:2]
    center = (w // 2, h // 2)

    if len(region) == 0:

        # 逆时针-90°(即顺时针90°)旋转图片
        M = cv2.getRotationMatrix2D(center, -10, 1)
        rotated_dilation = cv2.warpAffine(dilation, M, (w, h))
        region = findPlateNumberRegion(rotated_dilation)

    if len(region) == 0:

        # 逆时针-90°(即顺时针90°)旋转图片
        M = cv2.getRotationMatrix2D(center, -20, 1)
        rotated_dilation = cv2.warpAffine(dilation, M, (w, h))
        region = findPlateNumberRegion(rotated_dilation) 

    #如果还没找到相应的区域。 再调整逆时针10度角度

    if len(region) == 0:

        # 逆时针-90°(即顺时针90°)旋转图片
        M = cv2.getRotationMatrix2D(center, 10, 1)
        rotated_dilation = cv2.warpAffine(dilation, M, (w, h))
        region = findPlateNumberRegion(rotated_dilation)           

    if len(region) == 0:

        # 逆时针-90°(即顺时针90°)旋转图片
        M = cv2.getRotationMatrix2D(center, 20, 1)
        rotated_dilation = cv2.warpAffine(dilation, M, (w, h))
        region = findPlateNumberRegion(rotated_dilation) 

    if len(region) == 0:
        return none_img

    # 默认取第一个
    box = region[0]
    #在原图画出轮廓
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    # 找出box四个角的x点，y点，构成数组并排序
    ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
    xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
    ys_sorted_index = np.argsort(ys)
    xs_sorted_index = np.argsort(xs)
    # 取最小的x，y 和最大的x，y 构成切割矩形对角线
    min_x = box[xs_sorted_index[0], 0]
    max_x = box[xs_sorted_index[3], 0]
    min_y = box[ys_sorted_index[0], 1]
    max_y = box[ys_sorted_index[3], 1]
  
    # 切割图片，其实就是取图片二维数组的在x、y维度上的最小minX,minY 到最大maxX,maxY区间的子数组
    img_plate = res_img1[min_y:max_y, min_x:max_x]

    #所有的牌号都统一高度：80 长度自适应：

    height1,width1 = img_plate.shape[:2]  #获取原图像的水平方向尺寸和垂直方向尺寸。
    #print height1/80.0
    width2 = width1*(80.0/height1)
    #print width2
    res_img2 = cv2.resize(img_plate,(int(width2), 80),interpolation=cv2.INTER_CUBIC) 
    #cv2.imwrite("a_6_resize.jpg", res_img2)
    return res_img2
  
def licenceRecognize(imgurl):

    img = None
    status = 0
    try:
        img = io.imread(imgurl)
        #file = cStringIO.StringIO(urllib2.urlopen(imgurl).read())
        #img = Image.open(file)
        img = detect(img)
        status = 1

    except Exception, e:
        print e 
        img = none_img
    
    finally:
        #cv2.imwrite("aaaaa.jpg", none_img)
        img_encode = cv2.imencode('.jpg', img)[1]
        data_encode = np.array(img_encode)
        str_encode = data_encode.tostring()
        ret = {}
        ret["status"] = status 
        ret["img"] = "data:image/jpg;base64,"+base64.b64encode(str_encode) 
        return json.dumps(ret)

if __name__ == '__main__':

    #img = cv2.imread(sys.argv[1])
    #img = detect(img)
    print licenceRecognize(sys.argv[1])
