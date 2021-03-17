import numpy as np
import tensorflow as tf
import cv2
from core.yolov4 import filter_boxes
from core import utils
from PIL import Image
import os
import time


pathToTFliteModel = "checkpoints/yolov4-416.tflite"
pathToClassesNames = "data/classes/coco.names"
input_size = 416
image_path = 'data/test1.jpeg'
iou_treshold = 0.45
score_threshold = 0.25
ressources = 'RessourcesImagesVins'

#initialise descriptor
orb = cv2.ORB_create(1000)

#Import Labels
labels = open(pathToClassesNames).read().strip().split("\n")


#Import original image
original_image0 = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image0, cv2.COLOR_BGR2RGB)
image_h, image_w, _ = original_image.shape

#initialise image base
images=[]
classNames=[]
myList = os.listdir(ressources)
print('Total wines bottle detected in Wine Base',len(myList))
for cl in myList:
   imgCur = cv2.imread(f'{ressources}/{cl}',0)
   imgCur = cv2.resize(imgCur, (input_size, input_size))
   images.append(imgCur)
   classNames.append(cl)

#Preprocess original image to images data
image_data = cv2.resize(original_image, (input_size, input_size))
image_data = image_data / 255.

images_data = []
for i in range(1):
    images_data.append(image_data)
images_data = np.asarray(images_data).astype(np.float32)

#Import model
interpreter = tf.lite.Interpreter(model_path=pathToTFliteModel)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Make prediction
interpreter.set_tensor(input_details[0]['index'], images_data)
interpreter.invoke()
pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))

boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou_treshold,
        score_threshold=score_threshold
    )
pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]


#drawkeypoints function
def drawKeyPts(im,keyp,col,th):
    for curKey in keyp:
        x=int(curKey.pt[0])
        y=int(curKey.pt[1])
        size = int(curKey.size)
        cv2.circle(im,(x,y),10, col,thickness=th, lineType=2, shift=0)
    return im

def findDes(images):
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append([kp,des])
    return desList

desList= findDes(images)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
flann = cv2.FlannBasedMatcher(index_params, {})

def findID(img,desList,thres = 15):
   start = time.time()
   kp2,des2 = orb.detectAndCompute(img,None)
   matchList=[]
   finalVal=-1
   for kp1,desL in desList:
      if (len(desL) == 0):
         print('desL is empty')
         exit(1)
      if (len(des2) == 0):
         print('des2 is empty')
         exit(1)
      matches = flann.knnMatch(desL, des2, k=2)
      cor = []
      # ratio test as per Lowe's paper
      for m_n in matches:
         if len(m_n) != 2:
            continue
         elif m_n[0].distance < 0.75 * m_n[1].distance:
            cor.append([kp1[m_n[0].queryIdx].pt[0], kp1[m_n[0].queryIdx].pt[1],
                     kp2[m_n[0].trainIdx].pt[0], kp2[m_n[0].trainIdx].pt[1],
                     m_n[0].distance])
      matchList.append(len(cor))
   #print(matchList)
   if len(matchList)!=0:
      if max(matchList)>thres:
         finalVal=matchList.index(max(matchList))
   end = time.time()
   print("[INFO] Matching took {:.6f} seconds".format(end - start))
   return finalVal

# FLANN parameters
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
flann = cv2.FlannBasedMatcher(index_params, {})


n = 0
nbouteilles=0
for i in classes.numpy()[0][:valid_detections.numpy()[0]]:
    print(str(i) +" : " + labels[int(i)])
    if int(i) == 39 and scores.numpy()[0][n] >= 0.29:
        nbouteilles +=1
        print("A bottle has been found")
        print("Score :" + str(scores.numpy()[0][n]))
        coor = boxes.numpy()[0][n]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        print("Corresponding coord : " + str(coor))

        crop_img = original_image0[abs(int(coor[0])):int(coor[2]), abs(int(coor[1])):int(coor[3])]

        crop_img_resize = cv2.resize(crop_img, (416, 416))
        cv2.imshow("cropped" + str(nbouteilles), crop_img_resize)
        cv2.moveWindow("cropped" + str(nbouteilles), int(1920 / 10 * (nbouteilles - 1)), int(1080 / 2))
        kp, des = orb.detectAndCompute(crop_img_resize, None)
        crop_img_wkp = drawKeyPts(crop_img_resize.copy(), kp, (0, 255, 0), 2)

        cv2.imshow("cropped with keypoints" + str(nbouteilles), crop_img_wkp)
        cv2.moveWindow("cropped with keypoints" + str(nbouteilles), int((1920 / 2) + 1920 / 10 * (nbouteilles - 1)),
                       int(int(1080 / 2)))
        id = findID(crop_img, desList, thres=1)
        if id != -1:
            correspondingImage = cv2.imread(f'{ressources}/{myList[id]}', 1)
            correspondingImage = cv2.resize(correspondingImage, (int(1920 / 8), int(1080 / 2)))
            cv2.imshow("Corresponding image found for bottle " + str(nbouteilles), correspondingImage)
            cv2.moveWindow("Corresponding image found for bottle " + str(nbouteilles), int((1920 / 4) * 3),
                           int(0))
        else:
            print('Not found in Wine base')



    n+=1

image = utils.draw_bbox(original_image, pred_bbox)
# image = utils.draw_bbox(image_data*255, pred_bbox)
image = Image.fromarray(image.astype(np.uint8))
image.show()
image.save('result.png')
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
cv2.waitKey(0)