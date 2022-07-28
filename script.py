#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import time
import cv2
import numpy as np


# In[2]:


# 1280X1280 ALTA RESOLUÇÃO, PRECISA E BAIXA PERFOMANCE
SAVED_MODEL_1 = "saved_model_640" 

# 640X640 MEDIA RESOLUÇÃO, BOA PRECISÃO E MÉDIA PERFOMANCE
SAVED_MODEL_2 = "saved_model_320"

# 320X320 BAIXA RESOLUÇÃO, PRECISÃO MÉDIA E BOA PERFOMANCE
SAVED_MODEL_3 = "saved_model_224"

#OBS
# O NÍVEL DE PRECISÃO INDICA A CAPACIDADE DO MODELO EM DETECTAR
# OBJETOS CADA VEZ MENORES E COM BOA CONFIABILIDADE.

SAVED_MODEL = SAVED_MODEL_1


# In[3]:


model = tf.saved_model.load(SAVED_MODEL)
print("Modelo carregado!")


# In[4]:


category_index = {
    1: "com mascara",
    2: "sem mascara",
    3: "mascara incorreta"
}

colors = {
    1: (0,0,255),
    2: (255,0,0),
    3: (255,100,20)
}


# In[5]:


def nms(boxes, classes, scores, max_boxes=20, iou_treshold=0.5, score_treshold=0.6):
    boxes = tf.squeeze(boxes, 0)
    scores = tf.squeeze(scores, 0)
    classes = tf.squeeze(classes, 0)
    boxes_indexes, scores = tf.image.non_max_suppression_with_scores(boxes,
                                                                     scores,
                                                                     max_boxes,
                                                                     iou_threshold=iou_treshold,
                                                                     score_threshold=score_treshold)
    return tf.gather(boxes, boxes_indexes), tf.gather(classes, boxes_indexes), scores

def predict(image, model):
#     if SAVED_MODEL == SAVED_MODEL_4:
#         image = tf.cast(image, tf.float32)
    
    image_tensor = tf.expand_dims(image, 0)
    start = time.time()
    detection_dict = model(image_tensor)
    end1 = time.time()
    return drawn_boxes_on_image_array(image, detection_dict, start, end1)
    

def drawn_boxes_on_image_array(image, detection_dict, t1, t2, min_score=0.3):
    boxes = detection_dict['detection_boxes']
    scores = detection_dict['detection_scores']
    classes = detection_dict['detection_classes']
    boxes, classes, scores = nms(boxes,
                                 classes,
                                 scores,
                                 max_boxes=10,
                                 score_treshold=min_score)
    nms_time = time.time() - t2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    H,W,_ = image.shape
    image_shape = np.array([H,W,H,W])
    
    for i, box in enumerate(boxes):
        class_ = classes[i].numpy()
        score = scores[i].numpy()
        box = (box.numpy()*image_shape).astype(int)
        
        text = "{} - {:.2f}".format(category_index[class_], score)
        
        color = colors[class_]
        
        y1, x1, y2, x2 = box
        #print(class_, color)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        text_size, _ = cv2.getTextSize(text, font, 0.8, 2)
        w,h = text_size
        
        cv2.rectangle(image, (x1-1, y1-h-2), (x1+w, y1), color, -1)
        cv2.putText(image, text, (x1, y1-1), font, 0.8, (255,255,255), 2)

    detection_time = t2 - t1
    final_time = time.time()
    total_time = final_time - t1
    vis_time = final_time - t2
    fps = 1/total_time
    cv2.putText(image, 'det:{}ms | FPS: {:.1f}'.format(
        int(detection_time*1000), fps), (30,30), font, 0.8, (255,255,255),2)
    
    return image


# In[6]:


LOWEST = (320, 320)
LOW = (1066, 600)
MID = (1920, 1080)
HIGH = (3840, 2160)

RES = MID


# In[7]:


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])


# In[8]:


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
time_stamp = time.strftime('%Y-%m-%d-%H%M%S')
video = cv2.VideoWriter(f'results/video_{time_stamp}.mp4', fourcc, 10, RES)


# In[9]:


while cam.isOpened():
    try:
        ret, frame = cam.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = predict(frame, model)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        video.write(frame)
        cv2.imshow("image", frame/255)
        #video.write()
        cv2.waitKey(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break

cam.release()
video.release()


# In[ ]:




