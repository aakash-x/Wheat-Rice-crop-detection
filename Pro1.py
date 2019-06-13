
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

wheat_casecade = cv2.CascadeClassifier('cascade.xml')

img = cv2.imread('P152.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

wheat  = wheat_casecade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in wheat:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

