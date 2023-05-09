import cv2
import os
from keras.models import load_model
import numpy as np
import streamlit as st
from pygame import mixer
import time
from PIL import Image
mixer.init()
sound = mixer.Sound('alarm.wav')

st.set_page_config(
   page_title="Driver Drowsiness Detection App",
   page_icon="png.png",
   layout="wide",
   initial_sidebar_state="expanded",
)
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

# Initialize camera capture
cap = cv2.VideoCapture(0)



# Set up Streamlit app
st.title("Driver Drowsiness Detection")


if st.button("Open Camera"):

    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2] 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = np.argmax(model.predict(r_eye),axis=1)
            if(rpred[0]==1):
                lbl='Open' 
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = np.argmax(model.predict(l_eye),axis=1)
            if(lpred[0]==1):
                lbl='Open'   
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
            
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>10):
            #person is feeling sleepy so we beep the alarm
            cv2.putText(frame,"You are drowsy",(120,height-120), font, 2,(0,0,255),1,cv2.LINE_AA)
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()
                
            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()


img = Image.open('pic.jpg')


# Create two columns of different widths
col1 , col2 = st.columns(2)

# Display the image in the first column
with col1:
    with st.expander("", expanded=True):
        st.image(img,width=1200, use_column_width=True)

# Write text to the right of the image in the second column
with col2:
    st.write("This system uses computer vision and deep learning techniques to detect when a driver is becoming drowsy and alert them to take a break or stop driving.")
    st.write("It works by analyzing the driver's face in real time using a camera mounted on the dashboard. The system detects facial landmarks and uses them to determine if the driver's eyes are closed.")
    st.write("If the system detects signs of drowsiness, it triggers an alert to notify the driver to take a break. This alert could be a sound and visual warning.")
    st.write("The system is built using the Keras deep learning framework and OpenCV computer vision library. It uses a convolutional neural network (CNN) to classify the driver's facial features like eyes and determine their level of alertness.")
    st.write("This type of system could be useful in improving road safety and preventing accidents caused by driver drowsiness. It could be integrated into existing driver assistance systems or installed as a standalone device.")

st.write("<h3>","Press 'q' to quit camera ", "</h3>",unsafe_allow_html=True)
st.write("<h2 style='text-align: center;'>","Made By Deepak and Karan", "<h2>",unsafe_allow_html=True)