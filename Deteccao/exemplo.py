from re import I
import cv2 as cv

classificador = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

beatles = cv.imread('Deteccao\pessoas\\pessoas3.jpg')
imagemCinza = cv.cvtColor(beatles, cv.COLOR_BGR2GRAY)

facesDetectada = classificador.detectMultiScale(imagemCinza, 1.1, 4, I, (30, 30))

print(len(facesDetectada))
for (x,y,w,h) in facesDetectada:
    cv.rectangle(beatles,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow('Faces', beatles)
cv.waitKey(0)