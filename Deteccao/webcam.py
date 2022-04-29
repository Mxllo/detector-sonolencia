from re import I
import cv2 as cv
from cv2 import CAP_MSMF

classificador = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
classificadorOlhos = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
classificadorBoca = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_mcs_mouth.xml')

isOlhosDetectados = False
isBocaDetectada = False
isFaceDetectada = False

font                   = cv.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

video = cv.VideoCapture(0)

def detectaSonolencia(isOlhosDetectados, isBocaDetectada, isFaceDetectada):
    if(isOlhosDetectados == True and isBocaDetectada == True and isFaceDetectada == True):
        cv.putText(frame, "Não detectado", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    else:
        print("detectou sonolencia")
        print("Olhos: ", isOlhosDetectados)
        print("Boca: ", isBocaDetectada)
        print("Face: ", isFaceDetectada)
        cv.putText(frame, "Sonolencia detectada", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
        

while True:
    conectado, frame = video.read()
    frame = cv.flip(frame, 1)
    frameCinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(frameCinza, scaleFactor=1.08, minNeighbors=8, minSize=(30, 30))
    for (x, y, l, a) in facesDetectadas:
        if(len(facesDetectadas) > 0):
            isFaceDetectada = True
        else:
            isFaceDetectada = False

        cv.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv.cvtColor(regiao, cv.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.08, minNeighbors=9, minSize=(30, 30))
        print(len(olhosDetectados))
        for (ox, oy, ol, oa) in olhosDetectados:
            if(len(olhosDetectados) >= 2):
                isOlhosDetectados = True
            else:
                isOlhosDetectados = False

            cv.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

        regiaoCinzaBoca = cv.cvtColor(regiao, cv.COLOR_BGR2GRAY)
        bocaDetectada = classificadorBoca.detectMultiScale(regiaoCinzaBoca, minSize=(150, 20))
        
        for (bx, by, bl, ba) in bocaDetectada:
            if(len(bocaDetectada) >= 1):
                isBocaDetectada = True
            else:
                isBocaDetectada = False

            cv.rectangle(regiao, (bx, by), (bx + bl, by + ba), (0, 255, 0), 2)   
        
    
    detectaSonolencia(isOlhosDetectados, isBocaDetectada, isFaceDetectada)
        
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break   


