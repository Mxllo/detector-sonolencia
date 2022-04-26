from re import I
import cv2 as cv
from cv2 import CAP_MSMF

classificador = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
classificadorOlhos = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
classificadorBoca = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_mcs_mouth.xml')

countOlhos = 0
countBoca = 0

font                   = cv.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

video = cv.VideoCapture(0)

while True:
    conectado, frame = video.read()
    frame = cv.flip(frame, 1)
    frameCinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(frameCinza, scaleFactor=1.08, minNeighbors=8, minSize=(30, 30))
    for (x, y, l, a) in facesDetectadas:
        cv.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv.cvtColor(regiao, cv.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.08, minNeighbors=9, minSize=(30, 30))
        for (ox, oy, ol, oa) in olhosDetectados:
            cv.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
            if(len(olhosDetectados) < 2):
                countOlhos += 1
                if(countOlhos > 12):
                    cv.putText(frame, 
                'ALERTA! SONOLENCIA DETECTADA', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv.LINE_4)
                    print("Olho não detectado")  #aviso de sonolencia
            else:
                countOlhos = 0
        regiaoCinzaBoca = cv.cvtColor(regiao, cv.COLOR_BGR2GRAY)
        bocaDetectada = classificadorBoca.detectMultiScale(regiaoCinzaBoca, minSize=(150, 20))
        for (bx, by, bl, ba) in bocaDetectada:
            cv.rectangle(regiao, (bx, by), (bx + bl, by + ba), (0, 255, 0), 2)
            if(len(bocaDetectada) < 1):
                countBoca += 1
                if(countBoca > 5):
                    print("Boca não detectada")  #aviso de sonolencia
            else:
                countBoca = 0        

            
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break          
