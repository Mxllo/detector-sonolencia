from re import I
import cv2 as cv
from cv2 import CAP_MSMF
from numpy import ndarray

# Setup de classificadores
classificador = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
classificadorOlhos = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
classificadorBoca = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_mcs_mouth.xml')

# Setup variaveis de controle
isOlhosDetectados = False
isBocaDetectada = False
isFaceDetectada = False
contadorSonolencia = 0

# Seta propriedades da fonte de texto
font                   = cv.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

# Setup da webcam
video = cv.VideoCapture(0)

# Realiza logica de detecção de sonolência
def detectaSonolencia(isOlhosDetectados, isBocaDetectada, isFaceDetectada):
    global contadorSonolencia
    if(isFaceDetectada):
        if(isOlhosDetectados and isBocaDetectada):
            contadorSonolencia = 0
            cv.putText(frame, "Face detectada", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
        else:
            contadorSonolencia = contadorSonolencia + 1

        if(contadorSonolencia > 10):
            cv.putText(frame, "Sonolencia detectada!", bottomLeftCornerOfText, font, fontScale, (0,0,255), thickness, lineType)
    else:
        cv.putText(frame, "Face nao detectada", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
        
# Valida se há face detectada
def validaDeteccaoFacial(facesDetectadas):
    global isFaceDetectada
    if(len(facesDetectadas) > 0):
        isFaceDetectada = True
    else:
        isFaceDetectada = False

# Valida se há olhos detectados
def validaDeteccaoOlhos(olhosDetectados):
    global isOlhosDetectados
    if(len(olhosDetectados) >= 2):
        isOlhosDetectados = True
    else:
        isOlhosDetectados = False

# Valida se há boca detectada
def validaDeteccaoBoca(bocaDetectada):
    global isBocaDetectada
    if(isinstance(bocaDetectada, ndarray)):
        isBocaDetectada = True
    else:
        isBocaDetectada = False

# Loop principal - captura e processa imagens
while True:

    _, frame = video.read()
    frame = cv.flip(frame, 1)
    frameCinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(frameCinza, scaleFactor=1.08, minNeighbors=8, minSize=(30, 30))

    for (x, y, l, a) in facesDetectadas:
        validaDeteccaoFacial(facesDetectadas)
        cv.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiaoFacial = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv.cvtColor(regiaoFacial, cv.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.08, minNeighbors=9, minSize=(30, 30))

        for (ox, oy, ol, oa) in olhosDetectados:
            cv.rectangle(regiaoFacial, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
        validaDeteccaoOlhos(olhosDetectados)

        regiaoCinzaBoca = cv.cvtColor(regiaoFacial, cv.COLOR_BGR2GRAY)
        bocaDetectada = classificadorBoca.detectMultiScale(regiaoCinzaBoca, scaleFactor=1.08, minNeighbors=23 ,minSize=(110, 20))
        
        for (bx, by, bl, ba) in bocaDetectada:
            cv.rectangle(regiaoFacial, (bx, by), (bx + bl, by + ba), (0, 255, 0), 2)
        validaDeteccaoBoca(bocaDetectada)

    detectaSonolencia(isOlhosDetectados, isBocaDetectada, isFaceDetectada)
        
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break