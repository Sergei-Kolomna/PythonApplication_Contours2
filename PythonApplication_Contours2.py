import cv2
import numpy as np

# сжимаем картинку на scale_percent %
def img_zip(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output

def nothing(*arg):
    pass

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

cv2.namedWindow("result") # создаем главное окно
cv2.namedWindow("settings") # создаем окно настроек

#set a thresh
cv2.createTrackbar('threshold', 'settings', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'settings', 25)

# Работа с видео
cap = cv2.VideoCapture("SCHOM.mp4")
while cap.isOpened():
    ret, image = cap.read()
    #inverse_image = cv2.bitwise_not(image)
    monochrome_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.getTrackbarPos('threshold', 'settings')
    ret, thresh_image = cv2.threshold(monochrome_image, thresh, 255, cv2.THRESH_BINARY)

    
    cv2.imshow("Image", thresh_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



        # Работа с фото
#my_phot = cv2.imread('picture2.jpeg')
#my_photo = img_zip(my_phot, 50)

#filterd_image1 = white_balance(my_photo)
##filterd_image  = cv2.medianBlur(filterd_image1,7)
##Инвертируем цвета
##filterd_image = cv2.bitwise_not(filterd_image1)
#img_grey = cv2.cvtColor(filterd_image1,cv2.COLOR_BGR2GRAY)

##create an empty image for contours
##img_contours = np.uint8(np.zeros((my_photo.shape[0],my_photo.shape[1])))

#while True:
#    thresh = cv2.getTrackbarPos('threshold', 'settings')
#    ret, thresh = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#    #th = cv2.adaptiveThreshold(threshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
#    #find contours
#    contours, hierarchy = cv2.findContours(image=thresh.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#    image_copy=my_photo.copy()
#    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
#    cv2.imshow('original', image_copy)
#    ch = cv2.waitKey(5)
#    if ch == 27:
#        break
#cv2.destroyAllWindows()
