def main():
	'''  method to detect the face using haarcascade xml file '''
	import cv2
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

	cap = cv2.VideoCapture(0)

	while True:
		# start the camera to capture the real time video
		ret, img = cap.read()
		# convert the whole video to gray scale
		grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# detect faces using haarcascade for frontal images
		faces = face_cascade.detectMultiScale(grayscale, 1.1, 5)
		# draw the rectangle around the detected faces
		for (x, y, h, w) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
			roi_gray = grayscale[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

			eyes = eye_cascade.detectMultiScale(roi_gray)
			for (x1, y1, h1,  w1) in eyes:
				cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 1)
		
		cv2.imshow('img', img)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	''' entry point of the program '''
	main()
