# opencv-samples



Task:  Licence plate detection(LPD)


Algorithm                                                  drawbacks


1.Opencv Contours and text                  1.The accuracy depends on the clarity of image,
 recognition with Tesseract                     orientation, light exposure etc           
 

2.LPD using ML algorithms                   2.Difficult to train the data each and every time .

 1.Connected component analysis
   (Segment the image)
 2.Suppoort Vector Classifier
  (train ml model to predict characters)


3.LPD using YOLO V3                         3.yolo trained for russian plates only


4.LPD using haarcascade file and            4.Sometimes Tesseract library has failed to recognize the 
text recognition with Tesseract               characters properly
