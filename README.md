# face-detection
THIS PROJECT DETECTS FACES FROM IMAGES AND LIVE VIDEOS.


#FOR THE EFFECTIVE FUNCTIONING OF THE PYTHON SCRIPTS YOU NEED TO INSTALL opencv-python,imutils,argparse,pickles,numpy,scikit-learn.


This repository consists of two python scripts , pre-trained ML caffe model , commands to run the python scripts as two txt files, and sample images to test on.

#for running the face detection from images - change the directory to  where these files are located. then run the command from command_faces.txt file (which is locateed in this repository). while this code is running it performs operations on the sampleimage1 and display the image with bounding boxes around detected faces. you can exit from this by pressing q . you can change the image which the modle is working on by replacing the sample1.jpg  from --image sample1.jpg with the name of image you want to perform operation on.

#for running the face detection from live videos using your web cam - hange the directory to  where these files are located. then run the command from command_video.txt file (which is locateed in this repository). the web cam opens up and starts detcting the face with bounding boxes around detected faces.
