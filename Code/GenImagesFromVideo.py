import os
import cv2
from os import listdir

TRAINING = 0
TESTING = 1

CASIA = 0
REPLAYATTACK = 1

def GenImagesFromVideo(PROCESSFLAG,DataBaseOpt,skipNum):
    print("In function GenImagesFromVideo, start processing!")
    if DataBaseOpt == REPLAYATTACK:
        Head_train = "../../Data/CBSR-Antispoofing/Train/"
        Head_S_train = Head_train + "Train_original/"  # from unzip the train/test.rar
        Head_T_train = Head_train + "Train_frames/"

        Head_test = "../../Data/CBSR-Antispoofing/Test/"
        Head_S_test = Head_test + "Test_original/"  # from unzip the train/test.rar
        Head_T_test = Head_test + "Test_frames/"
    elif DataBaseOpt == CASIA:
        Head_train = "../../Data/CBSR-Antispoofing/Train/"
        Head_S_train = Head_train + "Train_Original/"  # from unzip the train/test.rar
        Head_T_train = Head_train + "Train_frames/"

        Head_test = "../../Data/CBSR-Antispoofing/Test/"
        Head_S_test = Head_test + "Test_Original/"  # from unzip the train/test.rar
        Head_T_test = Head_test + "Test_frames/"
    else:
        print "DataBase Unsupport!!"
        return


    if PROCESSFLAG == TRAINING:
        Head_S = Head_S_train
        Head_T = Head_T_train
    else:
        Head_S = Head_S_test
        Head_T = Head_T_test

    for d in listdir(Head_S):  # d is file folder
        if not os.path.exists(Head_T + d):  # not exist
            os.makedirs(Head_T + d)
        else:
            pass
        for f in listdir(Head_S + d):
            if not os.path.exists(Head_T + d + "/" + f):  # not exist
                os.makedirs(Head_T + d + "/" + f)
            print(Head_S + d + "/" + f)
            
        for f in listdir(Head_S + d):
            vidcap = cv2.VideoCapture(Head_S + d + "/" + f)
            if vidcap.isOpened() == False:
               print "Open Video File Failed!!!"
               return
            print "Processing" + Head_S + d + "/" + f
            success, image = vidcap.read()
            count = 0
            skipcount = 0
            success = True                
            while success:
                success, image = vidcap.read()
                print("processing " + Head_S + d + "/" + f + '\n')
                if success:
                    skipcount += 1
                    if int(skipcount % skipNum) == 0:
                        cv2.imwrite(Head_T + d + "/" + f + "/frame%d.jpg" % count, image)
                        print(Head_T + d + "/" + f + "/frame%d.jpg" % count)
                        count += 1
            vidcap.release()
    print("Processing GenImagesFromVideo for train_attack Success!")


    return
