######################################################################
###########All the Function doing the data preprocessing Packaged#####
######################################################################

import GenImagesFromVideo
#import CreatPath_Labeltxt
#import Landmark_5p_caffe

TRAINING = 0
TESTING = 1

CPU = 0
GPU = 1

CASIA = 0
REPLAYATTACK = 1

def main():
    skipNum = 1

    ######video  frames pictures###

    GenImagesFromVideo.GenImagesFromVideo(TRAINING, CASIA, skipNum)
    GenImagesFromVideo.GenImagesFromVideo(TESTING, CASIA, skipNum)


    #####mtcnn aligned  Landmark_5p_caffe#####

    #mtcnnParam1 = [40, 0.709, [0.6, 0.7, 0.7], [0.5, 0.7, 0.7, 0.7]]
    #detect_acc, average_time = Landmark_5p_caffe.ProcessLandmark_5p(TRAINING, REPLAYATTACK, mtcnnParam1, GPU)
    #print "######detect_acc is %f, average_time is %f#########"%(detect_acc,average_time)
    #detect_acc, average_time = Landmark_5p_caffe.ProcessLandmark_5p(TESTING, REPLAYATTACK, mtcnnParam1, GPU)
    #print "######detect_acc is %f, average_time is %f#########"%(detect_acc,average_time)


    # # #Create label path

    #CreatPath_Labeltxt.CreatLabelPath(TRAINING,REPLAYATTACK)
    #CreatPath_Labeltxt.CreatLabelPath(TESTING,REPLAYATTACK)


if __name__ == "__main__":
    main()
