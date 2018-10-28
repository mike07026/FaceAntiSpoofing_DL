import os
from os import listdir

TRAINING = 0
TESTING = 1

CASIA = 0
REPLAYATTACK = 1

def CreatLabelPath(PROCESSFLAG,DataBaseOpt):
    print("In function CreatLabelPath, start processing!")
    true_img_dir_CASIA = ['1.avi', '2.avi', 'HR_1.avi', 'HR_4.avi']

    if DataBaseOpt == REPLAYATTACK:
        path_train_root_img = "../../../Data/ReplayAttack/Train/Train_aligned_5p/"
        path_train_output = "../../../Data/ReplayAttack/Train/label_img_train_5p.txt"

        path_test_root_img = "../../../Data/ReplayAttack/Test/Test_aligned_5p/"
        path_test_output = "../../../Data/ReplayAttack/Test/label_img_test_5p.txt"
    elif DataBaseOpt == CASIA:
        path_train_root_img = "/media/haoran/Data1/LivenessDetection/Data/CBSR-Antispoofing/Train/Train_aligned_ALL/"
        path_train_output = "/media/haoran/Data1/LivenessDetection/Data/CBSR-Antispoofing/Train/label_img_train_all.txt"

        path_test_root_img = "/media/haoran/Data1/LivenessDetection/Data/CBSR-Antispoofing/Test/Test_aligned_ALL/"
        path_test_output = "/media/haoran/Data1/LivenessDetection/Data/CBSR-Antispoofing/Test/label_img_test_all.txt"

    if PROCESSFLAG == TRAINING:
        path_root_img = path_train_root_img
        path_output = path_train_output
        if os.path.exists(path_train_output):
            os.remove(path_train_output)
    else:
        path_root_img = path_test_root_img
        path_output = path_test_output
        if os.path.exists(path_test_output):
            os.remove(path_test_output)

    for dir_img in listdir(path_root_img):
        dir1_img = os.path.join(path_root_img, dir_img)
        #print "dir1_img" + dir1_img
        # for g_img in listdir(dir2_img):
        #     img_path = os.path.join(dir2_img,g_img)
        #     print(img_path)
        print(dir_img)
        if dir_img == 'Real':
            for f_img in listdir(dir1_img):
                img_path = os.path.join(dir1_img, f_img)
                with open(path_output, 'a') as f:
                    f.write(img_path + ' ' + '1' + '\n')
        elif dir_img == 'Attack':
            for f_img in listdir(dir1_img):
                img_path = os.path.join(dir1_img, f_img)
                with open(path_output, 'a') as f:
                    f.write(img_path + ' ' + '0' + '\n')
    print("Processing CreatLabelPath Success!")

CreatLabelPath(TRAINING,CASIA)
CreatLabelPath(TESTING,CASIA)
