import os
import glob
import shutil
import numpy as np
from tkinter import filedialog
RatPe=0.8
path =filedialog.askdirectory(initialdir="/",title='Please select a directory')
train_path=os.path.join(path,'train')
test_path=os.path.join(path,'valid')
dir_list=[name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
def Get_List(path,What):
    filelist = []
    filecount=0
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root,file))
    if(What=='data'):
        return(filelist)
    else:
        return(len(filelist))

def Move_File(Aux_Path,train_p,test_p,label,All_Image_List,ratio,sameCount):
     
     Image_List=All_Image_List[:sameCount]
     Auxilary=All_Image_List[sameCount:]
     print('Shuffeling Data for '+label+' Class')   
     random_set = np.random.permutation(len(Image_List))
     train_list = random_set[:round(len(random_set) * ratio)]
     valid_list = random_set[-(len(Image_List) - len(train_list))::]
     train_images = []
     valid_images = []
     for index in train_list:
         train_images.append(Image_List[index])
     for index in valid_list:
         valid_images.append(Image_List[index])
     print('Applying . . .') 
     for images in Auxilary:
        # print(images)
         shutil.move(images,os.path.join(os.path.join(Aux_Path,label),os.path.basename(images)))

     for images in train_images:
        # print(images)
         shutil.move(images,os.path.join(os.path.join(train_p,label),os.path.basename(images)))
     print('......................................................................')
     for images in valid_images:
        # print(images)
         shutil.move(images,os.path.join(os.path.join(test_p,label),os.path.basename(images)))
     print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
Aux=os.path.join(os.path.dirname(path),'Auxilary')
if not os.path.exists(Aux):
    os.mkdir(Aux)

dir_list=[name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
Counting=[]
for index in dir_list:
    Counting.append(Get_List(os.path.join(path,index),'dat'))
Counting.sort() 
minimum=Counting[0]
if "train" and "valid" in dir_list:
    total=Get_List(train_path,'dat')+Get_List(test_path,'dat')
    print('Detected Dataset Train to Test Ratio: '+str(round(Get_List(train_path,'dat')/total*100,0))+' %')
    print('Applying Dataset Train to Test Ratio: '+str(RatPe*100)+' %')
    label=[name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
    Counting=[]
    for item in label:
        Counting.append(Get_List(os.path.join(train_path,item),'dat')+(Get_List(os.path.join(test_path,item),'dat')))
    Counting.sort() 
    minimum=Counting[0]
    print('Train Path: '+train_path)
    print('Test Path: '+test_path)
    print('Auxilary Path: '+Aux)
    print('______________________________________________________________________'+'\n')
    for item in label:
        if not os.path.exists(os.path.join(Aux,item)):
            os.mkdir(os.path.join(Aux,item))
        Move_File(Aux,train_path,test_path,item,Get_List(os.path.join(train_path,item),'data')+(Get_List(os.path.join(test_path,item),'data')),RatPe,minimum)
    print('\n'+'Applied')   
else:
    print('Detected Virgin Dataset')
    print('Applying Dataset Train to Test Ratio: '+str(RatPe*100)+' %')
    label=[name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    print('Train Path: '+train_path)
    print('Test Path: '+test_path)
    print('Auxilary Path: '+Aux)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    for item in label:
        os.mkdir(os.path.join(Aux,item))
        os.mkdir(os.path.join(train_path,item))
        os.mkdir(os.path.join(test_path,item))
        Move_File(Aux,train_path,test_path,item,Get_List(os.path.join(path,item),'data'),RatPe,minimum)
        os.rmdir(os.path.join(path,item))   
    print('\n'+'Applied')   
          
       

    
    
