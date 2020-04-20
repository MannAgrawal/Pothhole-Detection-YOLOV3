# Pothole-Detection via YOLOV3
==============================


## Abstract
This Project is based upon collection of images from wide range of opensource pothole dataset and usinng Deep Learning
Technique( yolov3 ), we have trained a custom machine learning model for detection of potholes
inside.


if you are trying to test model that I've trained run this simple command referenced from the 
```
./darknet.exe detector demo data/obj.data cfg/yolov3-tiny_obj_custom.cfg backup/yolov3-tiny_obj_custom_final.weights video_path -out_filename output_path

```

#### Training Dataset

I did a large amount of labeling on different shape of image gathered from different opensource datasets. I used  [LabelIMG](https://github.com/tzutalin/labelImg) to identify areas of interest with bounding boxes. Potholes can be tricky, since they are different shapes, so I spent about a 2 days labeling so the algorithm could detect potholes of different sizes. This process uses xml(PascalVoC format) files, which were later converted into YOLO format using ````VOCtoYOLO.py````for each file but you can can convert directly to yolo format.

You don't need to resize the images for training but you can  **resize all the images** for faster processing to **416x416**



**__After annotating around 2000 images, I have got this dataset__**


**Download YOLO-pothole dataset**

| Folder Name        | Download Link           |
| -------------------|:-----------------------:|
| obj.zip | [Gdrive link](https://drive.google.com/file/d/192tnRaXvKxwx3frvtpTv3Fu23_4xgTpx/view?usp=sharing) |

**[Test video Download Link](https://www.youtube.com/watch?v=BQo87tGRM74&t=78s)**
```

**For custom Training Process**
````
extract the zip files and create a folder named as "obj" and move folder to data folder
Structure for the whole Test Dataset Links
````
├── obj
├──----- IMG_7300.jpg
├──----- IMG_7301.txt
├──----- IMG_7310.jpg
├──----- IMG_7311.txt
````
#### I have trained model till 4000 epoch

| Folder Name        | Download Link           |
| -------------------|:-----------------------:|
| yolov3_custom_1000.weights | [Gdrive link](https://drive.google.com/file/d/1--hOA0iz3eosuggp37vCVDHTY060IvsL/view?usp=sharing) |
| yolov3_custom_1000.weights | [Gdrive link](https://drive.google.com/file/d/1-12yxFkaftqLYj_znWoVlPedSEtrbnLA/view?usp=sharing) |
| yolov3_custom_3000.weights | [Gdrive link](https://drive.google.com/file/d/1xTUiIrVLX_EYwjBQ-1X--BK6kovpyjZE/view?usp=sharing) |
| yolov3_custom_last.weights | [Gdrive link](https://drive.google.com/file/d/1KIdYTwQQYuH6CkT6SeurT_tnF1086Vi_/view?usp=sharing) |



### Requirements
* Windows or Linux
* **CMake >= 3.8** for modern CUDA support: https://cmake.org/download/
* **CUDA 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* on Linux **GCC or Clang**, on Windows **MSVC 2015/2017/2019** https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community


# Process to Build Darknet

PLEASE INSTALL APPLICATIONS IN THE CORRECT ORDER. I am speaking from experience when I say that it can be a nightmare if you don’t follow this. Also, this is my recommendation for installing darknet and it may or may not work for you. **Full disclaimer**.

The original GitHub repository for Darknet is [here](https://github.com/pjreddie/darknet); however, we will be using [AlexeyAB’s version](https://github.com/AlexeyAB/darknet/) which is an exact copy of Darknet with additional Windows support.

I will assume that you have a GPU that has a compute compatibility version greater than 3.0. (Check if your GPU is good at this link.)


#### 1) The First Step is OpenCV


OpenCV was a nightmare for me but hopefully, it won’t be a pain for you. I used this [tutorial](https://www.learnopencv.com/install-opencv-4-on-windows/) to get OpenCV4. I will give a brief walkthrough and some advice.


##### Step 0.1: Install Visual Studio
Download and install Visual Studio 2017 community edition from https://visualstudio.microsoft.com/downloads/. Run the installer and click on Continue.


![VS Studio 2017 download](https://www.learnopencv.com/wp-content/uploads/2018/10/Visual-Studio-Installer.png)


Once the download is complete, the installer state would look like the following.


![VS Studio 2017 download](https://www.learnopencv.com/wp-content/uploads/2018/10/Visual-Studio-Installer-Download-Complete.png)


Next, we select the packages. We will select Desktop development with C++.


![VS Studio 2017 download](https://www.learnopencv.com/wp-content/uploads/2018/10/Select-Packages.png)


Finally, click on Install while downloading and wait while Visual Studio is installed.


##### Step 1.2: Install CMake


Please note that the version mentioned in the screenshots might be different from the latest versions available on the website. Please download the latest versions and treat the screenshots as reference.

Download and install CMake  from [here](https://cmake.org/download/)

During installation select **Add CMake to system PATH**


![cmake](https://www.learnopencv.com/wp-content/uploads/2018/09/cmake-3.png)


Remember to add CMake to your system PATH. If you forget to do so, you can add **{LOCATION OF CMAKE FOLDER}\bin** to your System Path in the Environment variables. An example CMake path is 


```
C:\Program Files\CMake\bin
```


If you do not know how to edit the System Path, please refer to this [link](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/).



#### Step 1.3: Install Anaconda (a python distribution)

Download and install Anaconda 64-bit version from https://www.anaconda.com/download/#windows.

Likewise, if you forget to add Anaconda to your System Path, simply add ```{LOCATION OF ANACONDA FOLDER}\Scripts``` to your System Path. An example is ```D:\Anaconda3\Scripts.```



### 2) The Second Step is CUDA
Click this [link](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10) and select Download. Of course, if you have Windows 7 or 8 then change the Version.

Once you run the installer, just keep clicking Next and verify that you do not encounter the screen below while installing.
![Your Visual Studio 2017 did not install properly](https://miro.medium.com/max/1400/1*2L-tkFVKRBSWzECGKZn4sw.png)


If it did then you need to uninstall Visual Studio 2019 and redownload. If you did not encounter the message above and CUDA was successfully installed then you can move on to the next part!


### 3) The Third Step is getting CuDNN
To download CuDNN please click this [link](https://developer.nvidia.com/rdp/cudnn-download). You will need to register for an Nvidia Developer account before getting CuDNN. Once you register, agree to the terms and conditions and click the installer as shown in the screenshot below.


![screenshot](https://miro.medium.com/max/1400/1*MvzOO0d76D4G5y0mAOYI4w.png)


Once the file has been downloaded, extract the contents directly to your C drive. Once you have finished extracting, verify that the Cuda folder exists in your C drive as shown below.


![screenshot](https://miro.medium.com/max/1400/1*MvzOO0d76D4G5y0mAOYI4w.png)


Checking for Cuda folder presence
After that open up your environment variables, and add ```C:\cuda\bin``` as a new entry to your System Path. Congratulations you have installed the major requirements!!



### 4) Final Step is getting Darknet


clone [this](https://github.com/AlexeyAB/darknet) repository:

How to compile on Windows (using CMake-GUI)
This is the recommended approach to build Darknet on Windows if you have already installed Visual Studio 2015/2017/2019, CUDA > 10.0, cuDNN > 7.0, and OpenCV > 2.4.

Use CMake-GUI as shown here on this IMAGE:

1. Configure
2. Optional platform for generator (Set: x64)
3. Finish
4. Generate
5. Open Project
6. Set: x64 & Release
7. Build
8. Build solution


# Create a custom YOLOv3 detector


#### In order to create a custom YOLOv3 detector we will need the following:

1. Labeled Custom Dataset
2. .cfg file (yolov3 architature configration file)
3. obj.data 
and 
4. obj.names files
5. train.txt and test.txt files (test.txt is optional here)
*



1. ### Manually Labeling Images with Annotation Tool
    
   You use annotation tool to manually draw your labels. For this project I did a large amount of labeling on different shape of image gathered from different opensource datasets. I used  [LabelIMG](https://github.com/tzutalin/labelImg) to identify areas of interest with bounding boxes. Potholes can be tricky, since they are different shapes, so I spent about a 2 days labeling so the algorithm could detect potholes of different sizes. This process uses xml(PascalVoC format) files, which were later converted into YOLO format using ````VOCtoYOLO.py````for each file but you can can convert directly to yolo format.
   
   It will create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and     put to file: object number and object coordinates on this image, for each object in new line: 

`   <object-class> <x_center> <y_center> <width> <height>`

   Where: 
   * `<object-class>` - integer object number from `0` to `(classes-1)`
   * `<x_center> <y_center> <width> <height>` - float values **relative** to width and height of image, it can be equal from `(0.0 to        1.0]`
   * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
   * atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

   For example for `img1.jpg` you will be created `img1.txt` containing:

   ```
   1 0.716797 0.395833 0.216406 0.147222
   0 0.687109 0.379167 0.255469 0.158333
   1 0.420312 0.395833 0.140625 0.166667
   ```


   After creating YOLO format datset put all images and all .txt file in one folder (for this project I named it obj putte into data        folder) you to to set your dataset path  in .data file.

2. ### Configuring Files for Training
   This step involves properly configuring your custom .cfg file, obj.data, obj.names and train.txt file.

   I have a detailed video on how to properly configure all four of these files to train a custom yolov3 detector. I will spare the time and ask you to watch the video in order to properly learn how to prepare the files.

   You can access the video with this link! [Configuring YOLOv3 Files for Training](https://www.youtube.com/watch?v=zJDUhGL26iU&t=300s)
   1. #### Cfg File
        Copy over the yolov3.cfg to edit couple of line instructed below.
       
        *I recommend having batch change line batch to batch=64 and subdivisions = 16 for ultimate results. If you run into any issues           then up subdivisions to 32.
        *change line max_batches to (classes*2000 but not less than 4000), f.e. [`max_batches=6000`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20) if you train for 3 classes.
        *change line steps to 80% and 90% of max_batches, f.e. f.e. [`steps=4800,5400`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L22)
        *set network size width=416 height=416 or any value multiple of 32.
        *change ```[filters=255]``` to ```filters=(classes + 5)x3```in the 3 ```[convolutional]``` before each ```[yolo]``` layer, keep         in mind that it only has to be the last ```[convolutional]``` before each of the ```[yolo]``` layers.
        *Make the rest of the changes to the cfg based on how many classes you are training your detector on.
        *when using ```[Gaussian_yolo]``` layers, change ```[filters=57] filters=(classes + 9)x3``` in the 3 [convolutional] before each                     ```[Gaussian_yolo]``` layer
        ```So if classes=1 then should be filters=18. If classes=2 then write filters=21.```
        (Do not write in the cfg-file: filters=(classes + 5)x3)
        
        (Generally filters depends on the classes, coords and number of masks, i.e. ```filters=(classes + coords + 1)*<number of                 mask>````, where mask is indices of anchors. If mask is absence, then ```filters=(classes + coords + 1)*num)```

        So for example, for 2 objects, your file yolo-obj.cfg should differ from yolov3.cfg in such lines in each of 3 [yolo]-layers:
        
        **Note: I set my max_batches = 4000, steps = 3600, 3200, I changed the classes = 1 in the three YOLO layers and filters = 18 in           the three convolutional layers before the YOLO layers.**

        *Optional: In each of the three yolo layers in the cfg, change one line from random = 1 to random = 0 to speed up training but           slightly reduce accuracy of model. Will also help save memory if you run into any memory issues.

   2. ##### obj.names and obj.data
      *Create file obj.names in the directory build\darknet\x64\data\, you will make this file exactly the same as your classes.txt            in the dataset generation step.

      *Create file obj.data in the directory build\darknet\x64\data\, containing (where classes = number of object):
       
       
         ```
         classes= 2
         train  = data/train.txt
         valid  = data/test.txt
         names = data/obj.names
         backup = backup/
         ```
          
          
   3. ##### Generating `train.txt` and `test.txt` 
      Luckily I have created a script that I showed in a past video that generates `train.txt` and `test.txt` for us.
        
        
       ```
       import os
       import random
       image_files = []
       os.chdir(os.path.join("data", "obj"))
       for filename in os.listdir(os.getcwd()):
           if (filename.endswith(".jpg")) or (filename.endswith(".JPG")):
                image_files.append("data/obj/" + filename)
       os.chdir("..")

       random.shuffle(image_files)
    
       # 80:20 split between train and test images
       with open("train.txt", "w") as outfile:
           for image in image_files[:round(0.8/len(image_files))]:
               outfile.write(image)
               outfile.write("\n")
           outfile.close()

       with open("test.txt", "w") as outfile:
           for image in image_files[round(0.8/len(image_files)):]:
               outfile.write(image)
               outfile.write("\n")
           outfile.close()
       ```
          
       an in directory `build\darknet\x64\data\`, with filenames of your images, each filename in new line, with path                          relative to `darknet.exe`, for example containing:
        
       ```
       data/obj/img1.jpg
       data/obj/img2.jpg
       data/obj/img3.jpg
       ```

3. ### Download pre-trained weights for the convolutional layers.
    This step downloads the weights for the convolutional layers of the YOLOv3 network. By using these weights it helps your custom         object detector to be way more accurate and not have to train as long. You don't have to use these weights but trust me it will help     your modle converge and be accurate way faster. USE IT!
    Download pre-trained weights for the convolutional layers and put to the directory build\darknet\x64
    
     * for `yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
    * for `yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-        PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)
 

4.### Start training by using the command line: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74`

   * (file `yolo-obj_last.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations)
   * (file `yolo-obj_xxxx.weights` will be saved to the `build\darknet\x64\backup\` for each 1000 iterations)
   * (to disable Loss-Window use `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show`, if you train on computer without monitor like a cloud Amazon EC2)
   * (to see the mAP & Loss-chart during training on remote server without GUI, use command `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map` then open URL `http://ip-address:8090` in Chrome/Firefox browser)

4.1. For training with mAP (mean average precisions) calculation for each 4 Epochs (set `valid=valid.txt` or `train.txt` in `obj.data` file) and run: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map`

5. ### After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

 * After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: `darknet.exe detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights`



# This fork #
This is fork of the Darknet repo with training data, and configs for aerial car detection by YOLOv3.
Weights can be downloaded here: https://drive.google.com/drive/folders/1fODck5uqfh3AXFE4ArqGcuPvIbdl7Z97?usp=sharing

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.


