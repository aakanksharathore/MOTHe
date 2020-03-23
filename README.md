# MOTHe

Mothe is a pipeline developed to detect and track multiple animals in a heterogeneous environment. MOTHe is a python based repository and it uses *Convolutional Neural Network(CNN)* architecture for the object detection task.  It takes a digital image as an input and reads its features to assign a category. These algorithms are learning algorithms which means that they extract features from the images by using huge amounts of labeled training data. Once the CNN models are trained, these models can be used to classify novel data (images). MOTHe is designed to be generic which empowers the user to track objects of interest even in a natural setting. 

__MOTHe has been developed and tested on Ubuntu 16.04 using python 3.5.2__

## PIPELINE DESCRIPTION:

MOTHe can automate all the tasks associated with object classification and is divided into 5 executables dedicated to the following tasks.

1. __System configuration__: The system configuration is used to setup MOTHe on the users system. Basic details such as the path to the local repository, path to the video to be processed, the size of the individial to be cropped and the size of the bounding box to be drawn during the detection phase.

2. __Dataset generation__: The dataset generation is a crucial step towards object detection and tracking. The manual effort required to generate the required amount of training data is huge. The data generation class and executable highly automates the process by allowing the user to crop the region of interest by simple clicks over a GUI and automatically saves the images in appropriate folders. 
   
3. __Training the convolutional neural network__: After generating sufficient number of training example, the data is used to train the neural network. The neural network produces a classifier as the output. The accuracy of the classifier is dependent on how well the network is trainied, which in turn depends on the quality and quantity of training data (See section "__How much training data do I need?__"). The various tuning parameters of the network are fixed to render the process easy for the users. This network works well for binary classification - object of interest (animals) and background. Multi-class classification is not supported on this pipeline.

4. __Object detection__: This is the most crucial module in the repository. It performs two key tasks - it first identifies the regions in the image which can potentially have animals, this is called localisation; then it performs classification on the cropped regions. This classification is done using a small CNN (6 convolutional layers). Output is in the form of *.csv* files which contains the locations of the identified animals in each frame.
   
5. __Object tracking__: Object tracking is the final goal of the MOTHe. This module assigns unique IDs to the detected individuals and generates their trajectories. We have separated detection and tracking modules, so that it can also be used by someone interested only in the count data (eg. surveys). This modularisation also provides flexibility of using more sophisticated tracking algorithms to the experienced programmers. We use an existing code for the tracking task (from the Github page of ref). This algorithm uses Kalman filters and Hungarian algorithm. 
This script can be run once the detections are generated in the previous step. Output is a \textit{.csv} file which contains individual IDs and locations for each frame. A video output with the unique IDs on each individual is also generated.
   
__MOTHe flowchart__
 
 <br>
 <img height="700" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/flowchart_mothe.jpg">
 <br>
 
 The flowchart depicts the order of the functions to be used in MOTHe. The user should follow the following steps to successfully detect and track multiple animals using MOTHe
 
 ## INSTALLATION

The current version of MOTHe is supported for Linux (ubuntu) machines. We provide command line user functionality for all the executable files of MOTHe. 
The usage of MOTHe requires certain packages to be installed on your ubuntu machine. A *mothe.sh* shell script is provided which helps in the installation of these packages in a single step. 

To get started with MOTHe, you need to download the MOTHe git repository on your local machine.
Open the terminal and use the 'git clone' command to clone the MOTHe repository as shown in the figure below. You can also download the MOTHe repository directly from the [github page](https://github.com/tee-lab).

`$ git clone https://github.com/tee-lab/MOTHe/`


<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/clone_mothe.png">
<br>
Once the repository is succesfully clones, you will see below output on the screen.

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/change_directory.png">
<br>

After downloading the repository on the local system, open the terminal and navigate to the local MOTHe directory 
as shown below:

`$ cd Desktop/user/MOTHe`

Replace above path with the path to MOTHe repository in your system.
            
<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/change_directory.png">
<br>


The usage of MOTHe requires certain packages to be installed on your ubuntu machine. A *mothe.sh* shell script is provided which helps in the installations of these packages in a single step. List the files in the MOTHe directory and take note of the *mothe.sh* file

`$ ls`

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/mothe_content.png">
<br>

To use the *mothe.sh* script, you have to first change the permissions to the file using the following command

`$ chmod +x ./mothe.sh`

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/chmod.png">
<br>

After changing the permissions to the *mothe.sh* file, use the bash command to use the shell script as shown below

`$ sudo bash mothe.sh`

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/bashmoth.png">
<br>

The *mothe.sh* also installs and configures a virtual environment where an environment can be created specific to MOTHe. Create a virtual environment by the name of mothe as shown below

`$ mkvirtualenv mothe -p python3`

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/mkvir.png">
<br> 

After creating the the virtual environment, activate the virtual environment as follows

`$ workon mothe`

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/workon.png">
<br> 

After activating the mothe environment, configure all python modules using the *requirement.txt* as follows:

`$ python3 -m pip install -r requirement.txt`
          
<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/reqpip.png">
<br>

User also has an option to install these modules manually. Open the *requirement.txt* file and install all the modules listed in this file.


 ## USING MOTHe 
 
 Users can run MOTHe to detect and track single or multiple individuals in the videos (or images). In this section, we describe the step-by-step procedure to run/test MOTHe. If you are interested in learning the procedure first by running it on our videos (and data), follow the guidelines under subsection **"Testing"** in each step.
 
 1. __Step 1 - System configuration__: File name - *configure.py*
 
This step is used to set parameters of MOTHe. All the parameters are saved in *config.yml*.
Parameters to be set in this step - home directory, cropping size of animals in the videos, path to video files etc. 
It wil also prompt to enter runmode, if you are exploring MOTHe on blackbuck or wasp data, enter 0 otherwise for your data enter 1.
 
Run the following command in your terminal from your MOTHe directory-
 
 `$ python configure.py`
 
 Once the above command is executed, user will be prompted to enter path to MOTHe directory in user's system.
 
 <br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/configure.png">
<br>

After entering the path, a prompt for selecting a video will appear.
 Select any video from the videos in which you want to track the animals and draw a bounding box around the animal.
 Press **c** to confirm the selection, **r** to discard and draw again.
 
 <br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/boundingbox.png">
<br>

2. __Step 2 - Dataset generation__: File name - *generate_dataset.py*

This program will pick frames from the videos, user can click on animals or background in these images to create samples for both categories (animal and background). Examples of animal of ineterst will be saved in the folder **yes** and background in the folder **no**.
User needs to generate at least 8k-10k samples for each category (see section **How much training data do I need?** for detailed guidelines). One must ensure to take a wide representation of forms in which animals appears in the videos and same for the background variation.

To generate the training data, run the following code in the terminal:

`$ python generate_dataset.py {step}`

Replace `{step}` with the number of frames you want to skip. This number will depend on the duration and number of videos.
For example, if you have 20 10-minute videos at 30 FPS and on an average 50 animals in each frame, to generate 10k images we need at least 500 images from each video. On an average we will get 50 animals from one frame so we need 10 frames from each video. So, in this case we would set step size as 150.

 <br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/dataset1.png">
<br>

Once the code is run, it will ask user to enter **"yes"** if they wish to create samples for animal category and **"no"** for background category. Please note that you would need to run this code multiple times for "yes" and "no" categories as it is better to take sample representation from many videos in your dataset.


 <br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/dataset2.png">
<br>

After this input, it will prompt user to select a video from which samples will be saved. After selecting the video, frames from this video will be displayed. User can click on the animals or different points on the background dependingon which category data is being created in the particular execution. Once you have clicked on desired number of points in one frame, press **n** to coninue to the next frame. Press **q** when done and you want to exit. 


 <br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/dataset3.png">
<br>

**For testing:**
If you wish to test (learn how to run) this module, download our video clips from the [google drive] (). You can then generate samples by choosing any of these videos. If you directly want to proceed to next steps, download our training data from the same drive.

 4. __Step 3 - Network training__: Filename - *train.py*
 
 To train the neural network, run the following code in the terminal. The following code trains the neural network
 on default parameters. If you wish to change the parameters you can do that directly in the script file. 
 
`$ python train.py`

You will see the status of training in the command line.

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/training0.png">
<br>

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/Training.png">
<br>


Once the training is complete, training accuracy will be displayed and trained model will saved with name *mothe_model.h5py* in the folder *classifiers*.

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/training1.png">
<br>

Training accuracy should be at least 98% for the model to work decently, if you wish to increase the accuracy, try training with more data.

**For testing -**
You can completely skip this step if you want to run MOTHe on wasp or blackbuck videos. For these videos, trained models are already saved in the repository.

4. __Step 4 - Object detection__: Filename - *object_detection.py*

This module will detect the animals (object of interest) in the video frames. As mentioned earlier, this process is done in two steps - first the code predicts the areas in which animal may be potentially present (localisation) and then these areas are passes to the network for classification task. For localisation, we need thrsholding approach which gives us regions which have animals as well as background noise. To perform this task, user needs to provide threshold values. See section **Parametrization** for details of how to choose threshold values for new datasets.

For object detection, run the following code in the terminal-
 
`$ python object_detection.py`

 <br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/detection1.png">
<br>

This code will also prompt for number of frames to skip. This will depend on the temporal resolution needed by user. If you need to track animal in each frame then select one, if you want data only 5 frames per second then enter 6. 
We recommend to not exceed this step size above 6 if user needs to track the individuals. For count data, this number can be as high as number of frames in the video.

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/detection2.png">
<br>

The status and number of frames processed will be shown on the screen-

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/detection3.png">
<br>


The output from this step is saved in the form of *.csv* files and videos.

**For testing -**

If you wish to test this module on blackbuck or wasp data, you can use default parameters provided in *config.yml*.
Open *config.yml* in any text editor and set the values to wasp or blackbuck default mentioned in the comments of these parameters in the file.

<br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/threshold.png">
<br>
 
5. __Step 5 - Track linking__: Filename - *object_tracking.py*

This step is used to ascribe unique IDs to the detected animals and it gives us thetrajectoris of the animals. 
It will use the detections from previous step. Hence, the input for this step would be original video clip and *.csv* generated in the previous step.

Run command -
 
`$ python object_tracking.py`

 <br>
<img height="350" src="https://github.com/aakanksharathore/MOTHe/blob/master/help_screenshots/tracking1.png">
<br>

## HOW MUCH TRAINING DATA DO I NEED?

MOTHe uses a CNN which uses a set of labelled examples to learn the features of the objects. Neural Networks generally work well with huge number of training samples. We recommend using at least 8-10k image examples for the animal category. This number may need to be increased if the animal of interest shows a lot of variation in morphology. For example, if males and females are of different colors, it is important to include sufficient examples for both of them. Similarly, if the background is too heterogeneous then you may need more training data (around 1500-2000 samples for different types of variations in the background).
For example to train the MOTHe on our blackbuck videos, we used 9800 cropped samples for blackbuck (including males and females) and 19000 samples for the background because background included grass, soil, rocks, bushes, water etc.


## CHOOSING COLOR THRESHOLDS

The object detection steps requires user to enter threshold values in the config files. Object detection in MOTHe works in two steps, it first uses a color filter to identify the regions in the image on which to run the classification. We use color threshold to select these regions. You can see the values of thresholds for blackbuck and wasp videos in the *config.yml* file.
If you are running MOTHe on a new dataset, there are two ways to select appropriate threshold values:

1. You may open some frames from different videos in an interactive viewer (for example MATLAB image display), you can then click on the pixels on animal and check the RGB values (take the avergae of all 3 channels). Looking at these values in multiple instances will give you an idea to choose a starting minimum and maximum threshold. 
Run the detection with these thresholds and you can improve the detection by hit and trial method to tweak the threshold.

2. You can compare your videos to wasp and blackbuck videos and start with threshold values to which your data is more similar. For example, if your animal looks more similar to blackbuck in color and lighting conditions, you may start with default thresholds and improve the detection by changing lower and upper threshold by little amount at a time.



=======
Software to detect and track animals in their natural environment
>>>>>>> 4d4318b012c65d1f8915c007fd1cbaee09fe1244
