# Computer Pointer Controller
In this project I create an app using the Intel Distribution of OpenVino that controls the mouse pointer using a video or camera as input.
The inference pipeline makes use of three different pretrained models, one for Face Detection, another for Facial Landmarks Detection, another for Head Pose Estimation, and yet another for Gaze Estimation, that takes as input the ouputs of the previous three models.

## Project Set Up and Installation
In order to run this project you first need to install the Intel Distribution of OpenVino on your machine. Detailed instructions (I did the project on Windows) can be found [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html).
After that, I suggest you install miniconda and create a conda environment using the .yaml file provided. You can find instructions on how to do that [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
Then, you can clone this repository.
After that, open up Anaconda prompt and activate the conda environment.
Next, initialize the OpenVino environment. On Windows, you cd to wherever you installed openvino/bin and you run the setupvars.bat file.
Then, use the model downloader that OpenVino provides to download the following three models: face-detection-adas-binary-0001, landmarks-regression-retail-0009, head-pose-estimation-adas-0001, and gaze-estimation-adas-0002.
An example is '''python wherever openvino is installed/deployment_tools/tools/model_downloader/downloader.py --name "name of model"'''
Note that the models will be downloaded in wherever openvino is installed/deployment_tools/tools/model_downloader/intel, if not specified otherwise in the arguments to the model downloader. 

## Demo
Open an Anaconda prompt, activate your environment, and initialize the OpenVino environment as detailed previously.
Then, cd to the src of the repository.
Run the main.py file as follows: ''' python main.py -fdm <Path of xml file of face detection model> \
-flm <Path of xml file of facial landmarks detection model> \
-hpem <Path of xml file of head pose estimation model> \
-gem <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam> 
-flags <Optional flags for visualization of model outputs'''
You can moreover use the -h command to see mode detailed help on running main.py.

## Documentation
You can find the documentation of the pretrained models that are used.
[Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
[Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
[Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
[Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

The command line arguments for running main.py are further explained here:
   -fdm, required, Path to .xml file of Face Detection model
   -flm, required, Path to .xml file of Facial Landmarks Detection model
   -hpem, required, Path to .xml file of Head Pose Estimation model
   -gem,required, Path to .xml file of Gaze Estimation model
   -i, required, Path to video file or enter cam for webcam
   -flags, optional, Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Separate each flag by space)
                             for seeing the visualization of different model outputs of each frame,
                             fd for Face Detection, fld for Facial Landmark Detection,
                             hp for Head Pose Estimation, ge for Gaze Estimation
   --cpu_extension, optional, CPU Extension for custom layers
   --prob_threshold, default="0.6", Probability threshold to be used by the model
   --device, requirements, default="CPU", Specify the target device to perform inference on: CPU, GPU, FPGA, or MYRIAD
   
Further explanation on Directory Structure:
As explained, the main.py is the script that runs the inference app on video or camera.
The face_detection.py, facial_landmarks_detection.py, head_pose_estimation.py, and gaze_estimation.py scripts are used to handle the input and output of the inference pipeline for each model.
The mouse_controller.py script uses the ouput from the gaze_estimation.py to control the mouse on the screen.
Moreover, the input_feeder.py is used as a helper function in the main.py script in order to read in frames from video or webcam.
Finally, in the bin file a sample video file that can be used for the demo is located.

## Benchmarks

The benchmark result of running the model on my machine's CPU with multiple model precisions are :

FP32:

The total model loading time was : 3.2sec
The total inference time was : 10.3sec
The total FPS was : 0.35fps

FP16:

The total model loading time was : 1.65sec
The total inference time was : 8.6sec
The total FPS was : 0.41fps

INT8:

The total model loading time was : 2.2sec
The total inference time was : 7.7sec
The total FPS was : 0.43fps

## Results

Based on the above results, we can see that reducing the precision we get slightly better results in model load time and inference time, with the FPS not changing that much.
The results though with the INT8 precision are worse in terms of accuracy compared to the other two.
Because the face detection model is used as input for the later models as well, it may be a good idea to use FP32 for this model and FP16 for the rest, striking a good balance between speed and accuracy.
