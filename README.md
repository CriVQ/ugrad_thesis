# ugrad_thesis

This github repository contains all materials and code for my undergrad thesis. This does not include the raw video data of the participants in the study as stipulated in the signed consent form.

The project integrates the Unity component and the Python component to analyze biomechanical data through pose estimation and sensor fusion, combining computer vision (via MediaPipe and camera input) with EMG signals processed in Python. Unity serves as the visualization and interaction environment, while Python handles data acquisition and computation. 

The Unity component under folder "st" was from Noaman with permission from his former adviser who was my co-adviser in this study. 
The original can be seen here: https://github.com/noamanmazhar/VRAmputee

The Unity component under stroke builds upon the previous work but catered towards stroke patients. 


How It Works:
The movement data of the participant is captured live or through a recorded session while interacting with the VR environment. Simultaneously, the Unity grabs IMU data from the Oculus VR.
This configuration complements the other's weakness and provides minimal data loss. 

The participant interacts with the environment with the use of the EMG sensor. In this project, the myoware 2.0 was specifically used. 
The arduino here functions as a basic read and write, only transmitting the sensor strength to the Unity environment. 

Unity visualizes the results, providing feedback for movement analysis or rehabilitation applications.


On the other hand, the python component contains an obsolete folder of past unoptimized code. This was reduced to a single file named optimized_pipeline.py. 

normalize_cycles.py was mainly used for adjusting and visualizing data. 
testforfiles.py was a sanity check for directory issues. 


The project requires cuda and cudnn to run. Specific versions can vary depending on the GPU used.

