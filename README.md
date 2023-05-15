# Real-time Drowsiness Detection

This project was made as a part of **DA526(Image processing with Machine Learning)** course taken by **Prof. Debanga R. Neog** Jan-May 2023.
It aims to develop a real-time drowsiness detection system for individuals in a video feed. The system will first detect faces in the video and then classify them as drowsy,low vigilant or awake using a pretrained model trained on a dataset of alert,low vigilant and drowsy individuals. The model will output a probability score for each class, and the system will use a threshold to determine whether the individual is drowsy or awake along with localisation of the class(bounding box). This approach can have practical applications in driver safety and workplace safety, but the system's accuracy will depend on the quality of the face detection algorithm, the training data, and the threshold chosen.


Project Contributors
 - **Mayukh Das**
 - **Debarpan Jana**
 - **Bitan Guha Roy**
 - **Mahasin Hossen Munshi**
 - **Madhurima Sen**

The github repo consists of 6 jupyter notebook files each of which performs a specific subtasks of the overall project.

## 1. Dataset Collection and Preprocessing

UTA-RLDD(Real Life Drowsiness Dataset)  is used for training and validation and custom dataset for testing, it was created for the task of multi- stage drowsiness detection, targeting not only extreme and easily visible cases, but also subtle cases of drowsiness. It consists of around 30 hours of RGB videos of **60** healthy participants. For each participant we obtained one video for each of three different classes: **awake, drowsy, and low vigilant**, for a total of **180** videos.

The three classes were explained to the participants as follows:
1) Awake: Subjects were told that being alert meant they were experiencing no signs of sleepiness.
2) Low Vigilant: this state corresponds to subtle cases when some signs of sleepiness appear, or sleepiness is present but no effort to keep alert is required.
3) Drowsy: This state means that the subject needs to actively try to not fall asleep.

We have extracted images from these videos and notebook for the code is [dataset_preparation.ipynb](dataset_preparation.ipynb).Anyway We have already extracted the images and uploaded in kaggle to be used.
To use kaggle datasets kaggle must be installed as follows :

 ``` pip install kaggle ```

All datasets api command are already provided in [datasets.txt](datasets.txt) 

For YOLOv5 model the repo as well as dataset is present in ``` data/train ``` folder consisting of 330 labelled(in yolo format) images of each class(awake,drowsy,low vigilant). These images are subset of the pre-processed larger dataset

## 2. Baseline Model Set up
## 3. Finetuned Model Set up
## 4. YOLOv5 Model Set up


