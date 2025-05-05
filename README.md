# Demo Pulse Rate Estimation

## <img src="images/mentahan/Panah.svg" width="30px;"/> **Table Of Contents**
[Introduction]()

[Key Features](#key-features)

[Technology Application](#technology-application)

[Installation Steps](#installation-steps)

## <img src="images/mentahan/Panah.svg" width="30px;"/> **Introduction**
This project develops a pulse rate detection system using **remote photoplethysmography (rPPG)** technology based on a camera. The system uses Flask for the backend and OpenCV and MediaPipe for real-time image processing. The rPPG signal data is processed to calculate the pulse rate and display it on the frontend as a dynamic graph.

## <img src="images/mentahan/Panah.svg" width="30px;"/> **Key Features**
- **Pulse Rate Detection**: Using rPPG signals obtained from the camera to calculate and display the pulse rate (BPM).
- **Real-time Video Processing**: Camera images are processed to detect faces and extract the rPPG signal using the POS (Plane-Orthogonal-to-Skin) algorithm.
- **Interactive Frontend**: Displays live video from the camera, rPPG signal graphs, and pulse rate estimates.
- **Start and Stop Recording**: Users can start and stop the recording and signal processing through buttons.


## <img src="images/mentahan/Panah.svg" width="30px;"/> **Technologies Aplication**
<div align="left">

| Technology | Name | Description |
| :---: | :---: | :---: |
| <img src="images/logo apps/python.jpg" style="width:50px;"/> | **Python** | Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. |
| <img src="images/logo apps/flask.jpg" style="width:50px;"/> | **Flask** | Flask is web framework used to build the backend application. |
| <img src="images/logo apps/opencv.jpg" style="width:50px;"/> | **OpenCV** | Used for image and video processing. |
| <img src="images/logo apps/mediapipe.jpg" style="width:50px;"/> | **MediaPipe** | MediaPipe is a cross-platform framework for building multimodal applied machine learning pipelines. MediaPipe is used for detecting faces and facial landmarks in the images. |
| <img src="images/logo apps/scipy.jpg" style="width:50px;"/> | **Scipy** | For filtering and processing rPPG signals. |
| <img src="images/logo apps/html5.jpg" style="width:50px;"/> | **HTML5 Canvas** | Used to draw the dynamic rPPG signal graph. |

</div>

## <img src="images/mentahan/Panah.svg" width="30px;"/> **Installation Steps**
### <img src="images/mentahan/Panah2.png" width="30px;"/> **Preparation of Needs**
Some of the preparations needed to carry out this research project are as follows:

<li> Install python software/code first </li>

```bash
https://www.python.org/downloads/
```

<li> After installing, first check whether Python has been installed properly using the following command, make sure the Python version you are using is between 3.10 and 3.12. : </li>

```bash
python --version
```

<li> Once the python version appears, please open a text editor that supports it such as Visual Studio Code. Here are the links to use (please download and install) :</li>

```bash
[Software VISUAL STUDIO CODE](https://code.visualstudio.com/)
```

### <img src="images/mentahan/Panah2.png" width="30px;"/> **Program Running Stage**
<li> Open a terminal / something like GitBash etc. Please clone this Repository by following the following command and copy it in your terminal: </li>

```bash
    git clone https://github.com/Ardoni121140141/Demo-Pulse-Rate-Estimation.git
```

<li>Please change the directory to point to the clone folder with the following command:</li>

```bash
   cd Demo-Pulse-Rate-Estimation
```
<li> To install requirements, please use the following command: </li>

```bash
pip install -r requirements.txt
```

<li> After that, please run the following command to run the program:</li>

```bash
python deployment\app.py
```

