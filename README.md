# Video2Sceneario

The main goal of this project is to generate test scenarios from a video that contains either a bug, crash, or sequence of steps on an Android application.

### Phase 1: Video Segmentation and Action Identification
The purpose of this step is taking the frames of the video and extract characteristics/features from the frames to identify different windows. There is a similar problem when analysing videos for movies, where researches extract information from the frames and sounds of the video to index videos. This problem is called video scene segmentation problem [1] in which you are interested to know how to divide video filmed into different shots in scenes. Techniques used to partition the videos involve defining two features: (i) global features that consider the information on the entire image/frame, (ii) local features that focuses on information in certain locations of the image/frame.

This will allow the detection of changes in the frames of the video to extract the different window candidates. To achieve this we computed the histogram between subsequent frames using a treshold to identify when the video changes activities/screens based on the colors.

Additionally, we want to extract the actions performed in the video assuming the feature to show touches on the screen is activated in Android settings. This will visually show a circle every time the user interacts with the screen with any type of action. We used different Computer Vision (CV) filters to identify the touch indicator but we were not able to successfully identify it under some conditions like low and high contrast. We decided to use Faster RCNN to identify the touch indicator since this fits perfectly in this problem where we want to detect the locations of the touch actions. We used the object detection library implemented in Tensorflow [4].

For the actions that require interaction with the keyboard we will have to identify the field that has focus and the content that is in there. This will involve OCR to extract the text as well as identification of component changes (i.e., editTexts). As output of this phase we will have a list of actions and information about when there is a possible change on the window to indicate a transition in the app.

### Phase 2: Replay Validation
This phase involves the validation of each of the steps generated in phase 1. We will execute each step in a device (e.g., physical device or emulator) and validate the current state with the frames extracted from the video. To compare the current state and the video we compare the screenshot from the device and a frame from the video using a CV technique. We are still exploring this part but we believe that Complex-Wavelet Structural Similarity Index (CW-SSIM) can help us with this based on preliminar results.

Furthermore, we will need to consider two cases when executing the events:
* Missed event
* Incorrect event identification

To solve these cases we could execute steps and validate each transition trying a different component when we trigger a different behavior than the one expected in the video. Additionally we could use CrashScope execution model to identify possible paths to take when we have one of the aforementioned cases.

Moreover, if we identify that using CV to compare two different images one from the current execution and the other one from the video is not enough. We could run ReDraw to extract information of the components.

### Phase 3: Script Generation
This phase will execute type by type to check feasibility of the type candidates comparing the current state of the execution with the candidate image. This will be executed similarly as MonkeyLab sequential that will validate each of the actions to make sure we generate a valid script. This phase will require to define a target device to generate a script for that particular device.

#### References
1. Kishi, Rodrigo Mitsuo, and Rudinei Goularte. "Video scene segmentation through an early fusion multimodal approach." Anais do XXII Simpósio Brasileiro de Sistemas Multimídia e Web 2 (2016).
2. Lindeberg, Tony. "Scale invariant feature transform." (2012)
3. Andow, Benjamin, Akhil Acharya, Dengfeng Li, William Enck, Kapil Singh, and Tao Xie. "UiRef: analysis of sensitive user inputs in Android applications." In Proceedings of the 10th ACM Conference on Security and Privacy in Wireless and Mobile Networks, pp. 23-34. ACM, 2017
4. https://github.com/tensorflow/models/tree/master/research/object_detection
