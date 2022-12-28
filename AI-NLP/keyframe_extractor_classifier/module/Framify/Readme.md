# Hi, I'm Framify! 👋

I am a module known for extracting key frames from a video file. What a cool name!🤠

**What are Key Frames ?:**  `Keyframes are a set of summary frames to represent a video sequence accurately.`

**What is Key Frames Extraction?:** `Key Frame extraction is the process of extracting a frame or set of frames (key frames) that have a good representation of a shot. It must preserve the salient feature of the shot, while removing most of the repeated frames.`

## How to use me ? 💁

```javascript
from Framify import model_serve

response = model_serve(test_input, keyframe_output_location)
```
> __*keyframe_output_location*__ is the output directory where you want to save the extracted keyframes.
## Want to test me ? 🧐

No issues! check me, run the following command. Let me give you some dependency issues

```
python main.py --mode=package_test --func_test=all --file_path=path/to/file.json [optional]
```
**Here are the different options for package test**:

- _func_test_ = "all" -> runs test for all the components.
- _func_test_ = "serve" -> runs the test for serving component only.
- _func_test_ = "eval" -> runs the test for evaluation component only.

### Need to Evaluate an output ? ⚙️
> Change the GROUND_TRUTH and EVAL_VIDEO in config.py to the ground truth directory and Evaluation video respectively

Run
```sh
python3 main.py --mode=eval
```

#### Here is my Pluggable Component ⚓

This mode of running the package, showcase the capability to be able to plug in the eval and serving
component of this module into API integration or MLOPs engine.

- _mode_="serve" - run the serving component of the package.
- _mode_="eval" - run the evaluation component of the package.

`python main.py --mode=serve`

### Do the Parameters Setup

**Network Hyperparameters**

- **FRAME_RATE:** Number of Frames per second. The format is 1 / FPS.
- **DEFAULT_WINDOW_TYPE:** Default window type for smoothening of the array.
- **dsize:** height and width for resizing the image before processing.
- **Hashing**: Whether to use Image hashing for the process or not.


## Something related to installation 🔨

Dude! Install the dependencies.

```sh
pip install -r requirements.txt
```
## Let me tell you about my hidden features 📣

**I have three modes for getting your Key frames:**

- Using Image Hashing and local maxima/ threshold.
- Using Absolute difference and local maxima/ threshold (faster 🚀)
- Optical Flow and local maxima/ threshold (better accuracy ✔️)

Now it's up to you which one to use

> Go to config and change HASHING = True/ False or DO_OPTICAL_FLOW = True/ False
> To use Threshold based extraction change THRESHOLD = True


## Doubts?? Keep them coming 🎤

**Q1. What kind of models are used in this package ?**
**Ans.** As already told, I support two kind of models:
1. Image Hashing + Local Maximum/ threshold
2. Absolute difference and local maxima/ threshold
3. Optical Flow and local maxima/ threshold

**Q2. That was too fast. What is Image hashing + local maxima/ threshold ?**
**Ans.** Okay. Here we go! This algorithm consists of calculating the hashes of image using average hashing and perceptual hashing algorithm, then calculating the difference between two consecutive frames and then either saving the local maximum frames from the list of _difference values_ we got using the hash difference or using a threshold to save the frames.
Now are you thinking about __image hashing__ ?
> It is also called perceptual hashing, and it’s a process of constructing hash values based on the visual contents of an image. It created similar hashes for images that are similar in perception.

**Q3. What kind of image hashing algorithm are you using ?**
**Ans.** I am using a combination of these image hashing algorithms:
1. __a-Hash__: _a-hash_ or _Average hashing_ is an algorithm which uses only a few transformations. Scale the image, convert to greyscale, calculate the mean and binarize the greyscale based on the mean. Now convert the binary image into the integer.
2. __p-Hash__: _p-hash_ or _Perceptual hashing_ uses similar approach but instead of averaging relies on discrete cosine transformation
3. __d-Hash__: _d-hash_ or _Difference hashing_ works by calculating the difference (i.e., relative gradient) between adjacent pixels.

**Q4. Now what is Absolute difference + local maxima/ threshold ?**
**Ans.** : This algorithm consists of converting the image to LUV color space and then calculating the absolute difference between two consecutive frames, which will give us a list of difference values and then either we are saving the local maximum from this list or we are saving the frames using a threshold value.

**Q5. Now what is Optical Flow + local maxima/ threshold ?**
**Ans.** : This algorithm consists of calculating the _gunner farneback_ dense optical flow between two consecutive frames, and then comparing the difference value with a black image using histogram comparison which will give us a list of difference values and then either we will save the local maximum from this list or we will save the frames using a threshold value.

So you are confused about what is _Optical Flow_.

> Optical Flow is the motion of an object between two consecutive frames caused by relative motion of camera and the object.

**Q6. How are you getting the local maxima ?**
**Ans.** : Glad you asked! We get a list of difference values from the algorithm. Something like this: ```difference_arr = [44,0,32,12,0]```. This list is first smoothened, and then I calculate the relative extrema of data, which are then saved and extracted as key frames. 
> Smoothening of a list (signal) is based on the convolution of a scaled window with the signal. The signal is prepared by introducing reflected window-length copies of the signal at both ends so that boundary effects are minimized in the beginning and end part of the output signal.

**Q7. What are the different kinds of formats supported for input data ?**
**Ans.** Currently I support JSON as an input for the API.

**Q8. What if I want to see the logs of what is happening ?**
**Ans.** That too has been taken care of. Whenever the process is completed, you can go to results/run_logs/ to see the logs.

**Q9. Are you a robot ?**
**Ans.** I prefer to think of myself as your friend. Who also happens to be artificially intelligent.😃


## This Guy made me 🦸‍♂️

- Amit Joshi _aka_ A.J
