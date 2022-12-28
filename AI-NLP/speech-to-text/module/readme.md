# Speaker Diarization Package Instruction Set


### Overview 

The Package can be run in major two modes

* Pluggable Engine mode.
It can be run in three modes: 
	1. Train
	2. Evaluate
	3. test

* Standalone testing and execution mode. It can be run in three modes:
	1. train 
	2. Evaluate
	3. Test

### Environment Setup

Check requirements.txt for the same.

### File Setup

* Place the package testing csv in the speaker_diarization/datasets/gt_csv.csv.
* Place the recording for testing/inference in the speaker_diarization/datasets/.
* Place the UISRNN model under speaker_diarization/models/uisrnn.
* Place the Vggvox model under speaker_diarization/models/vggvox.

### Parameters Setup

The config file needs to be setup with some parameters.

* TEST_FILE - It is the test recording path.
* GT_CSV - The ground truth file for the validation of the system output. Default name is gt_csv.csv
* RUN_MODE - The parameter to use the in the api interface. DEFAULT - "infer".
	

### Run commands

The script speaker_diarization/main.py is the entry point of the package execution

#### Standalone Execution
Run the functional test for the package, to check all the components of the engine work correctly or not.
* func_test = "all" -> runs test for all the components.
* func_test = "train" -> runs the test for training component only.
* func_test = "serve" -> runs the test for serving component only.
* func_test = "eval" -> runs the test for evaluation component only.

 `python main.py --mode=package_test --func_test=all`

#### Pluggable Component
This mode of running the package, showcase the capability to be able to plug in the train, eval and serving 
component of this module into API integration or MLOPs engine.
* Mode="train" - run the training component of the package.
* Mode="serve" - run the serving component of the package.
* Mode="eval" - run the evaluation component of the package.

`python main.py --mode=train`

#### Results and Logs

* The training results are stored in /models timestamped with date and time, when the package is run in pluggable mode 
whereas no model is stored in the package testing mode.
* The execution run logs after each run can be found in results/store_logs, timestamped with time of execution and date.
* The output chunks of the serving component can be found in the results/store_segs.
