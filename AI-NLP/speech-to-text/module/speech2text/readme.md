### SPEECH TO TEXT(S2T)
- - - -
Speech to text is a package for conversion of speech(audio) into text. 

The Package can be run in  two modes

* Pluggable Engine mode.
It can be run in three modes: 
	1. train
	2. eval
	3. infer

* Standalone testing and execution mode. It can be run in three modes:
	1. train 
	2. eval
	3. infer


###Software Requirements
- - - -
1. Python <=3.7
2. tensorflow-gpu <= 1.15(optional)
3. CUDA >= 9.0, cuDNN >= 7.0
4. Horovod >= 0.13(Optional)

###Quick start
- - - -
Package requirements can be installed using pip:
```python 
pip install -r requirements.txt
```
Speech to text package can be installed using pip: 
```python 
pip install -e .
```
If you want to run the serve component of the code, you can run using python:
```python
python api.py
```



[comment]: <> (###Config)

[comment]: <> (- - - -)

[comment]: <> (Copy the config file from example_configs folder )


### Run commands
- - - -

The script speech2text/main.py is the entry point of the package execution

#### Standalone Execution
Run the functional test for the package, to check all the components of the engine work correctly or not.
* func_test = "all" -> runs test for all the components.
* func_test = "train" -> runs the test for training component only.
* func_test = "serve" -> runs the test for serving component only.
* func_test = "eval" -> runs the test for evaluation component only.

```python
python main.py --mode=package_test --func_test=all
```

#### Pluggable Component
This mode of running the package, showcase the capability to be able to plug in the train, eval and serving 
component of this package into API integration or MLOPs engine.
* Mode="train" - run the training component of the package.
* Mode="serve" - run the serving component of the package.
* Mode="eval" - run the evaluation component of the package.

```python
python main.py --mode=train
```
