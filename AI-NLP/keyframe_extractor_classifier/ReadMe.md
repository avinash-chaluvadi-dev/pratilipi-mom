# Keyframe Extraction Classification Model merged with Keyframe extractor and Keyframe label classifier

## Getting Started

Setup project environment with python venv(https://docs.python.org/3/library/venv.html) and [pip](https://pip.pypa.io).

```cmd
$ cd KeyframeExtraction/
$ python -m venv env
$ env/bin/activate
$ pip install -r requirements.txt
$ pre-commit install # This step is important. This will install pre-commit hook in developers system and it will run the pytest scripts before every. Which checks for pep-8 standards
$ python manage.py migrate
$ python manage.py runserver

```

```
If you face no module found errors please follow the bellow commands to run the model

$ cd KeyframeExtraction/
$ cd sense_maker_shell/
$ cd module/
$ pip install -e . # run this command, after that cd..
$ cd../.. # roll back to sense_maker_shell folder and 
$ python manage.py migrate
$ python manage.py runserver

```
