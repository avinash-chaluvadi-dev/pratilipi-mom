# Classifier Models

## Getting Started

Setup project environment with python venv(https://docs.python.org/3/library/venv.html) and [pip](https://pip.pypa.io).

```cmd
$ cd classifier-models/
$ python -m venv env
$ env/bin/activate
$ pip install -r requirements.txt
$ pre-commit install # This step is important. This will install pre-commit hook in developers system and it will run the pytest scripts before every. Which checks for pep-8 standards
$ python manage.py migrate
$ python manage.py runserver

```

**Consolidated MODEL's**

This sensemaker shell is the consolidated shell for all the below classification models:

-Sentiment Classifier
-Deadline and Escalation Classifier
-Recommendation Classifier
-Meeting Type Classifier
-Marker Classifier
-Label Classifier
-Topic Extraction
-Headline Generation
-Text Summarizer
-Minute Of Meeting Generation

As of now this package has been configured with below classification models
-Marker Classifier
-Label Classifier

**SENSEMAKER SHELL :: 10-01-2021**

**Marker classifier**
Marker classifier GET API http://localhost:8000/module/markercls/<request_id>/.
Where request_id is masked request id for a meeting
Getting Started
Command to run the development server:

**Label classifier**
Label classifier GET API http://localhost:8000/module/labelcls/<request_id>/.
Where request_id is masked request id for a meeting
Getting Started
Command to run the development server:

python manage.py runserver (python manage.py runserver port_number)
Open http://localhost:8000 with your browser to see the result.

Learn More about branching policy and hive:
https://confluence.anthem.com/display/LDTS/The+Bitbucket+Branching+Strategy
