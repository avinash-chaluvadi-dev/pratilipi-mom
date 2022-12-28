insta# Spech to Text

## Getting Started

Setup project environment with python venv(https://docs.python.org/3/library/venv.html) and [pip](https://pip.pypa.io).

```cmd
$ cd speech-to-text/
$ python -m venv env
$ env/bin/activate
$ pip install -r requirements.txt
$ pre-commit install # This step is important. This will install pre-commit hook in developers system and it will run the pytest scripts before every. Which checks for pep-8 standards
$ python manage.py makemigrations
$ python manage.py migrate
$ python manage.py runserver
```
