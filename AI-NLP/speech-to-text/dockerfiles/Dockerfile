FROM python:3.7.11

RUN apt-get update \
    && apt-get install -y libopenmpi-dev

WORKDIR /home/speech-to-text

COPY requirements.txt .

RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org -r requirements.txt

COPY ./ ./

RUN python manage.py migrate

EXPOSE 8000

CMD [ "python", "./manage.py", "runserver", "0.0.0.0:8000"]