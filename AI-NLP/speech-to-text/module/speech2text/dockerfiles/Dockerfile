FROM python:3.7
WORKDIR /root/src
COPY C:\Users\AG84959\Downloads\Pratilipi\setup.py .
COPY C:\Users\AG84959\Downloads\Pratilipi\speech2text ./speech2text

RUN /bin/bash -c "pip install -e . \
    && cd speech2text \
    && pip install -r requirements.txt"

#CMD [ "python", "api.py" ]