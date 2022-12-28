FROM sit1-dtr.anthem.com/core/python-base-3.9:1.0.0

RUN yum update -y && yum -y install vim && yum clean -y all

WORKDIR /app/

COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

COPY . /app/
VOLUME /momefs

ENTRYPOINT ["sh", "entrypoint.sh"]
