**Rest API**

This app contains all the rest apis required for processing files.

## Getting Started

First, go to `rest_api` directory, and run below command to install required dependencies.

```bash
pip install -r rest_api/requirements.txt
```

Command to run the development server:

```bash
python manage.py runserver  (python manage.py runserver port_number)
```

Open [http://localhost:8000](http://localhost:8000) with your browser to see the result.

## Learn More

To learn more about Python and Django take a look at the following resources:

-   [Python Documentation](https://www.python.org/doc/) - learn about Python features.
-   [Learn Django](https://docs.djangoproject.com/en/3.2/intro/tutorial01/) - an interactive Django tutorial.
-   [Django Deployment Tutorial](https://docs.djangoproject.com/en/3.2/howto/deployment/) - Learn about Django deployment.

## Function details
- FileUpload
- FileUploadSerializer
- insert_request_id
- UploadAPIViewTest
- test_file_upload
- FileUploadView