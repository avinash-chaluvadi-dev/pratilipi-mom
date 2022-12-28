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
- RequestLedgerAdmin
- upload_to
- FileUpload
- RequestLedger
- FileUploadSerializer
- FileSerializer
- insert_request_id
- test_readme_exists
- test_function_details_in_readme
- test_readme_contents
- test_readme_file_for_formatting
- test_indentations
- test_function_name_had_cap_letter
- UploadAPIViewTest
- UploadAPITestCasesWithIntake
- test_file_upload
- test_file_upload_without_intake
- create
- FileUploadView
