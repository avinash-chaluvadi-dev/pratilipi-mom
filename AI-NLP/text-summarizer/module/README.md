**Module**

This app contains all the rest apis required for running the ML model for meeting summarizers.

As of now this package has been configured with below APIs

1. API for annotation adaptor:  
   The URL for this API is http://localhost:8000/module/mom/<request_id>/. Where request_id is masked request id for a meeting
2. API for Text Summarizer:  
   The URL for this API is http://localhost:8000/module/summarizer/<request_id>/. Where request_id is masked request id for a meeting


## Getting Started

Command to run the development server:

```bash
python manage.py runserver  (python manage.py runserver port_number)
```

Open [http://localhost:8000] with your browser to see the result.

## Learn More

To learn more about Python and Django take a look at the following resources:

- [Python Documentation](https://www.python.org/doc/) - learn about Python features.
- [Learn Django](https://docs.djangoproject.com/en/3.2/intro/tutorial01/) - an interactive Django tutorial.
- [Django Deployment Tutorial](https://docs.djangoproject.com/en/3.2/howto/deployment/) - Learn about Django deployment.
