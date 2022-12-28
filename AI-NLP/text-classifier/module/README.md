**Module**

This app contains all the rest apis required for running the ML model for meeting classifiction.

As of now this package has been configured with below APIs

1. API for Named Entity Recognition:  
   The URL for this API is http://localhost:8000/module/ner/<request_id>/. Where request_id is masked request id for a meeting
2. API for Deadline Escalation Model:  
   The URL for this API is http://localhost:8000/module/allocator/<request_id>/. Where request_id is masked request id for a meeting
3. API for Recommendation Classifier Model:  
   The URL for this API is http://localhost:8000/module/recommendation/<request_id>/. Where request_id is masked request id for a meeting

## Getting Started

Command to run the development server:

```bash
python manage.py runserver  (python manage.py runserver port_number)
```

Open [http://localhost:8000](http://localhost:8000) with your browser to see the result.

## Learn More

To learn more about Python and Django take a look at the following resources:

- [Python Documentation](https://www.python.org/doc/) - learn about Python features.
- [Learn Django](https://docs.djangoproject.com/en/3.2/intro/tutorial01/) - an interactive Django tutorial.
- [Django Deployment Tutorial](https://docs.djangoproject.com/en/3.2/howto/deployment/) - Learn about Django deployment.

## Funtion Detail

**ner_api_view**: This is an api wrapper over ner model

**allocator_api_view**: This is an api wrapper over Deadline Escalation model

**recommendation_api_view**: This is an api wrapper over Recommendation Classifier model

**marker_api_view**: This is an api wrapper over Marker model

- marker_classifier

# tests.py Function details

- ClassifierViewTest:
  - test_invalid_request_id
  - test_marker_classifier
  - test_validate_inputfeed

**label_api_view**: This is an api wrapper over Label model

**sentiment_api_view**: An API wrapper over sentiment classifer

**escalation_api_view**: An API wrapper over esclation classifer
