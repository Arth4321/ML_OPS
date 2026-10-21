from fastapi.testclient import TestClient

#import the application object defined in main.py
from main import app 

# create reusable test client for all the functions
client = TestClient(app)

def test_health_endpoint():

    '''
      Verify that the application is running and can respond to health check request
    assert is like if else if code correct then good if not then assertion error
    '''
    response = client.get("/health")

    assert response.status_code == 200
    #checking response.body
    assert response.json() == {"status":"healthy"}

def test_valid_iris_prediction():

    '''
      verify that valid flower measurements produce a prediction
    '''
    payload = {
  "sepal_length": 1,
  "sepal_width": 1,
  "petal_length": 1,
  "petal_width": 1
    }
    #send simulated HTTP post request with JSON input 

    response = client.post("/predict", json = payload)

    assert response.status_code == 200

    #checking response.body

    # convert the json response into a python dictionary 
    result = response.json()
    
    # then check contracts : both fields must exists
    assert "Prediction_class" in result
    assert "class_name" in result
    
    #predicted class must be one of three iris class_names = ["setosa", "versicolor", "virginica"]
    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    #prediction must be 0,1,2
    assert result["Prediction_class"] in [0,1,2]

def test_iris_input():
    """
    verify that invalid inputs are reject by fast api

    """
    payload = {
      "sepal_length": -1,
      "sepal_width": 1,
      "petal_length": 1,
      "petal_width": 1
        }
        #send simulated HTTP post request with JSON input 
    
    response = client.post("/predict", json = payload)

    # FastAPI normally returns HTTP 422 for validation error
    
    assert response.status_code == 422
  