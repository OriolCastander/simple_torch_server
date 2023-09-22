A proof of concept simple python server that serves a torch model.

This specific model predicts the temperature in New York (in fahrenheit) on a specific day, in a specific month on a specific hour.
To obtain your prediction, hit the /predict endpoint with the day, month and time as query parameters (or put them in the url: /predict?day=5&month=10&time=14)