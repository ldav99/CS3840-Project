# CS3840-Project

## Team Members
- *Luke Davidson*
- *Monica Brutto*
- *Sam Webster*

# Applied Machine Learning Final Project Outline

The goal of this project is to take the information that we have acquired so far in this class and train a model of our choosing.
This document goes over the procedures and steps taken to create this project.

## Objectives

### Domain: Airline Passenger Experience 

This dataset falls within the domain of customer satisfaction analysis for the airline industry. Airlines constantly seek to improve customer experience by analyzing feedback from passengers. Understanding the factors that contribute to passenger satisfaction can help airlines enhance their services, optimize resource allocation, and ultimately improve customer retention.

### Potential Benefits
- Identifying key factors that influence passenger satisfaction.  
- Predicting whether a passenger will be satisfied or dissatisfied based on given features.  
- Providing actionable insights to airlines for service improvement.  
- Enhancing customer retention by improving areas with low satisfaction scores.  

### Project Goals

The goal of this project is to build a machine learning model that predicts passenger satisfaction based on survey responses and flight-related data. By analyzing which features contribute most to satisfaction, the airline industry can optimize customer experience strategies. 

### Dataset Features

### Passenger Demographics & Travel Information 
 - Gender: Passenger’s gender (Male, Female). 
 - Customer Type: Classification of the passenger as either a Loyal Customer or Disloyal Customer. 
 - Age: Passenger’s age in years. 
 - Type of Travel: Purpose of the flight, categorized as either Business Travel or Personal Travel. 
 - Class: Ticket class of the passenger (Business, Economy (Eco), or Economy Plus (Eco Plus)). 
 - Flight Distance: Distance traveled on the flight (measured in miles or kilometers). 

 ### Service Satisfaction Ratings (1-5 Scale, unless specified)
 - Inflight WiFi Service: Satisfaction with in-flight WiFi connectivity (0 = Not Applicable, 1-5 = Satisfaction Level). 
 - Departure/Arrival Time Convenience: Rating for the convenience of departure and arrival times. 
 - Ease of Online Booking: Passenger's satisfaction with the airline's online booking process. 
 - Gate Location: Satisfaction with the accessibility and location of the boarding gate. 
 - Food & Drink: Rating of the quality and variety of food and beverages offered. 
 - Online Boarding: Satisfaction with the online check-in and boarding process. 
 - Seat Comfort: Passenger's rating of the comfort level of their seat. 
 - Inflight Entertainment: Satisfaction with onboard entertainment options (movies, music, etc.). 
 - On-board Service: Rating of overall service provided by the flight crew. 
 - Leg Room Service: Satisfaction with the amount of legroom available. 
 - Baggage Handling: Passenger rating of how well baggage was managed. 
 - Check-in Service: Satisfaction with the efficiency and ease of the check-in process. 
 - Inflight Service: Overall rating of the in-flight experience and hospitality. 
 - Cleanliness: Rating of the cleanliness of the airplane. 

### Delay Information (Numerical Values in Minutes) 
- Departure Delay in Minutes: Amount of time the flight was delayed before departure. 
- Arrival Delay in Minutes: Amount of time the flight was delayed upon arrival. 

## Target Variable 
- Satisfaction: Passenger's overall experience with the airline, classified as: 
    - Satisfied 
    - Neutral
    - Dissatisfied 

### Dataset Being Used  
We will be using the **[Airline Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data)** dataset, which contains passenger feedback and service quality ratings.  

## Neural Network Model

For this project, we utilize a neural network to classify airline passenger satisfaction based on various features related to passenger demographics, travel information, service satisfaction ratings, and flight delay times.

### Model Architecture

The architecture of the neural network consists of the following layers:

1. **Input Layer**:  
   The input layer contains a number of neurons corresponding to the number of features in the dataset.

2. **Hidden Layers**:  
  We use one hidden layer with n-amount of neurons and a linear activation function. This allows the model to capture linear relationships in the data.

3. **Output Layer**:  
   The output layer consists of a single neuron with a sigmoid activation function. This will output the probability of the passenger being satisfied (1) or not satisfied (0), making it a binary classification problem.

### Model Training
The model is trained using a validation split to monitor its performance on unseen data during training. The training process will run for a specified number of epochs, and the learning rate will be adjusted dynamically using early stopping if the validation accuracy plateaus.
