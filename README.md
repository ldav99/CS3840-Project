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
We will be using the **Airline Satisfaction** dataset, which contains passenger feedback and service quality ratings.  
[Click here to view the dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data).

## Neural Network Model

For this project, we utilize a neural network to predict airline passenger satisfaction based on various features related to passenger demographics, travel information, service satisfaction ratings, and flight delay times.

### Model Architecture

The architecture of the neural network consists of the following layers:

1. **Input Layer**:  
   The input layer contains a number of neurons corresponding to the number of features in the dataset. These features include passenger demographics, travel information, satisfaction ratings for various services, and flight delays.

2. **Hidden Layers**:  
   We start with two hidden layers. Each hidden layer is composed of 128 neurons and uses the **ReLU** activation function. ReLU is chosen for its ability to efficiently handle non-linearity in the data and prevent vanishing gradients.

3. **Output Layer**:  
   The output layer consists of three neurons, corresponding to the three possible classes: **Satisfied**, **Neutral**, and **Dissatisfied**. We use the **Softmax** activation function to output a probability distribution across the classes, ensuring the sum of the output probabilities is 1.

### Model Hyperparameters

- **Loss Function**: Since this is a multi-class classification problem, we use **Categorical Cross-Entropy** as the loss function.
- **Metrics**: We evaluate the model’s performance using accuracy as the primary metric.
  
### Data Preprocessing

Before training the model, we preprocess the data as follows:
- **Normalization**: Continuous features, such as age, flight distance, and delay times, are normalized to a range of [0, 1] to improve model convergence.

- **Handling Missing Values**: Any missing or incomplete data points are imputed with the mean or mode of the respective feature, depending on the feature type.

### Regularization

To prevent overfitting and improve generalization, we employ **dropout** with a dropout rate of 0.2. This technique randomly deactivates neurons during training, which forces the model to learn more robust features.

### Model Training

The model is trained using a validation split to monitor its performance on unseen data during training. The training process will run for a specified number of epochs, and the learning rate will be adjusted dynamically using early stopping if the validation accuracy plateaus.
