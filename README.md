### Customer Churn Prediction Model using ANN

This project involves building a customer churn prediction model using an Artificial Neural Network (ANN) to determine whether a customer will leave (churn) or stay based on various features. We also provide a Streamlit web application that allows users to input customer details and get predictions on whether they are likely to churn.

#### **1. Data Preprocessing**
The dataset used is the **Churn_Modelling.csv** file, which contains customer data with various features such as CreditScore, Geography, Gender, Age, and so on.

##### Steps in Preprocessing:
- **Loading Data**: 
  ```python
  df = pd.read_csv('Churn_Modelling.csv')
  ```

- **Removing Unnecessary Columns**: 
  Columns like `RowNumber`, `CustomerId`, and `Surname` were removed since they don't provide useful information for our model:
  ```python
  df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
  ```

- **One-Hot Encoding**: 
  Categorical columns like `Geography` and `Gender` were encoded using one-hot encoding.
  ```python
  df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True, dtype=int)
  ```

- **Train-Test Split**:
  The data was split into training and testing sets with a test size of 20%.
  ```python
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  ```

- **Feature Scaling**: 
  StandardScaler was used to scale the features.
  ```python
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.fit_transform(x_test)
  ```

#### **2. Model Building: ANN**
We used **TensorFlow** and **Keras** to build an Artificial Neural Network for the classification task.

##### ANN Architecture:
- **Input Layer**: 11 input features.
- **Hidden Layers**: 
  - First hidden layer: 11 neurons with ReLU activation.
  - Second hidden layer: 7 neurons with ReLU activation.
- **Output Layer**: 
  - 1 neuron with Sigmoid activation for binary classification.

##### Model Summary:
```python
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=11))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

##### Model Compilation:
- Loss function: `binary_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`

```python
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
```

##### Model Training:
We trained the model for 25 epochs with 20% of the training data used as validation.
```python
history = model.fit(x_train_scaled, y_train, epochs=25, validation_split=0.2)
```

#### **3. Evaluation**
The model's performance was evaluated using accuracy and loss metrics on the training and validation data.

##### Accuracy and Loss Plot:
```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
```

##### Prediction and Threshold:
The model's output is a probability (due to Sigmoid activation). Predictions are thresholded at 0.5 to classify as 0 (No Churn) or 1 (Churn).
```python
y_pred_prob = model.predict(x_test_scaled)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)
```

##### Model Accuracy:
```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

#### **4. Deployment: Streamlit Application**
We deployed the trained ANN model using **Streamlit** to provide an interactive interface where users can input customer data and receive churn predictions.

##### Loading the Model:
```python
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
```

##### User Inputs:
The user can select or input various customer details such as `Gender`, `Geography`, `CreditScore`, `Age`, etc.

##### One-Hot Encoding for Inputs:
We manually perform one-hot encoding for the `Geography` and `Gender` inputs:
```python
if Geography == 'Germany':
    Geography_Germany = 1
    Geography_Spain = 0 
elif Geography == 'Spain':
    Geography_Germany = 0
    Geography_Spain = 1
else:
    Geography_Germany = 0
    Geography_Spain = 0
```

##### Prediction Logic:
The input data is scaled using the pre-trained scaler, and the model predicts whether the customer will churn based on the inputs.
```python
result = model.predict(input_array_scaled.reshape(1,11))
output = 1 if result > 0.5 else 0
```

##### Displaying Results:
The output is displayed as a success or warning message, with the predicted probability of churn or retention.
```python
if output:
    st.warning(f'Can Leave, with probability of {round(result[0][0] * 100, 3)}%')
else:
    st.success(f'Not Leave, with probability of {round((1 - result[0][0]) * 100, 3)}%')
```

#### **5. Saving the Model**
The trained model and scaler were saved using `pickle` to facilitate future deployment.
```python
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

## How to Run

0. **Create a new Virtual Environment**
    ```
    pip install virtualenv
    virtualenv .venv
    .venv\Scripts\activate
    ```

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/manishKrMahto/Churn-Prediction-using-ANN.git
    cd Churn-Prediction-using-ANN
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

4. **Interact with the Web**:
    - Fill information as given and enjoy

**Summery**
This documentation provides a complete guide to understanding, training, evaluating, and deploying the customer churn prediction model using ANN.

## Feedback and Suggestions
We welcome your suggestions and feedback! Feel free to open an issue or submit a pull request for improvements. You can also reach me directly at manishcode123@gmail.com.