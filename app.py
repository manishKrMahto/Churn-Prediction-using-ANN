import streamlit as st
import pickle 
import numpy as np 

model = pickle.load(open('model.pkl' ,'rb'))
scaler = pickle.load(open('scaler.pkl' ,'rb'))

st.title('Customer Churn prediction')

Gender = st.selectbox('Choose you Gender ' , ['Male' ,'Female'])

Geography = st.selectbox('Choose your Geography' , ['France' ,'Germany', 'Spain'])

CreditScore = st.text_input('Etner Credit Score : ')

Age = st.text_input('Etner Age : ')

Tenure = st.text_input('Etner Tenure : ')

Balance = st.text_input('Etner Balance : ')

NumOfProducts = st.text_input('Etner NumOfProducts : ')

HasCrCard = st.text_input('Etner HasCrCard : ')

IsActiveMember = st.text_input('Etner IsActiveMember : ')

EstimatedSalary = st.text_input('Etner EstimatedSalary : ')


# OneHotEncoding using manually Because it is simple so not using it's object

# Geography variable 
if Geography == 'Germany':
    Geography_Germany = 1
    Geography_Spain = 0 
elif Geography =='Spain':
    Geography_Germany = 0 
    Geography_Spain = 1
else : # for France
    Geography_Germany = 0
    Geography_Spain = 0
    
# Gender Variable 
if Gender == 'Male':
    Gender_Male = 1
else :
    Gender_Male = 0
    

status = st.button('What ?')

if status :
    # converting all inputs into a list
    input_lst = [int(CreditScore) , int(Age), int(Tenure), float(Balance), int(NumOfProducts), int(HasCrCard),int(IsActiveMember),float(EstimatedSalary),Geography_Germany, Geography_Spain,Gender_Male]

    # now converting this list into NumPy Array 
    input_array = np.array(input_lst)

    # scaling values of input array 
    input_array_scaled = scaler.transform(input_array.reshape(1,11))
    
    # predicting the result using ANN model 
    result = model.predict(input_array_scaled.reshape(1,11))

    # result contain probabilties (loss function is Sigmoid) so taking a Thresold of 0.5 to  decide
    output = 1 if result > 0.5 else 0

    if output :
        st.warning(f'Can Leave , with probability of {round(result[0][0] * 100, 3)}%')
    else :
        st.success(f'Not Leave, with probability of {round((1 - result[0][0]) * 100, 3) }%')
