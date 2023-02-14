import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('model.pkl','rb'))
encoder = pickle.load(open('target_encoder.pkl','rb'))
transformer = pickle.load(open('transformer.pkl','rb'))

st.title("CREDIT CARD DEFAULT PREDICTION")


with st.sidebar:
    st.info("""INFO REGARDING REPAYMENT STATUS
            -1=pay duly,
            1=payment delay for one month,
            2=payment delay for two months,
            .
            .
            .
            8=payment delay for eight months, 
            9=payment delay for nine months and above
            """)

Gender = st.selectbox("Gender",['Male','Female'])

age = st.text_input('Enter Age', 18)
age = int(age)

Marrital_Status = st.selectbox("Marrital Status",['Married','Unmarried'])

Education = st.selectbox("Education Level",['High School','Under Graduate','Post Graduate'])

Limit_balance= st.text_input("Limit balance",1000)
Limit_balance=int(Limit_balance)

st.subheader('REPAYMENT STATUS')

April=st.selectbox("April",['-1','1','2','3','4','5','6','7','8','9'])

May=st.selectbox("May",['-1','1','2','3','4','5','6','7','8','9'])

June=st.selectbox("June",['-1','1','2','3','4','5','6','7','8','9'])

July=st.selectbox("July",['-1','1','2','3','4','5','6','7','8','9'])

August=st.selectbox("August",['-1','1','2','3','4','5','6','7','8','9'])

Sept=st.selectbox("September",['-1','1','2','3','4','5','6','7','8','9'])


st.subheader("BILL AMOUNTS FOR BELOW MONTHS")

Bill_april= st.text_input("Bill amount for April",1000)
Bill_april=int(Bill_april)

Bill_may= st.text_input("Bill amount for May",1000)
Bill_may=int(Bill_may)

Bill_june= st.text_input("Bill amount for June",1000)
Bill_june=int(Bill_june)

Bill_july= st.text_input("Bill amount for July",1000)
Bill_july=int(Bill_july)

Bill_august= st.text_input("Bill amount for August",1000)
Bill_august=int(Bill_august)

Bill_sept= st.text_input("Bill amount for September",1000)
Bill_sept=int(Bill_sept)

st.subheader("PAY AMOUNTS FOR BELOW MONTHS")


pay_april= st.text_input("Paying amount for April",1000)
pay_april=int(pay_april)

pay_may= st.text_input("Paying amount for May",1000)
pay_may=int(pay_may)

pay_june= st.text_input("Paying amount for June",1000)
pay_june=int(pay_june)

pay_july= st.text_input("Paying amount for July",1000)
pay_july=int(pay_july)

pay_august= st.text_input("Paying amount for August",1000)
pay_august=int(pay_august)

pay_sept= st.text_input("Paying amount for September",1000)
pay_sept=int(pay_sept)


l={}

l['LIMIT_BAL']= Limit_balance
l['SEX']=Gender
l["EDUCATION"]= Education
l['MARRIAGE']= Marrital_Status
l['AGE']= age
l['PAY_0']=Sept
l['PAY_2']=August
l['PAY_3']=July
l['PAY_4']=June
l['PAY_5']=May
l['PAY_6']=April
l['BILL_AMT1']=Bill_sept
l['BILL_AMT2']=Bill_august
l['BILL_AMT3']=Bill_july
l['BILL_AMT4']=Bill_june
l['BILL_AMT5']=Bill_may
l['BILL_AMT6']=Bill_april
l['PAY_AMT1']=pay_sept
l['PAY_AMT2']=pay_august
l['PAY_AMT3']=pay_july
l['PAY_AMT4']=pay_june
l['PAY_AMT5']=pay_may
l['PAY_AMT6']=pay_april


df=pd.DataFrame(l,index=[0])



df['SEX'] = df['SEX'].map({'Male':1, 'Female':0})
df['MARRIAGE']= df['MARRIAGE'].map({'Married':1, 'Unmarried':0})
df['EDUCATION']=df['EDUCATION'].map({'High School':0, 'Under Graduate':1, 'Post Graduate':2})

df= transformer.transform(df)

y_pred= model.predict(df)

if st.button("PREDICT"):
    st.header(f"{round(y_pred[0],2)}")
