# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 06:35:40 2022

@author: stevi
"""
import streamlit as st
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Setting my model's title
st.title("Breezie WebApp💫💫.")

def main():
    activities=['EDA','Visualisation','model','About us']
    option=st.sidebar.selectbox("Select option:",activities)
    
    if option=='EDA':
        st.subheader("Exploratory Data Analysis")
        
        data=st.file_uploader("Upload dataset:",type=['csv'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(10))
            
            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("Display Null values"):
                st.write(df.isnull().sum())
            if st.checkbox("Select multiple columns"):
                selected_columns=st.multiselect("Select preffered columns, N/B Let target be  the last column of selection:",df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            
            if st.checkbox("Display summary"):
                st.write(df1.describe().T)
            
        
    elif option=='Visualisation':
        st.subheader("Data visualisation")
        
        data=st.file_uploader("Upload dataset:",type=['csv'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(10))
            
        if st.checkbox("Select multiple columns to plot"):
            selected_columns=st.multiselect("Select your prefferd columns",df.columns)
            df1=df[selected_columns]
            st.dataframe(df1)
            
        if st.checkbox("Display barchart"):
            st.bar_chart(df1)
        if st.checkbox("Dispay line chart"):
            st.line_chart(df1)
        if st.checkbox("Display Pie Chart"):
            all_columns=df.columns.to_list()
            pie_columns=st.selectbox("Select column to display", all_columns)
            pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(pieChart)
            st.pyplot()
            
            
    elif option=="model":
        st.subheader("Model Building")
        
        data=st.file_uploader("Upload dataset:",type=['csv'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(10))
            
            if st.checkbox("Select multiple columns"):
                new_data=st.multiselect("Select your preffered columns, N/B Let your target column to be the last to be selected", df.columns)
                df1=df[new_data]
                st.dataframe(df1)
                
                #Dividing the data into x and y variables
                x=df1.iloc[:,0:-1]
                y=df1.iloc[:,-1]
                
            seed=st.sidebar.slider('seed',1,200)
            Classifier_name=st.sidebar.selectbox("Select your preffered Classifier", ('KNN','SVM','LR','naive_bayes'))
            
            
            def add_parameter(name_of_clf):
                params=dict()
                if name_of_clf=='SVM':
                    C=st.sidebar.slider('C',0.01,20.0)
                    params['C']=C
                else:
                    name_of_clf=='KNN'
                    K=st.sidebar.slider('K',1,20)
                    params['K']=K
                    return params
            params=add_parameter(Classifier_name)
            
            
            def get_classifier(name_of_clf,params):
                clf=None
                if name_of_clf=='SVM':
                    clf=SVC(C= params['C'])
                elif name_of_clf=='KNN':
                    clf=KNeighborsClassifier(n_neighbors= params['K'])
                elif name_of_clf=='LR':
                    clf=LogisticRegression()
                else:
                    st.warning("Select your preffered Algorithm")
                    return clf
                
            clf=get_classifier(Classifier_name, params)
            
            #Diving the data into training and testing data
            x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=seed)
            clf.fit(x_train,y_train)
            
            y_pred=clf.predict(x_test)
            st.write('Predictions:',y_pred)
            
            accuracy=accuracy_score(y_test, y_pred)
            
            st.write("Name of Classifier:",Classifier_name)
            st.write("Accuracy:",accuracy)
            
            

   
        

    else:
        st.write(""" ## This is a well designed and interactive webapp for machine learning deployment projects. Feel free to use it.""")
        st.write(""" ## Created and designed by Breezie Foundation.""")
        st.write(""" ## Contact: 0700 495 575. """)
        st.write(""" ## Email: steviebreezie35@gmail.com. """)
        
        
        
if __name__ == '__main__':
    main()
