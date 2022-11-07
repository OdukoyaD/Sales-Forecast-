import numpy as np
import pandas as pd
import pickle 
import streamlit as st

import os 
os.system('sudo pip install scikit-learn')
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
import random 
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

#loading our saved model
#evaluation_data = pd.read_csv('s3://intern-t26-2201-sales-forecasting/data/sales_train_evaluation.csv')

#lightGBM_model = pickle.load(open('s3://intern-t26-2201-sales-forecasting/scripts/LightGBM_model.pkl', 'rb'))

#linear_submit = pd.read_csv('C:/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/submit_Linear_Regression_.csv')

lightGBM_submit = pd.read_csv('s3://intern-t26-2201-sales-forecasting/scripts/submit_LGBM_Regressor_.csv')


def main():

    page_options = ["About","EDA","Sale forecasting","Feedback",'Documentation']
    st.sidebar.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/walmart_img.jpg',use_column_width=True)

    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Sale forecasting":
        # Header contents
        st.write('# Walmart Sale forecasting model')
        select_model = st.selectbox('Model', ['Select model','LightGBM Regressor'])
        if select_model == 'LightGBM Regressor':
            st.write('Predictions when using lightGBM regression model')
            select_state = st.selectbox('select state',['select state','California', 'Texas', 'Wisconsin'])
            if select_state == 'California':
                select_store = st.selectbox('select store',['select store','CA_1','CA_2','CA_3','CA_4'])
                if select_store == 'CA_1':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_1') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_1') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_1') & (lightGBM_submit['cat_id'] == 'FOODS')].head())
                elif select_store == 'CA_2':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_2') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_2') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_2') & (lightGBM_submit['cat_id'] == 'FOODS')].head())    
                elif select_store == 'CA_3':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_3') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_3') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_3') & (lightGBM_submit['cat_id'] == 'FOODS')].head())
                elif select_store == 'CA_4':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_4') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_4') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'CA_4') & (lightGBM_submit['cat_id'] == 'FOODS')].head()) 
            elif select_state == 'Texas':
                select_store = st.selectbox('select store',['select store','TX_1','TX_2','TX_3'])
                if select_store == 'TX_1':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_1') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_1') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_1') & (lightGBM_submit['cat_id'] == 'FOODS')].head())
                elif select_store == 'TX_2':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_2') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_2') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_2') & (lightGBM_submit['cat_id'] == 'FOODS')].head())    
                elif select_store == 'TX_3':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_3') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_3') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'TX_3') & (lightGBM_submit['cat_id'] == 'FOODS')].head())       
            elif select_state == 'Wisconsin':
                select_store = st.selectbox('select store',['select store','WI_1','WI_2','WI_3'])
                if select_store == 'WI_1':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_1') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_1') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_1') & (lightGBM_submit['cat_id'] == 'FOODS')].head())
                elif select_store == 'WI_2':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_2') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_2') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_2') & (lightGBM_submit['cat_id'] == 'FOODS')].head())    
                elif select_store == 'WI_3':
                    select_category = st.selectbox('Select category',['select category','HOBBIES','HOUSEHOLD','FOODS'])
                    if select_category == 'HOBBIES':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_3') & (lightGBM_submit['cat_id'] == 'HOBBIES')].head())
                    elif select_category == 'HOUSEHOLD':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_3') & (lightGBM_submit['cat_id'] == 'HOUSEHOLD')].head())
                    elif select_category =='FOODS':
                        st.dataframe(lightGBM_submit.loc[(lightGBM_submit['store_id'] == 'WI_3') & (lightGBM_submit['cat_id'] == 'FOODS')].head())

    if page_selection == "About":
       st.write('# Walmart:Developing-an-opinionated-sales-forecasting-MVP')
       st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/walmart_img.jpg',use_column_width=True)
       st.write('## EXPLORE AI Academy Internship project')
       st.write('### Overview:')
       st.markdown('Department stores like Walmart have uncountable products and money transactions every day. Because of their rapid transaction rates, keeping a balance between inventory and customer is most important. Therefore making an accurate sales prediction for different products becomes an essential need for stores to optimize profits.Our task is to predict the product demand based on only the historical sales record.')
       st.write('### Data source:')
       st.markdown('Our data set is actual Customer and Transactional Data sourced from 10 Walmart Stores located across 3 cities namely:')
       st.markdown('California (CA- 4 stores)')
       st.markdown('Texas (TX- 3 stores)')
       st.markdown('Wisconsin (WI- 3 stores)')
       st.write('### Objective:')
       st.markdown('Develop an accurate forecasting approach to predict future sales per item per store with uncertainty estimates for the next 28 days, i.e. per day predict the median, 50%, 67%,95%, and 99%, confidence intervals.')
       st.markdown('○ The solution should be able to predict for individual items as well as aggregate item categories.')
       st.markdown('○ The pipeline should be robust to sporadic sales.')
       st.markdown('○ The pipeline should consider things like historic sale prices, special events throughout the year, promotions, etc. Future work can focus on extending the solution to also consider other relevant external datasets.')
    if page_selection == "EDA":
            st.header('Exploratory Data Analysis')
            st.write('Exploratory Data Analysis shows the distribution of the Data used in the creation of the recommender system model. This distributions are available after the cleaning and the preprocessing of the data. The graphs below shows the various distribution plots gained after cleaning the data.')
            select_option = st.selectbox('Select option', ['select option', 'Sales per year', 'Income per state','Monthly sales','Daily sales', 'Department sales'])
            
            if select_option == 'Sales per year':
                st.subheader('Sales per year')
                st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/%yearly sale.jpg')
                st.info('This shows sales per year(%).there is rapid increase between 2011 -2013 with a slight decrease in 2014 which later has a high increase in 2015.Sales rapidly drop in year 2016 because we do not have a complete dataset in that year(2016).') 
            
            if select_option == 'Income per state':
                st.subheader('Income per state')
                st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/income per state.jpg')
                st.info('Show the income generated per state within the given years.') 
                st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/sale per state.jpg')
                st.info('A display of the  snap sales per state.') 

            if select_option == 'Monthly sales':
                st.subheader('monthly sales ')
                st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/%monthly sale.jpg') 
                st.success('The graph shows monthly sales. There is a sales rise from Jan to March.there after falls till June. March, April, May have highest sales among all months.')
            
            if select_option == 'Daily sales':
                st.subheader('Daily sales ')
                st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/sales per day.jpg') 
                st.success('Illustrates the sales per day of the week') 
                st.info('Sales fall from Monday to Wednesday & have a rise from Thursday to Saturday with a little fall on Sunday. Sales are higher during weekends as compared to non-weekends.') 
            
            if select_option == 'Department sales':
                st.subheader('department sales')
                st.write('Walmart has three cstegory items i.e,food, hobbies and household.')
                st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/income per dept.jpg')
                st.success('Food item generated more income meaning it sold out more than other department.Household item generated half of what food item made in income meaning it is the second on the list.Hobbies is the least when compared with Food and Household in sale/income yet it did 12% of the entire sales..')
                st.image('/Users/Lenovo/Downloads/coursework_docs/Team_26_Developing_an_opinionated_sales_forecasting_MVP/dept sale.jpg')
                st.success('the daily distribution of sales per day per category. we can say that food categories generally are highly in sale and we also notice Saturday and Sunday recorded the higest sales for all categories.')
             
    if page_selection == "Feedback":
        
        #st.session_state;
        with st.container():
            
            name = st.text_input('Name')
            mail = st.text_input('Email')
            phone = st.text_input('Phone Number')
            
            #radio buttons
            feed_radio = st.radio('Select an option',('Feedback','Contact Us','Other'),key='radio_option')         
            
            if feed_radio == 'Other':
                subject = st.text_input('Subject') 
            
            message = st.text_area('Message')
            
            if st.button('Submit'):
                if feed_radio == "Feedback":
                    st.success('Thank you for your {}'.format(feed_radio))

                elif feed_radio == 'Contact Us':
                    st.success('Thank you for contacting us')

                else:
                    st.success('Thank you, Your {} has been logged'.format(subject))       




        




if __name__ == '__main__':
    
    main()



