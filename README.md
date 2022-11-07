# EXPLORE AI  Academy Internship project


![Walmart](https://user-images.githubusercontent.com/55085063/190649115-42879a8f-48b3-4561-bc32-85252156f62a.jpg)

# Team-26-Developing-an-opinionated-sales-forecasting-MVP
# Opinionated Sales forecasting M.V.P.



## Overview:
Department stores like Walmart have uncountable products and money transactions every day. Because of their rapid transaction rates, keeping a balance between inventory and customer is most important. Therefore making an accurate sales prediction for different products becomes an essential need for stores to optimize profits.

Our task is to predict the product demand based on only the historical sales record.


## Data Source: 
Our data set is actual Customer and Transactional Data sourced from 10 Walmart Stores located across 3 cities namely:

California (CA- 4 stores)

Texas (TX- 3 stores)

Wisconsin (WI- 3 stores)


## Data Structure:
![overview](https://user-images.githubusercontent.com/55085063/190647282-f393e520-0af7-41eb-ab6c-08923d6cf4a2.png)

## Raw Dataset (CSV Files):
Files provided include -
calendar.csv : Contains information about the dates on which the products are sold.

sales_train_validation.csv : Contains the historical daily unit sales data per product and store [d_1 to d_1913].

sell_prices.csv : Contains information about the price of the products sold per store and date.

sales_train_evaluation.csv : In addition to the historical daily unit sales data per product and store in sales_train_validation.csv [i.e. d_1 to d_1913], it also contains columns for d_1914 to d_1941 (i.e. 28 days).

Knowledge:

sales_train_validation has data of 30490 rows for all stores and has columns (1919).

The first column id is actually a combination of all the previously mentioned ids and also added with a string validation. 

The rest of the columns are from d_1 to d_1913 which shows the number of units sold on that particular day(column) of a particular product id(row). 

The calendar file has special events and SNAP columns to show the variation in sales on particular days. 

The sell_prices file has prices for each product(sell_price) in a particular week of a year using unique id — wm_yr_wk.

## Understanding the dataset.

The following table lists down various attributes found in the dataset along with their descriptions:-

## Column Descriptions for calendar.csv


> | Column Name      | Descriptions                                                                                             |
> | ---------------- | -------------------------------------------------------------------------------------------------------- |                                      
> | date             | Date                                                                                                     |
> |                  |                                                                                                          |                    
> | wm_yr_wk         | Some sort of combination of year and week                                                                |
> |                  |                                                                                                          |                    
> | weekday          | Day of the week                                                                                          |
> |                  |                                                                                                          |
> | wday             | weekday encoded                                                                                          |
> |                  |                                                                                                          |                    
> | month            | month of the year                                                                                        |
> |                  |                                                                                                          |                    
> | d                | signifying which day it is in absolute term. All values are unique                                       |
> |                  |                                                                                                          |                    
> | event_name_1     | Name of the primary event, e.g. Super Bowl etc. 29 events and 1 null                                     |
> |                  |                                                                                                          |                    
> | event_type_1     | Type of event. Whether it is Sporting or Cultural or National or Religious                               |
> |                  |                                                                                                          |
> | event_name_2     | Second event, if any. Only 5 values, rest is null                                                        |
> |                  |                                                                                                          |                    
> | event_type_2     | Second event, if any. Only 5 values, rest is null                                                        |
> |                  |                                                                                                          |                    
> | snap_CA          | Whether SNAP food stamp is there or not for California state                                             |
> |                  |                                                                                                          |
> | snap_TX          | Whether SNAP food stamp is there or not for Texas state                                                  |
> |                  |                                                                                                          |                    
> | snap_WI          | Whether SNAP food stamp is there or not for Wisconsin state                                              |



## Column description for sale_validation

> | Column Name      | Descriptions                                                                                             |
> | ---------------- | -------------------------------------------------------------------------------------------------------- |                                      
> | id               | combination of below IDs and a validation flag                                                           |
> |                  |                                                                                                          |                    
> | item_id          | Item ID                                                                                                  |
> |                  |                                                                                                          |                    
> | dept_id          | Department ID                                                                                            |
> |                  |                                                                                                          |
> | store_id         | Store ID                                                                                                 |
> |                  |                                                                                                          |                    
> | state_id         | State ID                                                                                                 |
> |                  |                                                                                                          |                    
> | d_1 to d_1913    | day wise units sold                                                                                      |


## Column description for sell prices csv

> | Column Name      | Descriptions                                                                                             |
> | ---------------- | -------------------------------------------------------------------------------------------------------- |                                      
> | store_id         | maps to store_id of Sales Train Validation table                                                         |
> |                  |                                                                                                          |                    
> | item_id          | Item ID                                                                                                  |
> |                  |                                                                                                          |                    
> | wm_yr_wk         | Some sort of combination of year and week  (Same as described above)                                     |
> |                  |                                                                                                          |
> | sell prices      | Price during that particular wm_yr_wk                                                                    |


## Objective:
Our main aim is to predict the products sales for the next 28 days.

## Steps for Developing our model:
1. Data Ingestion:- Loading the raw data (i.e. our .csv files) into a Jupyter notebook making use of the Pandas library.

2. Data Quality Testing:- A data verification process done with programming code. Programmer verifies data types of fields, length of characters, formats, and whether the values falls within an acceptable range.

3. Exploratory Data Analysis:- Analysing the Data via Histograms, bar charts, pie charts etc to further understand our data.

4. Feature Engineering:-  Manipulation — addition, deletion, combination, mutation — of our data set to improve machine learning model training, leading to better performance and greater accuracy. Effective feature engineering is based on sound knowledge of the business problem and the available data sources.

5. Developing our Model:- The best performing model was the LightGBM Model amongst a list of time series models. This was used to forecast the sales for the 28 days.

6. Confidence Intervals:- One of the deliverables of this project, we predicted future sales per item per store (using LightGBM Model) with uncertainty estimates for the next 28 days, i.e. per day predict the median, 50%, 67%, 95% and 99% confidence intervals.

7. Building a front-end (Client side interactive interface):- We achieved this using the Streamlit framework. Streamlit is an open source app framework in Python language that helps us create web apps for data science and machine learning in a short time.

8. Deployment to the cloud:- Moving the entire solution (Dataset, Jupyter Notebooks, Streamlit Application and all other resources associated with the project) to the cloud. For this an object storage (S3 bucket) was launched on A.W.S. and all resources were saved inside the bucket. Then an E.C.2 instance was spun up, mounted onto the S3 bucket and finally the Streamlit app was loaded onto the E.C.2 instance along with all required dependencies.  
 
