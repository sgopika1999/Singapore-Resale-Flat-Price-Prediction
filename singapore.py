import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle


file_paths=[r"C:\\Users\\prave\\Downloads\\Flat1990-1999.csv",r"C:\\Users\\prave\\Downloads\\Flat2000-Feb2012.csv",
        r"C:\\Users\\prave\\Downloads\\FlatMar2012-Dec2014.csv",r"C:\\Users\\prave\\Downloads\\FlatJan2015-Dec2016.csv",
        r"C:\\Users\\prave\\Downloads\\FlatJan2017onwards.csv"]
dataframes = [pd.read_csv(file, encoding='latin1') for file in file_paths]
df=pd.concat(dataframes,ignore_index=True)
df.drop(columns=["remaining_lease"],inplace=True)
df.duplicated().sum()
df=df.drop_duplicates()
df.reset_index(drop=True, inplace=True)
df['month']=pd.to_datetime(df['month'],format="%Y-%m")
df['lease_commence_date']=pd.to_datetime(df['lease_commence_date'],format="%Y")
df["Resale_year"]= df["month"].dt.year
df["lease_commence_year"]=df["lease_commence_date"].dt.year

df1=df.drop(columns=["month","street_name","lease_commence_date"])

numerical_cols = ["floor_area_sqm","resale_price","Resale_year","lease_commence_year"]
for col in numerical_cols:
    data = df1[col]
    percentile25 = np.nanpercentile(data, 25)
    percentile75 = np.nanpercentile(data, 75)
    iqr = percentile75 - percentile25

    # Step 4: Calculate lower and upper bounds for capping
    lower_limit = percentile25 - 1.5 * iqr
    upper_limit = percentile75 + 1.5 * iqr

    # Step 5: Cap values outside the lower and upper bounds
    df1[col] = np.where(df1[col] > upper_limit, upper_limit, df1[col])
    df1[col] = np.where(df1[col] < lower_limit, lower_limit, df1[col])

X=df1[['floor_area_sqm','Resale_year','lease_commence_year','town','flat_type','storey_range','flat_model','block']]
y=df1[['resale_price']]



st.set_page_config(page_title="SINGAPORE RESALE FLAT PRICES PREDICTING",layout="wide")
st.header(":red[SINGAPORE RESALE FLAT PRICES PREDICTING]")

tab1,tab2=st.tabs(["Introduction","Prediction"])
default_option="Introduction"
    
with tab1:  
    col1,col2,col3 = st.columns([6,0.1,6])
    with col1:
        st.image(Image.open("C:\\Users\\prave\\Downloads\\singapore.jpeg"), width=500)
        
    with col3:
        st.write("#### :red[**Overview :**] This project aims to develop a machine learning model and deploy it as a user-friendly online application to accurately predict the resale selling price of Flats in Singapore.")
        st.markdown("#### :red[**Technologies Used :**] Python, Pandas,NumPy, Visualization, Streamlit, Scikit-learn,Pickle")
    
with tab2:
   
    st.dataframe(df1)

    st.header("Fill the below following details to predict the Resale Flat Price")
    st.write("NOTE: Min and Max values are provided for reference, You can enter your desired value.")
   
    with st.form(key='my_form'):
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.subheader("Select Your Options")
            town_option=X['town'].unique()
            flat_type_option=X['flat_type'].unique()
            storey_option=X['storey_range'].unique()
            flat_model_options=X['flat_model'].unique()
            block_option=X['block'].unique()
            
            town=st.selectbox("Select the Town:",town_option)
            flat_type=st.selectbox("Select the Flat Type:",flat_type_option)
            flat_model=st.selectbox("Select the Flat Model:",flat_model_options)
            storey_range = st.selectbox("Select the Storey Range:",storey_option)
            block=st.selectbox("Select the block number:",block_option)
        with col3:
            st.subheader("Enter your values")
            floor_area=(st.text_input("Floor Area sqm(Min:28.0 & Max:173.0): "))
            resale_year=(st.text_input("Resale Year(Min:1990.0 & Max:2024.0): "))
            lease_commence=(st.text_input("Lease Commence Year(Min:1966.0 & Max:2018.5): "))
            
            submit_button = st.form_submit_button(label="PREDICT PRICE")

            if submit_button:
    
                with open('model.pkl', 'rb') as file:
                    training=pickle.load(file)

                with open('town.pkl', 'rb') as f:
                    town_encoder = pickle.load(f)
                with open('type.pkl', 'rb') as f:
                    type_encoder = pickle.load(f)
                with open('storey.pkl', 'rb') as f:
                    storey_encoder = pickle.load(f)
                with open('flat_model.pkl', 'rb') as f:
                    flat_model_encoder = pickle.load(f)
                with open('block.pkl','rb') as f:
                    block_encoder=pickle.load(f)

                with open('scaler.pkl','rb') as f:
                    scaler=pickle.load(f)

                new_sample = pd.DataFrame({
                    'floor_area_sqm': [floor_area],  
                    'Resale_year': [resale_year],
                    'lease_commence_year':[lease_commence],
                    'town': [town],
                    'flat_type': [flat_type],
                    'storey_range': [storey_range],
                    'flat_model': [flat_model],
                    'block':[block]
                })
                town_transformed = town_encoder.transform(new_sample['town']).reshape(-1, 1)
                type_transformed = type_encoder.transform(new_sample['flat_type']).reshape(-1, 1)
                storey_transformed = storey_encoder.transform(new_sample['storey_range']).reshape(-1, 1)
                model_transformed = flat_model_encoder.transform(new_sample['flat_model']).reshape(-1, 1)
                block_transformed=block_encoder.transform(new_sample['block']).reshape(-1,1)

                new_samples = np.concatenate((new_sample[['floor_area_sqm', 'Resale_year', 'lease_commence_year']].values, 
                              town_transformed, type_transformed, storey_transformed, model_transformed,block_transformed), axis=1)
                new_samples1= scaler.transform(new_samples)

                predicted_selling_price = training.predict(new_samples1)
                st.write(f'## :green[Predicted price:]:{predicted_selling_price}')





