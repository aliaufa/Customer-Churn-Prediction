import pandas as pd
import numpy as np
import streamlit as st
import datetime
import pickle
from tensorflow.keras.models import load_model

# Load the combined pipeline
with open('preprocessor.pkl', 'rb') as file_1:
  model_pipeline = pickle.load(file_1)

model_ann = load_model('churn_sequential_2.h5')

def run():
    # membuat title
    st.title('CUSTOMER CHURN PREDICTION')
    st.subheader('Predicting customer churn')
    st.markdown('---')
    st.write("# Customer Information")
    # Buat form
    with st.form(key='form_flight_delay'):
        age = st.number_input("Customer's age", min_value=0, max_value=100, value=30, step=1)
        gender_form = st.radio("Customer's gender", ('Male', 'Female'))
        gender = None
        if gender_form == 'Male':
            gender = 'M'
        elif gender_form == 'Female':
            gender = 'F'

        region = st.radio('Region that a customer belongs to', ('City', 'Village', 'Town'))
        
        membership = st.selectbox("Customer's membership", ('No Membership', 'Basic Membership', 'Silver Membership', 
                                                            'Gold Membership', 'Premium Membership', 'Platinum Membership'))
        
        join_date = st.date_input("Date when a customer became a member", datetime.date(2023, 6, 1))

        referral = st.radio('Whether a customer joined using any referral code or ID', ('Yes', 'No'))

        offer = st.selectbox('Type of offer that a customer prefers', ('Without Offers', 'Credit/Debit Card Offers', 'Gift Vouchers/Coupons'))

        medium = st.radio('Type of offer that a customer prefers', ('Desktop', 'Smartphone', 'Both'))

        internet = st.radio('Type of offer that a customer prefers', ('Wi-Fi', 'Fiber_Optic', 'Mobile_Data'))

        last_visit = st.time_input('The last time a customer visited the website', datetime.time(21, 00), step=300)

        last_login = st.number_input("Days since last login", min_value=0, max_value=30, value=1, step=1,
                                     help='Put 30 if more than 30 days')

        time_spent = st.number_input('Average time (minutes) customer spent in the platform', min_value=0, max_value=4000, value=10, step=1)
        
        transaction_value = st.number_input("Customer's average transaction value", min_value=0, max_value=100000, value=0, step=1, 
                                   help='minimum value = 0, maximum value = 100000')
        
        login_frequency = st.number_input("Number of times a customer has logged in to the website", min_value=0, max_value=100, value=0, step=1, 
                                   help='if more than 100 put 100')
        
        points = st.number_input("Points awarded to a customer on each transaction", min_value=0, max_value=100, value=0, step=1, 
                                   help='if more than 100 put 100')
        
        discount = st.radio('Whether a customer uses special discounts offered', ('Yes', 'No'))

        app_preference = st.radio("Whether a customer prefers offers", ('Yes', 'No'))

        past_complaint = st.radio("Whether a customer has raised any complaints", ('Yes', 'No'))

        complaint_status = st.selectbox("Whether the complaints raised by a customer was resolved", ('No Information Available', 'Not Applicable', 'Unsolved', 'Solved', 'Solved in Follow-up'))

        feedback = st.selectbox("Feedback provided by a customer", ('Poor Website', 'Poor Customer Service', 'Too many ads', 
                                                                                 'Poor Product Quality', 'No reason specified', 'Products always in Stock', 
                                                                                 'Reasonable Price', 'Quality Customer Care', 'User Friendly Website'))

        submitted = st.form_submit_button('Predict')

        # dataframe
        data_inf = {
                    'age': age,
                    'gender': gender,
                    'region_category': region,
                    'membership_category': membership,
                    'joining_date': join_date,
                    'joined_through_referral': referral,
                    'preferred_offer_types': offer,
                    'medium_of_operation': medium,
                    'internet_option': internet,
                    'last_visit_time': last_visit,
                    'days_since_last_login': last_login,
                    'avg_time_spent': time_spent,
                    'avg_transaction_value': transaction_value,
                    'avg_frequency_login_days': login_frequency,
                    'points_in_wallet': points,
                    'used_special_discount': discount,
                    'offer_application_preference': app_preference,
                    'past_complaint': past_complaint,
                    'complaint_status': complaint_status,
                    'feedback': feedback,
                    }

        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf.T, width=800, height=495)

    if submitted:
        # Predict using created pipeline
        data_inf_transform = model_pipeline.transform(data_inf)
        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
        if y_pred_inf == 0:
            pred = 'Not Churn'
        else:
            pred = 'Churn'
        st.markdown('---')
        st.write('# Prediction : ', (pred))
        st.markdown('---')

if __name__ == '__main__':
    run()