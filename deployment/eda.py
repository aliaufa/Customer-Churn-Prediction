import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title = 'CUSTOMER CHURN PREDICTION' ,
    initial_sidebar_state= 'expanded',
)

def run():

    # membuat title
    st.title('CUSTOMER CHURN PREDICTION')
    st.subheader('EXPLORATORY DATA ANALYSIS')
    st.markdown('---')

    # menambahkan gambar
    image = Image.open('churn.jpg')
    st.image(image)
    st.write('## Background')
    st.write('''Customer churn poses a significant challenge for businesses as it can result in revenue loss 
                and hinder sustainable growth. Companies need to identify customers who are at risk of churning 
                and take appropriate actions to retain them. By analyzing the given dataset, valuable insights 
                can be gained regarding customer characteristics and behaviors that contribute to churn.''')
    st.write('## Problem Statement')
    st.write('''The objective is to assist a company in predicting customer churn using the provided dataset. 
                By developing a churn prediction model, the company can proactively identify customers likely 
                to discontinue using their products or services. This model will enable the company to implement 
                targeted retention strategies and improve overall customer retention rates.''')

    # membuat garis lurus
    st.markdown('---')

    # show dataframe
    st.write('# The Dataset')
    data = pd.read_csv('churn.csv')
    st.dataframe(data)

    markdown_text = '''
    ## Variable Descriptions

    | Column | Description |
    | --- | --- |
    | `user_id` | ID of a customer |
    | `age` | Age of a customer |
    | `gender` | Gender of a customer |
    | `region_category` | Region that a customer belongs to |
    | `membership_category` | Category of the membership that a customer is using |
    | `joining_date` | Date when a customer became a member |
    | `joined_through referral` | Whether a customer joined using any referral code or ID |
    | `preferred_offer types` | Type of offer that a customer prefers |
    | `medium_of operation` | Medium of operation that a customer uses for transactions |
    | `internet_option` | Type of internet service a customer uses |
    | `last_visit_time` | The last time a customer visited the website |
    | `days_since_last_login` | Number of days since a customer last logged into the website |
    | `avg_time_spent` | Average time spent by a customer on the website |
    | `avg_transaction_value` | Average transaction value of a customer |
    | `avg_frequency_login_days` | Number of times a customer has logged in to the website |
    | `points_in_wallet` | Points awarded to a customer on each transaction |
    | `used_special_discount` | Whether a customer uses special discounts offered |
    | `offer_application_preference` | Whether a customer prefers offers |
    | `past_complaint` | Whether a customer has raised any complaints |
    | `complaint_status` | Whether the complaints raised by a customer was resolved |
    | `feedback` | Feedback provided by a customer |
    | `churn_risk_score` | Churn score <br><br> `0` : Not churn <br> `1` : Churn |
    '''

    st.markdown(markdown_text)

    st.markdown('---')
    # Buat visualisasi
    st.write('# Data Visualization')
    ## Target Plot
    st.write('### Customer Churn Ratio')
    fig = plt.figure(figsize=(10,10))
    churn_count = data['churn_risk_score'].value_counts()


    plt.pie(churn_count, 
            labels=['Churn', 'Not Churn'], startangle=90,
            colors=['#F38091','#70CAb0'],
            autopct='%1.1f%%', explode=[0,0.1])
    plt.axis('equal')
    plt.title('Churn Ratio')
    st.pyplot(fig)
    st.write('The data seems balanced between churn and not churn. It seems from the data that customer \
             churn ratio is 54%. Not looking good.')

    ## Categorical Data Plot
    st.write('### Categorical Data Ratio')
    pilihan_kategori = st.selectbox('Pick Categorical Column : ', ('gender', 'region_category', 'membership_category', 'joined_through_referral', 'preferred_offer_types', 'medium_of_operation', 
                                                                'internet_option', 'used_special_discount', 'offer_application_preference', 'past_complaint', 'complaint_status', 'feedback'))
    color_palette = {0: '#70CAb0', 1: '#F38091'}
    fig= plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=pilihan_kategori, hue='churn_risk_score', palette=color_palette)

    plt.xlabel(pilihan_kategori.capitalize())
    plt.ylabel('Count')
    plt.title(pilihan_kategori.capitalize()+' Ratio')
    plt.legend(title='Churn')

    st.pyplot(fig)
    st.markdown('''
                    Some insights that can be inferred:

                    - `Gender Ratio`: The platform has a balanced **female-to-male ratio**, indicating a relatively equal representation of both genders among the customers.
                    - `Region Category`: Most customers are from the **"Town" region** category, suggesting that the platform is more popular in urban areas compared to villages or cities.
                    - `Membership Category`: Customers with **"No Membership"** and **"Basic Membership"** are more likely to churn, while those with **"Premium"** and **"Platinum"** memberships show higher retention rates.
                    - `Referral Program`: Customers who **joined through referrals** have a slightly higher churn rate but the ratio of customers with and without referals are balanced, indicating that referrals are effective for acquiring new customers but not necessarily for long-term retention.
                    - `Preferred Offer Types`: Customers who **prefer to receive no offers** are more likely to churn, suggesting that targeted offers and incentives might help improve customer retention.
                    - `Medium of Operation`: Customers using **smartphones** as their medium of operation are slightly more likely to churn compared to desktop users, highlighting the importance of optimizing the mobile user experience.
                    - `Internet Options`: The churn ratio is similar across different internet options (Wi-Fi, Fiber Optic, and Mobile Data), indicating that the type of internet service used by customers **does not significantly impact churn**.
                    - `Special Discounts`: Although the churn ratio is similar, a higher number of customers have utilized **special discounts**, implying that offering discounts alone may not be sufficient for customer retention.
                    - `Offer Application Preference`: Customers who **do not have a preference for offer applications** are more likely to churn, emphasizing the importance of personalized and targeted offer delivery.
                    - `Past Complaints`: The churn ratio is similar for customers with and without past complaints, but those **with past complaints** have a slightly higher likelihood of churning, indicating that addressing customer concerns and grievances is essential for retention.
                    - `Complaint Status`: Customers who complain about **"Not Applicable" issues** are more likely to churn, suggesting that unresolved or unsatisfactory complaints contribute to customer attrition.
                    - `Feedback`: Customers who provide positive feedback are less likely to churn, while those who leave **negative feedback** have a higher churn risk. This highlights the significance of addressing customer dissatisfaction and utilizing feedback to enhance the customer experience.
                ''')

    ## Numerical Data Plot
    # Select the numerical column
    pilihan_numerik = st.selectbox('Pick Numerical Column:', ('age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'))

    # Plot the numerical data distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=data, x=pilihan_numerik, hue='churn_risk_score', kde=True, ax=ax)

    ax.set_xlabel(pilihan_numerik.capitalize())
    ax.set_ylabel('Count')
    ax.set_title(pilihan_numerik.capitalize() + ' Distribution')
    st.pyplot(fig)

if __name__ == '__main__':
    run()