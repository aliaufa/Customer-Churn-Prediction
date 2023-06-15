# Customer-Churn-Prediction
___

## Problem Statement
The objective is to assist a company in predicting customer churn using the provided dataset. By developing a churn prediction model, the company can proactively identify customers likely to discontinue using their products or services. This model will enable the company to implement targeted retention strategies and improve overall customer retention rates.

Minimizing incorrectly predicts that a customer will not churn (negative prediction) when they actually do churn (positive ground truth) is important because it helps businesses take appropriate actions to retain customers who are at risk of churning.

I also deployed the model on huggingface.

Huggingface URL : https://huggingface.co/spaces/aliaufa/Customer-Churn-Prediction

## Methods used
1. Data Visualization
2. Feature Engineering
3. Artificial Neural Network
    - Sequential API
    - Functional API
4. Callbacks
    - Early Stopping
5. Model Inference

## Technologies
1. Python
2. Pandas
3. Matplotlib
4. Seaborn
5. Scikit-Learn
6. Tensorflow
7. Streamlit
8. Huggingface

---

## Findings
From exploring data we found:

- The customer **churn ratio of 54%** indicates a significant challenge for the company in retaining customers. A churn rate of this magnitude suggests that more than half of the customers are discontinuing their usage of the company's products or services.

- Insights from categorical data provide valuable information for the company to focus on specific areas such as **membership categories**, **referral programs**, **personalized offers**, **mobile user experience**, **complaint resolution**, and **feedback management** to reduce churn and improve customer retention.

- Insights from numerical data shows that **customers that spent more** and **have more points** in their wallet are less likely to churn. The company can focus on improving **customer engagement and satisfaction**, as well as developing strategies to **increase transaction values and point accumulations** to reduce churn rates.

From evaluating ANN models we found:

- Our initial sequential and functional model is good with accuracy of 92% but there is exploding gradient. We improved the model by adding batch normalization and dropouts in the layer to reduce gradient exploding and reduce overfitting. We also reduce the number of hidden layer to simplify the model.

- From evaluation we can see that the improved model performs better at predicting customer that churns (True positive) and the number of false "not churn" prediction also reduced.

- Improved Functional model is chosen because it has the lowest false negative prediction. Minimizing false negatives is crucial because it helps to identify customers who are likely to churn but may be missed by the model. By correctly identifying these customers as potential churners, appropriate actions can be taken to retain them and prevent them from leaving the platform or service.

- Based on the classification report, our model shows good performance with high precision, recall, and F1-score for both classes, indicating that it is able to effectively distinguish between the two classes.

- The model performs better at predicting churn (class 1) than not churn (class 0) by a small margin.

- The overall accuracy of the model is 0.93, indicating that it correctly predicts the class for 93% of the samples

## Conclusion and Recommendation
In conclusion, the analysis of the churn prediction problem revealed a significant **churn ratio of 54%**, highlighting the pressing need for the company to focus on customer retention strategies. By leveraging insights from both categorical and numerical data, the company can target specific areas such as membership categories, referral programs, personalized offers, mobile user experience, complaint resolution, and feedback management to reduce churn and improve customer satisfaction.

The evaluation of the ANN models showed that the initial model could be **improved by addressing issues of exploding gradients and overfitting**. The **improved Functional model**, with added batch normalization and dropouts, demonstrated the **best performance by minimizing false negatives**. This is crucial as it allows the company to accurately identify customers at risk of churn and take appropriate actions to retain them.

The classification report further validated the effectiveness of the model, with high precision, recall, and F1-scores for both churn and non-churn classes. The model's ability to correctly predict churn was slightly better than its ability to predict non-churn. Overall, the model achieved an **accuracy of 93%,** indicating its reliability in classifying customer churn.

By leveraging these insights and employing the improved Functional model, the company can proactively address customer churn, enhance customer satisfaction, and ultimately improve business performance. It is important for the company to continually monitor and optimize its churn prediction strategies to maintain customer loyalty and maximize customer lifetime value.

To **further improve** the churn prediction model we can do the following :

- Consider other feature engineering approach to explore new features or transformations.
- Perform different hyperparameter tuning to optimize the model's settings.
- Collect more data to increase the dataset size.
- Seek domain expertise to gain insights and refine the model.
- Regularly monitor and update the model as new data becomes available.

Implementing these strategies can enhance the model's performance and accuracy in predicting churn.