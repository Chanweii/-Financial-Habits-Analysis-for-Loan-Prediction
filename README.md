# Financial-Habits-Analysis-for-Loan-Prediction
### Abstract:

This study aims to analyze the impact of financial habits on loan prediction. By examining various financial behaviors and patterns, we seek to identify key factors that can predict an individual's creditworthiness. The analysis is based on a diverse dataset comprising financial records, credit histories, and borrower information. Employing machine learning techniques, this research can assist lenders in making more informed decisions while extending credit to potential borrowers. 
　　 <br /> 

![Photo by [Ryoji Iwata]([https://unsplash.com/@ryoji__iwata?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/a-qsFZimp1M?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)](https://images.unsplash.com/photo-1512799545738-0625ef92a288?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb](https://images.unsplash.com/photo-1512799545738-0625ef92a288?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1169&q=80))

Photo by [Ryoji Iwata](https://unsplash.com/@ryoji__iwata?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/a-qsFZimp1M?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 
  <br /> 

### Background and motivation

In today's rapidly evolving financial landscape, lending institutions face an ever-increasing need to make informed and prudent decisions when granting loans. The economic environment, individual financial behaviors, and creditworthiness have a profound impact on the repayment capacity of borrowers. Traditional lending practices, although effective to some extent, often fail to capture the complexity of modern borrowers' financial habits. As a result, financial institutions encounter challenges in accurately predicting the likelihood of loan defaults, leading to potential losses and increased risk.

---

### Case Study

Our case study focuses on a specialized P2P lending company that aims to identify potential borrowers who qualify for loans based on their financial information. The company intends to use the completion of the final stage, "e-signature," in the loan application process as a criterion for assessing eligibility. The goal is to offer new loan proposals tailored to borrowers who have not completed the e-signature, thereby increasing the overall loan application completion rate.

### Objectives of the Study:

The primary objective of this research is to conduct a comprehensive analysis of financial habits and their impact on loan prediction. By analyzing a diverse dataset comprising financial records, credit histories, and borrower information, we aim to uncover patterns and trends that can serve as valuable predictors of creditworthiness. Through a combination of machine learning techniques and feature engineering, our study seeks to develop a robust loan prediction model capable of assisting financial institutions in making more accurate and reliable lending decisions.

The primary objectives of this research are as follows:

1. **Conducting Financial Habits Analysis**: We analyze borrower information to identify key financial habits influencing loan repayment capacity, providing a comprehensive view of borrowers' financial health.
2. **Feature Engineering and Model Development**: We engineer relevant features based on insights from financial habits analysis and build a tailored, robust machine learning model for P2P lending data.
3. **Enhancing Loan Default Prediction**: Our model is fine-tuned for high accuracy, precision, and recall, effectively identifying high-risk and creditworthy borrowers.
4. **Assessing Model Performance**: We evaluate the model's effectiveness using Accuracy, Precision, Recall, F1 score, and AUC-ROC metrics, employing cross-validation to ensure reliability and generalizability.
5. **Practical Implementation and Implications:** We discuss actionable strategies for P2P lending platforms to optimize loan portfolios and mitigate risk based on our findings.

---

### **Data**

The data is from the "[**E-Signing of Loan Based on Financial History**](https://www.kaggle.com/datasets/yashpaloswal/esigning-of-loanbased-on-financial-history)" dataset on Kaggle, comprising 17,908 records with 21 features (17 numerical, 1 categorical, and 3 boolean). It includes loan request characteristics, borrowers' personal statistics, and credit-related information.

Data Description:

|  |  |
| --- | --- |
| entry_id | The identification number of the borrower. |
| age | Age of the borrower. |
| pay_schedule | The frequency of the borrower's salary received (weekly, bi-weekly, monthly, semi-monthly). |
| home_owner | Whether the borrower owns a property (Yes: 1, No: 0). |
| income | Monthly income of the borrower (US dollars). |
| months_employed | Number of months employed by the borrower. |
| years_employed | Number of years employed by the borrower. |
| current_address_year | Number of years the borrower has been at the current address. |
| personal_account_m | Number of months the borrower has owned a personal account. |
| personal_account_y | Number of years the borrower has owned a personal account. |
| has_debt | Whether the borrower has existing debt (Yes: 1, No: 0). |
| amount_requested | The loan amount applied for by the borrower (US dollars). |
| risk_score | Risk scores calculated by different algorithms. |
| risk_score_2 | Risk scores calculated by different algorithms. |
| risk_score_3 | Risk scores calculated by different algorithms. |
| risk_score_4 | Risk scores calculated by different algorithms. |
| risk_score_5 | Risk scores calculated by different algorithms. |
| ext_quality_score | Credit quality scores calculated by different algorithms. |
| ext_quality_score_2 | Credit quality scores calculated by different algorithms. |
| inquiries_last_month | Number of inquiries made by the borrower in the previous month. |
| e_signed | Whether electronic signature has been completed (Yes: 1, No: 0). |

---
