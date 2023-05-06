# U-integrity
## Introduction
Students fail to benefit from learning opportunities and do not contribute their own work when they falsely utilized ChatGpt. As a result, we could potentially see downfalls in education systems leading to society losing the original purpose of education. We aim to compare the academic performance of students and ChatGPT by automatically grading student-written and ChatGPT-generated essays using a machine-learning model. And answer the following question: is it worthwhile for a student to use ChatGPT, knowing that they could potentially face academic penalties?

## Data 
- Kaggle (6230 essays)
  - Total of 8 essay prompts 
  - Excluding 4 prompts as they required background information to the texts we didn’t have access to
  - 4 prompts remained: prompts 1, 2, 7 and 8, total 6230 essays

- ChatGPT (609 essays)
  - Used the OpenAI API to generate essays
  - Chose the “text-curie-001” model as it’s very capable, fast and low cost

## Model
- LSTM Network (LSTM) Network
  - Learn, process, and classify sequential data because these networks can learn long-term dependencies between time steps of data 
  - Specifically chosen as it performs well with natural language and it will be able to understand the essays fairly easily
[![Overall LSTM Model](https://i.postimg.cc/7ZCNtyPQ/2023-03-26-013413.png)](https://postimg.cc/K12L4wMn)

- The model used mean squared error (MSE) loss
  - Loss: 0.0261
  - MAE: 0.127
  
[![Prediction Example](https://i.postimg.cc/3rSpsbqM/2023-05-06-113354.png)](https://postimg.cc/LhPnfvRv)

## Final Result
[![Distribution of Essays by Score](https://i.postimg.cc/ydsyvgb7/Picture1.png)](https://postimg.cc/NyCX0Mtn)
- When observing essay vs score distribution we notice that the ChatGpt score distribution is right-skewed and the human written score distribution is left-skewed, indicating that there may be some underlying issues with the data. However, a larger sample size may help reduce the impact of skewness and bring the means of both graphs closer together using the central limit theorem.
- As a result, it appears that ChatGPT is more capable of generating average essays, and is less likely to produce very poor or great ones compared to human ability. Because the ChatGPT model has a narrower distribution of scores and less extreme values than the human-generated essays.
