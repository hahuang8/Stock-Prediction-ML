# Introduction

Financial markets are highly complex and volatile, and prediction of stock price changes is a challenging but highly rewarding task (Al-Ali et al.). Earnings calls occur each quarter in which company executives discuss financial results and provide guidance containing valuable information that tend impact stock prices by multiple percentage points. Previous studies have shown that even small mistakes in wording can lead to drastic changes in stock price outcomes. Natural language processing (NLP) techniques have been used extensively to extract insights from textual data (Ma et al.). Our project aims to leverage NLP to predict stock price changes based on the content of earnings call transcripts using supervised learning.

The two datasets for this project include earnings call transcripts for S&P 500 companies over multiple recent years, and corresponding stock price information for the day of the earnings call. From this, we can derive the percentage change in a company's stock price immediately following the call to use as the target predictions/labels.

# Problem Definition

The motivation for this project lies in the potential for investors to benefit from real-time, automated analysis of earnings calls. Timely and accurate predictions of stock price movements can inform investment decisions, improving portfolio management and risk assessment (Medya et al.).

The central idea is predicting whether a company's stock price will increase, decrease, or remain relatively stable following an earnings call. We aim to extract quantifiable information from unstructured textual data to turn nuances in sentiment semantics into accurate predictions of the associated market reactions.

# Methods

## Data Collection and Preprocessing

We created two datasets.

We preprocessed earnings call transcripts from this [dataset](https://github.com/cdubiel08/Earnings-Calls-NLP/tree/main/transcripts/sandp500). We originally planned on tokenizing and converting the entire transcripts into input features suitable for BERT, but BERT can only take in 512 tokens at once, so we used Longformer instead to get the word embeddings from the tokens. Once we got the word embeddings, we got the appropriate labels from stock price information hosted on [Alpha Vantage](https://www.alphavantage.co/documentation/) through API requests. For our classification model, we created one hot label representing whether the stock price increased or decreased during the earnings call. We assumed each earnings call to be around one hour, so we found the percentage change in stock price one hour after the start of the earnings call.

We performed the same process again, but we split up the transcripts by one minute chunks. We assumed that the speaker spoke an average of 150 words per minute so each data point contained approximately 150 words. With only 150 words, we were able to use BERT to get the embeddings after they were tokenized. For these labels, the percentage change in stock price was calculated by the minute.

Alpha Vantage didn’t have stock data available for all of our data points, so we excluded those data points. Additionally, we excluded all of the data that changed by less than 2% during the designated time interval. This was to minimize the effects of class imbalance as seen in the chart below, when training our model, since there were many data points where the stock price did not change by a significant amount.

<img src="https://drive.usercontent.google.com/download?id=1fOLnpY2c6ha1m0USrddUdi3dpDwkGoQa&export=view" alt="labels" />

We performed PCA on both the datasets so that the retained variance of each was at least 90%. Below is a visualization of the transcript embeddings using t-SNE, and colored by positive or negative price change. In our scatter plots, 1 represents a positive change while -1 represents a negative change.

<img src="https://drive.google.com/uc?export=view&id=1RtdTIe_Rbcg54kobAy309kGL0qUGQ8Az" alt="embeddings1" />

<img src="https://drive.google.com/uc?export=view&id=14-l8EHfvdjxVpsAW5sJMWdLxbz1tDAu2" alt="embeddings2"/>

By the midterm report, we realized that using only word embeddings as parameters for the model was not sufficient, so we added earnings surprise (surprise eps) as an additional feature. Earnings surprise is given as a percentage and it is a measurement of how far off a company’s actual earnings per share was compared to their expected earnings per share for a quarter. We collected this data by scraping [Zack’s Investment Research’s data](https://www.zacks.com/stock/research/earnings-calendar) on earnings with Selenium. We hoped that a more quantitative feature would create more definite clusters in our data, helping our models learn. However looking at the scatterplot below, we can see that there were still no definite clusters after adding surprise eps.

<img src="https://drive.google.com/uc?export=view&id=1hzSMOEFBdN-r1qv_1rZYxeeIvOVTpkWC" alt="surpriseepspca"/>

Although we trained our model with a dataset with 85% retained variance, we projected the dimensions of the dataset to 2 with PCA for visualization purposes. There still seems to be no clusters in two dimensions, but there may be some in 33 dimensions (85% retained variance).

As an additional note, we only used the word embeddings from the 1 hour partitions with the surprise EPS. This was because the surprise EPS is better correlated with the entire earnings call transcript rather than 1 minute partitions.

## Classification Model

For the midterm, we had trained two classification models, one with the one minute partitions and one with the one hour partitions. We used a neural network with L2 regularization to prevent as much overfitting as we could. Each dataset was split such that 80 percent of the points were used to train the model and 20 percent of the points were used to test the model. We attempted to optimize the hyperparameters such as the number of epochs, batch size, and weight decay so that the model learned and did not overfit by plotting the cross entropy of every epoch for both the training and testing set. For a good model, the two trend lines should both decrease together and not diverge. Once results were satisfactory, we calculated the accuracy of both models.

For the final checkpoint, we train an additional classification model using a neural network. Our model was overfitting severely, so we tried different combinations of making the network simpler, using dropout, and L2 regularization to fix the overfitting. 

## Regression Model

For our final, we also trained a regression model. We used similar techniques to prevent overfitting such as L2 regularization, but we still ended up with overfitting issues and an inaccurate model because of our data. We tested our regression model’s performance with mean squared error. For a good model, the mean squared error trendline for the training and validation set should both decrease together at a similar rate. 

# Results and Discussion

After optimizing our hyperparameters, none of the models performed well. We believe that it is not possible to predict the direction of stock price change based on embeddings and surprise EPS. This is because word embeddings capture the sentiment of the words, but since each individual company is giving their own earnings call, they will all be speaking positively about their work. Even though surprise EPS is an objective, quantitative measurement of a company's performance, we believe that our parameters still were not sufficient for training an accurate model because of the volatility of stock prices and how many things affect stock prices in real life that were not accounted for. In order to accurately predict stock price movement, more quantitative financial information may be needed other than surprise EPS, such as overall market conditions, sector performance, or previous trends.

## Analysis

Below is a chart of our best performing model.

<img src="https://drive.google.com/uc?export=view&id=1Q8grGWpSOTdQtbdAKKE_ci6fF82KPIPE" alt="distrlabels"/>

Initially, the training and validation cross entropy decrease together, showing that the model is not overfitting. However, the model did not learn much as the cross entropy is still a relatively high value after 200 epochs. The cross entropy value for the training set after 200 epochs was 0.67 while the cross entropy value for the test set was 0.68.

We tested this model and got an accuracy of 57%. Since our cross entropy value is high and our accuracy is low, we can conclude that there is a problem with our data.

Looking at the scatterplots of our data, we can see that there are no obvious clusters for both the datasets. However, it is difficult to conclude anything from the graphs because the original data was in 768 dimensions (and reduced using PCA to <100), so we had to drastically reduce the dimensions for visualization. We trained the models in hopes that there would be more noticeable clustering in higher dimensions.

For our final report, our results were not much better.

<img src="https://drive.google.com/uc?export=view&id=1orjo8reWsESyQDdj47HQ7dbSUE5aZqo1" alt="classificationwitheps"/>

Our cross entropy for the training set was 0.62 and our cross entropy for the validation set was 0.70. We can see that there is not much of a difference from our midterm report where surprise EPS was not included as an additional parameter. The model’s accuracy for when EPS was included was 52% which is lower than the other classification model. One reason this might have happened is because we have less data to work with. There were some data points that we couldn’t find the surprise EPS so we excluded those data points for the newer model. We also got rid of surprise EPS values that were considered outliers using interquartile range, so this further decreased the size of our dataset. Since we were working with a smaller dataset, it increased the potential for overfitting and also contributed to our model being less accurate.

<img src="https://drive.google.com/uc?export=view&id=1gWW9S9faRA4IEhbnmmYBt055VcdrEcvC" alt="regressionwitheps"/>

For our regression model, we only ran it 1000 epochs. After the 1000 epoch mark, the validation set’s trendline for MSE started going upward, signifying that the overfitting was getting worse. By 1000s epochs, the MSE for the training set was 12.1 and the MSE for the validation set was 15.5. For a good model, the MSE would be close to 0. Since our MSE values are far off from 0, we can conclude that our regression model performed poorly as well for similar reasons as to why our classification model performed poorly.

## Next Steps

For our midterm we believed that the main inaccuracy in our results arose from the failure to choose the sufficient parameters for the neural network classification model to detect in the two datasets. We believe that using sentence embeddings as our only feature is not sufficient enough to predict stock price prediction, so we added surprise eps as an additional feature. However, the quantitative measurements did not help the models learn better and make accurate predictions. For our next steps, we should include other parameters such as the state of the economy and the sector the company is involved in. Adding as many relevant parameters as possible will help our models capture the change in stock prices.

# References:

- Al-Ali, Ahmed Ghanim, et al. Deep Learning Framework for Measuring the Digital Strategy of Companies from Earning Calls, Association for Computational Linguistics, 8 Dec. 2020, aclanthology.org/2020.coling-main.80.pdf.
- Ma, Zhiqiang, et al. Towards Earnings Call and Stock Price Movement, Association for Computing Machinery, 24 Aug. 2020, arxiv.org/pdf/2009.01317.pdf.
- Medya, Sourav, et al. An Exploratory Study of Stock Price Movements from Earnings Calls, Association for Computing Machinery, 3 June 2018, arxiv.org/pdf/2203.12460.pdf.

# Proposed Timeline:

[Link to Gantt Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/gramesh31_gatech_edu/EdeXTRvfNxpNrly5V-2R_T0B_FrxWppNGQxnsnZNPazKYQ?e=gY7nqz)
<img src="https://drive.google.com/uc?export=view&id=1GnEGqZB6PyifsQE-EGvnW9jxMgNr_YB7" alt="chart 1" />
![labels](https://hahuang8.github.io/Stock-Prediction-ML/chart1.png)
<img src="https://drive.google.com/uc?export=view&id=1WCq2wu7V1CXR2NqsuHDAAYg37D_j_fkq" alt="chart 2" />

# Contribution Table:

Proposal

<table>
  <tr>
    <th>Teammate</th>
    <th>Contributions</th>
  </tr>
  <tr>
    <td>Harry</td>
    <td>Introduction & Background, Video</td>
  </tr>
  <tr>
    <td>Wonho</td>
    <td>Problem Definition, Video</td>
  </tr>
  <tr>
    <td>Govind</td>
    <td>Methods, Video</td>
  </tr>
  <tr>
    <td>Taesung</td>
    <td>Potential Results & Discussion, Video</td>
  </tr>
</table>

Midterm

<table>
  <tr>
    <th>Teammate</th>
    <th>Contributions</th>
  </tr>
  <tr>
    <td>Harry</td>
    <td>Data Collection/Preprocessing, Article Writing</td>
  </tr>
  <tr>
    <td>Wonho</td>
    <td>Data Collection/Preprocessing, Article Writing</td>
  </tr>
  <tr>
    <td>Govind</td>
    <td>Model Training, Article Writing</td>
  </tr>
  <tr>
    <td>Taesung</td>
    <td>Model Training, Article Writing</td>
  </tr>
</table>

Final

<table>
  <tr>
    <th>Teammate</th>
    <th>Contributions</th>
  </tr>
  <tr>
    <td>Harry</td>
    <td>Data Collection/Preprocessing, Article Writing, Video</td>
  </tr>
  <tr>
    <td>Wonho</td>
    <td>Data Collection/Preprocessing, Article Writing, Video</td>
  </tr>
  <tr>
    <td>Govind</td>
    <td>Model Training, Article Writing, Video</td>
  </tr>
  <tr>
    <td>Taesung</td>
    <td>Model Training, Article Writing, Video</td>
  </tr>
</table>

# Checkpoints

Midterm Report:

- [x] [Dataset for transcripts](https://github.com/cdubiel08/Earnings-Calls-NLP/tree/main/transcripts/sandp500) cleaned and data preprocessing done
- [x] [Additional dataset for stock information](https://www.alphavantage.co/documentation/) cleaned and data preprocessing done
- [x] Neural network model completed
- [x] Written up midterm report

Final Report:

- [x] Neural network regression model completed
- [x] Written up finalized report
- [x] Presentation recorded
