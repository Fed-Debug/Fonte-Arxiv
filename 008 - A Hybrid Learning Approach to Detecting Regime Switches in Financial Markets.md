# A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets

Peter Akioyamen
pakioyam@uwo.ca
Western University
London, Ontario, Canada 

Yi Zhou Tang
ytang294@uwo.ca
Western University
London, Ontario, Canada 

Hussien Hussien
hhussien@uwo.ca
Western University
London, Ontario, Canada 

## ABSTRACT

Financial markets are of much interest to researchers due to their dynamic and stochastic nature. With their relations to world populations, global economies and asset valuations, understanding, identifying and forecasting trends and regimes are highly important. Attempts have been made to forecast market trends by employing machine learning methodologies, while statistical techniques have been the primary methods used in developing market regime switching models used for trading and hedging. In this paper we present a novel framework for the detection of regime switches within the US financial markets. Principal component analysis is applied for dimensionality reduction and the k-means algorithm is used as a clustering technique. Using a combination of cluster analysis and classification, we identify regimes in financial markets based on publicly available economic data. We display the efficacy of the framework by constructing and assessing the performance of two trading strategies based on detected regimes. 

## CCS CONCEPTS

• Information systems → Clustering. 

## KEYWORDS

clustering, dimensionality reduction, classification, trading, Markov regime switching, portfolio management 

## ACM Reference Format:

Peter Akioyamen, Yi Zhou Tang, and Hussien Hussien. 2020. A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets. In ACM International Conference on AI in Finance (ICAIF '20), October 15–16, 2020, New York, NY, USA. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3383455.3422521 

## 1 INTRODUCTION

There is a variety of empirical evidence and corresponding studies which suggest that the time-series behaviors of both economic and financial data may not express invariable patterns through time. Intervals of divergent behavior in financial and economic time-series are often referred to as regimes. The relations between financial markets, global economies, asset valuations, and world populations has made the detection of market regime switches of importance to researchers and practitioners. The financial markets may be qualitatively characterized by various non-disjoint regimes. Periods of high and low volatility influence the actions of traders and investors alike. Often, economic expansions and contractions dictate the long-term market environment, while certain assets can exhibit mean-reverting or trending behaviors in the short-term $[2, 6, 11]$ . The presence of herding, originally studied in $[3]$ , as well as contrarianism may generate changes in market conditions such as misalignments in price action and fundamental asset valuations, among others $[9, 10, 33, 38, 39]$ . The aforementioned behaviors and properties which present themselves over time characterize unique sets of market regimes which may occur simultaneously. 

To capture the dynamic behavior of such time-series, it is often preferable to employ multiple models. Particularly, when attempting to model financial time-series over a set of regimes, the Markov switching model $[18]$ , also known as the regime switching model, has been studied and applied. Generally, the Markov switching model is governed by an unobservable state variable and time-varying transition probabilities. Despite the many empirical applications and theoretical developments involving Markov switching models, a necessary condition for the method is the need to know the number of regimes which exist prior to estimating model coefficients. This limits the flexibility of the regime switching model in two contexts. The first being applications where modeling and forecasting the time-series may be undesirable, and identifying the occurrence of the underlying regimes may be all that is sought after. The second consists of contexts where it may be beneficial to directly infer the number of regimes from data. 

In this study, we seek to identify meaningful regimes financial time-series may be subject to at different points in time without explicitly modeling the time-series as an autoregressive process. As such, we provide an alternative approach to detecting regime switches through the use of a hybrid learning framework. The k-means clustering algorithm is used to detect points in time which share underlying characteristics, grouping the data into distinct regimes. We show how clustering may be a desirable approach to regime detection as the number of regimes in the data can be passively inferred. Classification models are fit to the regimes detected by k-means, and are then used to predict the regime corresponding to each observation in the out-of-sample data. Based on the historical performance of assets during regimes identified in training data, trading strategies are constructed, displaying one application of the approach in trading and portfolio management. 

The paper is organized as follows. Related work is presented in Section 2. In Section 3 we discuss the data set used in this work and the related pre-processing techniques applied to the data, including the use of principal component analysis. Section 4 outlines the framework for regime detection using unsupervised learning. Specifically, the combination of cluster analysis and classification used in the unsupervised learning approach along with in-sample model performance are presented in Sections 4.1 and 4.2. In Section 5 we display the efficacy of this approach and a potential use-case of this framework. The construction and performance of two trading strategies based on the detection of regime switches is discussed. The paper is then concluded in Section 6. 

## 2 LITERATURE REVIEW

The Markov switching model, which governs shifts in the coefficients of an autoregression through a discrete-state Markov process, was originally presented in $[18]$ . Extensions of this work were quickly established with the use of the EM algorithm for maximum likelihood estimation along with the application of the cointegrated vector autoregressive process subject to Markovian shifts $[19, 27]$ . The Markov switching model of conditional mean was extended to incorporate the switching mechanism into conditional variance models as well. The autoregressive conditional heteroscedasticity (ARCH) and generalized autoregressive conditional heteroscedasticity (GARCH) models were studied with Markov switching in $[5, 20]$ , among others. GARCH with Markov switching was used to model and forecast price volatility in gold; it was found that trading gold futures based on this model resulted in higher cumulative return compared to other GARCH type models $[37]$ . The regime switching ARCH model is also seen in the modeling of Taiwanese stock market volatility $[8]$ . The Markov switching model and its variants have been applied widely in the analysis of economic and financial time-series. The analysis and forecasting of economic business-cycles can be seen in $[16]$ , which has gone on to be an important application of the Markov switching model. Among other use-cases, variants of the Markov switching model have been employed to analyze the behavior of interest rates and foreign exchange rates as well $[13, 15, 17]$ . A multivariate extension of the regime switching model is used in $[36]$ where regime dependence is found in the relationship between the stock index spot and futures markets. The model has seen applications in trading and portfolio management for hedging as well as price and volatility modeling $[1, 4, 7, 12, 14, 21, 29, 31]$ . Naturally, many of these applied works are concerned with modeling the time-series and its behaviors rather than the identification of the regime switches and persistence themselves. 

To the best of our knowledge, there has been no attempt to apply a hybrid approach to regime detection which uses both unsupervised and supervised learning algorithms. A related approach has been taken in $[40]$ where authors seek to forecast the daily directional movements of the SPDR S&P 500 ETF closing price. Beyond this, $[28, 32]$ apply clustering as a technique for data mining in the stock market, assisting in the extraction of ancillary investment information or directly in the construction of investment portfolios. The study and application of unsupervised learning for regime switch detection is extremely limited, and this work attempts to fill this gap. 

## 3 DATA AND METHODS

In this study, open-source Federal Reserve Economic Data (FRED) from aggregator Quandl is used $[34]$ . The data used ranges from January 7, 1994 to April 1, 2020, with each unique time-series tracking an economic statistic by its day-on-day percentage change. These time-series can be grouped into eight categories: growth statistics, price and inflation indices, money supply statistics, interest rates, employment statistics, income and expenditure statistics, debt levels, and miscellaneous indicators which report housing starts and gross private domestic investment, among others. In practice, the release of economic data can be infrequent and impacts the way both investors and traders participate in the markets. Projections are often made in between data releases; although useful, these expectations are often given context using the latest official data releases. Different data sources and aggregators can exhibit inconsistencies in projections and forecasts depending on the economic statistic in question. Consequently, in this study, missing values which naturally occur in between reporting dates of a statistic were imputed using the latest observed percentage change in the data. This coincides with the most recent official information a trader or investor may have at a given point in time. Some time-series observations and economic statistics were removed from the data set prior to analysis due to having a late first reporting date or an extremely low frequency of reporting. The resulting matrix contains 48 time-series (columns) which track various macroeconomic statistics within the USA over 6625 time observations (rows). The data is split into in-sample (training) and out-of-sample (test) sets. The training data consists of all observations up to and including December 31, 2013 (5051 observations). Testing data contains observations from January 1, 2014 onward (1574 observations). 

## 3.1 Principal Component Analysis

Let R be the data matrix of size $T \times S$ , where T is the total number of time observations considered, and S is the total number of economic statistics used as features in this work. Every entry $r_{ij}$ of matrix R represents the day-on-day percentage change of economic statistic j at time i. Principal component analysis (PCA) is performed on the matrix $R^{*}$ , the column-standardized version of R, using its singular value decomposition (SVD). The resulting matrix, denoted X, contains 48 principal components. The derivation of PCA and its properties are discussed in [23–25, 30]. 

Principal component analysis is used as a dimensionality reduction and feature extraction technique. We seek to capture maximal variance from the original data set using linear combinations of the economic statistics at each point in time. The resulting principal components are uncorrelated time-series observations. This allows for a lower dimensional representation of the original data; we elect to use 26 principal components in the analysis which collectively contain 90% of the variance present in the original data. Table 1 displays a summary of the results produced by PCA – the eigenvalue of each dimension represents the amount of variance the dimension captures. 

The loadings of the first two principal components allow for their characterization with respect to the economic statistics contained in the original data. The first principal component primarily captures economic productivity within the USA. The top four factors contributing to the first principal component include gross domestic product, real gross domestic product, all employees: total nonfarm payrolls, and federal debt: total public debt as percent of gross domestic product. The top four contributors to the second principal component include personal saving rate, real disposable personal income, disposable income, and personal consumption expenditures. As such, principal component two primarily represents economic income and expenditure-related factors. 


Table 1: Results of principal component analysis.


<table><tr><td>Principal Component</td><td>Dim 1</td><td>Dim 2</td><td>Dim 3</td></tr><tr><td>Eigenvalue</td><td>8.987</td><td>3.089</td><td>3.023</td></tr><tr><td>% of variance</td><td>18.72</td><td>6.44</td><td>6.30</td></tr><tr><td>Cumulative % of variance</td><td>18.72</td><td>25.16</td><td>31.46</td></tr><tr><td>Principal Component</td><td>Dim 16</td><td>Dim 32</td><td>Dim 48</td></tr><tr><td>Eigenvalue</td><td>1.021</td><td>0.356</td><td>9.284e-7</td></tr><tr><td>% of variance</td><td>2.12</td><td>0.74</td><td>0.00</td></tr><tr><td>Cumulative % of variance</td><td>73.94</td><td>96.76</td><td>100.00</td></tr></table>

## 4 REGIME DETECTION

In this regime detection framework, we perform a cluster analysis on the principal components to identify intervals in which the time-series exhibit similar underlying behaviors and characteristics. Through this we are able to find distinct groupings in which each observation may belong, and these groupings are the regimes that are detected in the data. The process of regime detection on the out-of-sample principal component data is framed as a classification task where the regimes detected through cluster analysis act as the classification labels for the in-sample data. Although clustering algorithms such as k-means and k-medoids may be extended to cluster out-of-sample data by assigning each of the new out-of-sample observations to the pre-existing centroid or medoid with the smallest distance or dissimilarity, the ability to cluster out-of-sample data is not necessarily generalizable to other algorithms. By combining cluster analysis and classification in this way, we are able to develop a framework that is model agnostic with respect to the clustering algorithm chosen in a given instance. 

## 4.1 Regime Clustering

We perform cluster analysis on the training data set using the k-means clustering algorithm with euclidean distance $[22]$ . The value of k used in this work is inferred from the training data through the average silhouette width method $[26, 35]$ . The selection of k is automated by choosing the value of k which maximizes the average silhouette width. In this work we restrict the values of k considered to be in the range 1 to 6. The average silhouette width for the possible values of k considered is displayed in Figure 1. From this we select k = 2 such that the training data is segmented into two detected regimes. The k-means algorithm is run with 100 random initializations to ensure the stability of the clusters. 

Figure 2 displays the first two principal components colored by the occurrence of regimes 1 and 2. It can be seen that meaningful and distinct regimes are detected from the data. Both the latter half of the 2000–2002 dot-com bubble as well as the majority of the 2007–2009 financial crisis are captured in regime 2 (colored light blue). This is indicative of regime 2 coinciding with economic crises and market bubbles originating from the US financial markets. Regime 1 (colored black) coincides with non-crisis periods where the economy and financial markets within the US are generally operating in the absence of abnormal market behavior and events. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/cde5c9f6-986e-4f0b-8389-14ccff97ddea/152773fac96ddca8df8ffbfc0fb20c3ee680c39e5130e9558515a65bb70d10f6.jpg)



Figure 1: Average silhouette width across 6 values of k.


## 4.2 Regime Classification

Supervised learning models are fit to the in-sample principal component data and regimes, and is subsequently used to predict the regimes of the out-of-sample observations. The models considered include linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), logistic regression, decision tree classifier, adaptive boosting (AdaBoost), and naive bayes classifier. Each of the models used in this work take the first 26 principal components as input, which may be denoted $X_{1}, \ldots, X_{26}$ , and are trained over the time period 1994–2013. The models produce a binary output which is then mapped to a value in $\{1, 2\}$ , representing the regime predicted. Model performance is assessed using 10-fold cross-validation. Table 2 displays the in-sample performance results including area under the ROC curve (AUC), accuracy, and F1 score for each model. 


Table 2: Model performance on in-sample data.


<table><tr><td>Model</td><td>AUC</td><td>Accuracy</td><td>F1 Score</td></tr><tr><td>LDA</td><td>0.9996</td><td>0.9909</td><td>0.9950</td></tr><tr><td>QDA</td><td>0.9912</td><td>0.9661</td><td>0.9813</td></tr><tr><td>Logistic Regression</td><td>1.0000</td><td>0.9996</td><td>0.9998</td></tr><tr><td>Decision Tree</td><td>0.9957</td><td>0.9990</td><td>0.9995</td></tr><tr><td>AdaBoost</td><td>1.0000</td><td>0.9994</td><td>0.9997</td></tr><tr><td>Naive Bayes</td><td>0.9852</td><td>0.9596</td><td>0.9777</td></tr></table>

All models demonstrate a strong ability to classify regimes, with each model achieving an accuracy above 0.95 and a minimum AUC of 0.98. The naive bayes classifier performs the poorest across all three metrics while both logistic regression and AdaBoost produce ideal ROC curves with an AUC of 1. These performance metrics may assist in model selection for strategy construction by assessing each model's ability to generalize on the out-of-sample data. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/cde5c9f6-986e-4f0b-8389-14ccff97ddea/1fdbd3867478a83c26ede86473d833a3395f6c1896350f3c0e347b0ce4c4d1ba.jpg)



(a) Principal component 1 colored by regime after clustering.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/cde5c9f6-986e-4f0b-8389-14ccff97ddea/9929cf1ad21011046a5e5abb87594ab384130f65716fd4e1ba19e4810539361d.jpg)



(b) Principal component 2 colored by regime after clustering.



Figure 2: The first two principal components colored according to the regimes after performing clustering on the in-sample time period (1994–2013).


Despite this, we employ all models in constructing unique trading strategies to explore the existence of any relationships between the performance evaluation metrics and performance of the trading strategies considered. 

## 5 BACKTEST RESULTS

We display the efficacy of this hybrid approach through the construction of two trading strategies, tail-hedging and tactical allocation. The decision rules governing the assets in a specific strategy are defined by the historical performance of the assets through the regimes identified in the in-sample data. The tail-hedging strategy considered here is executed for a single asset, where performance is benchmarked against the underlying asset's passive buy-hold performance (Table 3). Tactical allocation is a rotational strategy which consists of multiple assets. The performance of the strategy is benchmarked against a passive buy-hold strategy applied to the S&P 500. Every asset is traded through its respective continuous front month futures contract. The performance of each strategy is assessed over the out-of-sample time period, 2014–2020. 

The trading mechanism for both strategies is as follows: at the end of day t, new data is released and pre-processed giving inputs $X_{1}, \ldots, X_{26}$ . These are used by the selected classification model to predict the current regime on day $t + 1$ . Based on the predicted regime, an appropriate trade is executed on day $t + 1$ in accordance with a strategy's decision rules. Positions are entered at the start of day $t + 1$ and held until the start of day $t + 2$ , where the same process follows and the position is reassessed. Transaction costs are assumed to be negligible for each trade that is executed. 

## 5.1 Strategy 1: Tail-hedging

The tail-hedging strategy is designed for investors that seek long exposure to a given asset while being hedged against crisis periods. The strategy holds a long position during regime 1 and switches to a short position during regime 2; it is designed for assets with positive expected returns during regime 1 and negative expected returns in regime 2. We demonstrate the performance of this strategy using the continuous front month futures contracts on crude oil and the S&P 500, respectively. 

Applied to the S&P 500, four out of six models were able to construct a tail-hedging strategy which achieves higher cumulative returns than the benchmark; five of the models achieved a lower maximum drawdown. The strategy produces an average alpha of 5.12% across the models while maintaining an average beta of 0.498. The cumulative return of the S&P 500 tail-hedging strategy constructed by the LDA model can be seen in Figure 3. The strategy outperforms the benchmark S&P 500 and is able to reduce maximum drawdown by approximately 7%. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/cde5c9f6-986e-4f0b-8389-14ccff97ddea/7daf7d905e53b64cba428e7d5e274f624e35fb716ad452b2461d5e68eab8f387.jpg)



Figure 3: Out-of-sample performance of the S&P 500 tail-hedging strategy constructed based on the LDA model's regime signals.


When applied to crude oil, all six models achieve higher cumulative returns and lower maximum drawdowns in comparison to the passive buy-hold strategy. The tail-hedging strategy preserves an average beta of 0.523 and produces an average alpha of 13.39% across models. Figure 4 displays the cumulative return of the crude oil tail-hedging strategy constructed by the decision tree model. The strategies resulting from the decision tree and AdaBoost models exhibit nearly identical performance. The strategies realize an improved cumulative return of approximately 132% compared to the benchmark's 20.41% over the same period, as well as an annualized alpha of 20.53%. The strategies also reduce maximum drawdown by 6.7%. The performance of the tail-hedging strategy produced by each model for crude oil and the S&P 500 is displayed in Table 4. Despite identifying medium–strong linear relationships between in-sample performance metrics and the performance of the resulting strategies, as assessed by Pearson's correlation coefficient (0.5 < |R|), none are statistically significant at $\alpha = .05$ . As reflected in the results, the tail-hedging strategy constructed through this hybrid learning approach would allow investors to attain superior risk-adjusted returns over the out-of-sample period compared to buy and hold passive investment strategies. The regime switching based tail-hedging strategy displays the capability to effectively generate positive alpha for investors while sustaining beta exposure to underlying assets. 


Table 3: Out-of-sample performance for benchmark asset buy-hold strategies.


<table><tr><td>Benchmark Asset</td><td>Cumulative Return (%)</td><td>Annualized Expected Return (%)</td><td>Annualized Volatility (%)</td><td>Daily Return Skewness</td><td>Daily Return Kurtosis</td><td>Maximum Drawdown (%)</td></tr><tr><td>S&amp;P 500</td><td>158.63</td><td>8.62</td><td>15.67</td><td>-0.080</td><td>11.958</td><td>27.95</td></tr><tr><td>Crude Oil</td><td>20.41</td><td>-17.72</td><td>38.77</td><td>-1.177</td><td>16.095</td><td>81.29</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/cde5c9f6-986e-4f0b-8389-14ccff97ddea/cfc6cc9d5861fb5f44abc82fb568d4f571ef9147cdb36653c1f7434c81d43b53.jpg)



Figure 4: Out-of-sample performance of the crude oil tail-hedging strategy constructed based on the decision tree model's regime signals.


## 5.2 Strategy 2: Tactical Allocation

Tactical allocation acts to hedge investors during crisis periods and minimize risk through multi-asset diversification, while providing investors with reasonable exposure to equity risk premium during non-crisis periods. The strategy rotates between two different portfolios. The active portfolio at a given point in time is dependent on the detected regime. Throughout regime 1, the strategy maintains a traditional 60/40 portfolio consisting of the S&P 500 and U.S. treasury bonds. During regime 2, the strategy switches to a portfolio of short positions on the S&P 500 and crude oil, and long positions in gold and U.S. treasury bonds, each allocated 25% of the total portfolio weight. All six models were able to achieve higher cumulative returns and lower maximum drawdowns in comparison to the benchmark (S&P 500). The strategy produces an average alpha of 8.24% across all models and an average beta of approximately 0.248. Figure 5 displays the simulated portfolio cumulative return for the tactical allocation strategy using the decision tree model for signal generation. Again, the portfolio performance resulting from the regime detection of both the decision tree and AdaBoost models is nearly identical, and exceeds those of the benchmark. The resulting strategies attain a significant reduction in maximum drawdown and annualized volatility with differences of 17.83% and 5.92%, respectively. Both strategies also show an increase of 2.91% in annualized expected return relative to the S&P 500. Overall, the tactical allocation strategy is able to generate positive alpha while maintaining low beta exposure to the equity market. As results show, the strategy would enable investors to capture superior risk-adjusted returns compared to the S&P 500 benchmark over the out-of-sample period. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/cde5c9f6-986e-4f0b-8389-14ccff97ddea/57c5596ffaa3a8b4591d3302dca8b6c10cbe5492f190f56ea32b0c7cfcd6af16.jpg)



Figure 5: Out-of-sample performance of the tactical allocation strategy constructed based on the decision tree model's regime signals.


The performance statistics for the tactical allocation strategy produced by each model are shown in Table 5. The accuracy of a model, assessed by cross-validation on the in-sample data, displays a strong correlation with the cumulative return $P = .04$ , annualized expected return $P = .04$ , and maximum drawdown $P = .01$ of the tactical allocation strategy; these correlations are 0.836, 0.832, and -0.905, all of which are statistically significant at $\alpha = .05$ . Statistically significant relationships are also found between the F1 score and the aforementioned trading strategy performance statistics; the correlations are 0.836, 0.832, -0.905, while the P-values are P = .04, P = .04, and P = .01, respectively. This implies that a notable association exists between these performance evaluation metrics and the out-of-sample performance of the tactical allocation strategy. In practice, one might seek a classification model which is able to maximize accuracy and F1 score in order to construct the optimal tactical allocation strategy based on regime switches. In-sample AUC shows no statistically significant relationships with any measure of performance for the tactical allocation trading strategy. 


Table 4: Tail-hedging strategy out-of-sample performance for the S&P 500 and crude oil.


<table><tr><td colspan="2"></td><td>Cumulative Return (%)</td><td>Annualized Expected Return (%)</td><td>Annualized Volatility (%)</td><td>Daily Return Skewness</td><td>Daily Return Kurtosis</td><td>Annualized Alpha (%)</td><td>Beta</td><td>Maximum Drawdown (%)</td></tr><tr><td rowspan="6">S&amp;P 500</td><td>LDA</td><td>202.72</td><td>12.54</td><td>15.66</td><td>0.054</td><td>11.968</td><td>7.02</td><td>0.641</td><td>20.63</td></tr><tr><td>QDA</td><td>164.14</td><td>9.16</td><td>15.67</td><td>-0.112</td><td>11.967</td><td>1.79</td><td>0.856</td><td>33.50</td></tr><tr><td>Logistic Regression</td><td>158.32</td><td>8.59</td><td>15.67</td><td>-0.809</td><td>12.059</td><td>4.78</td><td>0.443</td><td>20.64</td></tr><tr><td>Decision Tree</td><td>168.89</td><td>9.63</td><td>15.67</td><td>-0.796</td><td>12.077</td><td>6.04</td><td>0.416</td><td>17.46</td></tr><tr><td>AdaBoost</td><td>168.89</td><td>9.63</td><td>15.67</td><td>-0.796</td><td>12.077</td><td>6.04</td><td>0.416</td><td>17.46</td></tr><tr><td>Naive Bayes</td><td>142.36</td><td>6.89</td><td>15.68</td><td>-0.861</td><td>12.03</td><td>5.04</td><td>0.214</td><td>24.40</td></tr><tr><td rowspan="6">Crude Oil</td><td>LDA</td><td>67.22</td><td>1.05</td><td>38.78</td><td>1.192</td><td>16.196</td><td>10.85</td><td>0.553</td><td>74.59</td></tr><tr><td>QDA</td><td>33.29</td><td>-9.90</td><td>38.78</td><td>-1.086</td><td>16.142</td><td>5.29</td><td>0.857</td><td>75.14</td></tr><tr><td>Logistic Regression</td><td>76.59</td><td>3.13</td><td>38.78</td><td>1.258</td><td>16.179</td><td>12.33</td><td>0.519</td><td>74.59</td></tr><tr><td>Decision Tree</td><td>132.15</td><td>11.86</td><td>38.78</td><td>1.253</td><td>16.119</td><td>20.53</td><td>0.489</td><td>74.59</td></tr><tr><td>AdaBoost</td><td>132.15</td><td>11.86</td><td>38.78</td><td>1.253</td><td>16.119</td><td>20.53</td><td>0.489</td><td>74.59</td></tr><tr><td>Naive Bayes</td><td>95.96</td><td>6.73</td><td>38.78</td><td>1.288</td><td>16.151</td><td>10.80</td><td>0.229</td><td>67.86</td></tr></table>

## 6 CONCLUSION AND FUTURE WORKS

In this work, we present a novel approach for the detection of regime switches using a combination of unsupervised and supervised learning techniques. Specifically, we show that cluster analysis with classification provides a hybrid framework for regime detection. We identify and exhibit one method for inferring and subsequently automating the selection of the number of regimes which exist through the average silhouette width method. Further, we demonstrate a practical application of this approach through the construction of two successful trading strategies in the financial markets; though, specific use-cases may be adopted within areas of portfolio management and systematic trading. 

Building on the use of economic data in this study, assessing the effectiveness of this approach using asset specific factors and statistics may be a natural extension of this work. In this study we use differenced time-series which lack the memory that is often intrinsically embedded within time-series data – other pre-processing steps may be worth exploring in order to maximize information retention. Due to the approach being model agnostic it is desirable to consider the use of other measures of distance and dissimilarity, as well as clustering methods such as those which are hierarchical or density-based. Assessing how these methods influence the number and types of regimes inferred from data could allow for the development of systems configured specifically for collections of related time-series. Although associations are found between in-sample performance metrics and out-of-sample trading strategy performance in this study, the significance of such relationships is not directly evident. Further study is necessary to better understand these relationships, and would increase the confidence in the out-of-sample performance produced by trading strategies resultant from the proposed hybrid framework. 

## ACKNOWLEDGMENTS

We would like to express a deep gratitude to Professor Reinaldo Sanchez-Arias at Florida Polytechnic University and Professor Jiandong Ren at Western University for valuable suggestions in the development of this work. We also wish to acknowledge Shivam Sharma, as well as colleagues Neha Gulati and Carol Zhai for the constructive comments and useful critiques of this research. 

## REFERENCES



[1] Amir H Alizadeh, Nikos K Nomikos, and Panos K Pouliasis. 2008. A Markov regime switching approach for hedging energy commodities. Journal of Banking & Finance 32, 9 (2008), 1970–1983. 





[2] Turan G Bali, K Ozgur Demirtas, and Haim Levy. 2008. Nonlinear mean reversion in stock prices. Journal of Banking & Finance 32, 5 (2008), 767–782. 





[3] Abhijit V Banerjee. 1992. A simple model of herd behavior. The quarterly journal of economics 107, 3 (1992), 797–817. 





[4] Michael Bock and Roland Mestel. 2009. A regime-switching relative value arbitrage rule. In Operations research proceedings 2008. Springer, 9–14. 





[5] Jun Cai. 1994. A Markov model of switching-regime ARCH. Journal of Business & Economic Statistics 12, 3 (1994), 309–316. 





[6] Kausik Chaudhuri and Yangru Wu. 2003. Mean reversion in stock prices: evidence from emerging markets. Managerial Finance (2003). 





[7] Fei Chen, Francis X Diebold, and Frank Schorfheide. 2013. A Markov-switching multifractal inter-trade duration model, with application to US equities. Journal of Econometrics 177, 2 (2013), 320–342. 





[8] Shyh-Wei Chen and Jin-Lung Lin. 2000. Switching ARCH models of stock market volatility in Taiwan. Advances in Pacific Basin business, economics and finance 4 (2000), 1–21. 





[9] Thomas C Chiang and Dazhi Zheng. 2010. An empirical analysis of herd behavior in global stock markets. Journal of Banking & Finance 34, 8 (2010), 1911–1921. 





[10] Marco Cipriani and Antonio Guarino. 2008. Herd behavior and contagion in financial markets. The BE Journal of Theoretical Economics 8, 1 (2008). 





[11] Albert Corhay, A Tourani Rad, and J-P Urbain. 1993. Common stochastic trends in European stock markets. Economics Letters 42, 4 (1993), 385–390. 





[12] Min Dai, Qing Zhang, and Qiji Jim Zhu. 2010. Trend following trading under a regime switching model. SIAM Journal on Financial Mathematics 1, 1 (2010), 780–810. 





[13] Charles Engel and James D Hamilton. 1990. Long swings in the dollar: Are they in the data and do markets know it? The American Economic Review (1990), 689–713. 




Table 5: Tactical allocation strategy out-of-sample performance.


<table><tr><td></td><td>Cumulative Return (%)</td><td>Annualized Expected Return (%)</td><td>Annualized Volatility (%)</td><td>Daily Return Skewness</td><td>Daily Return Kurtosis</td><td>Annualized Alpha (%)</td><td>Beta</td><td>Maximum Drawdown (%)</td></tr><tr><td>LDA</td><td>193.55</td><td>11.07</td><td>9.96</td><td>2.623</td><td>47.864</td><td>8.40</td><td>0.309</td><td>11.51</td></tr><tr><td>QDA</td><td>167.73</td><td>8.75</td><td>9.67</td><td>0.159</td><td>18.509</td><td>4.92</td><td>0.445</td><td>17.59</td></tr><tr><td>Logistic Regression</td><td>182.09</td><td>10.06</td><td>9.69</td><td>2.669</td><td>50.799</td><td>8.13</td><td>0.224</td><td>11.51</td></tr><tr><td>Decision Tree</td><td>199.46</td><td>11.53</td><td>9.75</td><td>2.635</td><td>49.385</td><td>9.73</td><td>0.209</td><td>10.12</td></tr><tr><td>AdaBoost</td><td>199.46</td><td>11.53</td><td>9.75</td><td>2.635</td><td>49.385</td><td>9.73</td><td>0.209</td><td>10.12</td></tr><tr><td>Naive Bayes</td><td>173.45</td><td>9.35</td><td>10.37</td><td>1.933</td><td>38.015</td><td>8.55</td><td>0.093</td><td>14.96</td></tr></table>



[14] Wai Mun Fong and Kim Hock See. 2002. A Markov switching model of the conditional volatility of crude oil futures prices. Energy Economics 24, 1 (2002), 71–95. 





[15] Rene Garcia and Pierre Perron. 1996. An analysis of the real interest rate under regime shifts. The Review of Economics and Statistics (1996), 111–125. 





[16] Thomas H Goodwin. 1993. Business-cycle analysis with a Markov-switching model. Journal of Business & Economic Statistics 11, 3 (1993), 331–339. 





[17] Stephen F Gray. 1996. Modeling the conditional distribution of interest rates as a regime-switching process. Journal of Financial Economics 42, 1 (1996), 27–62. 





[18] James D Hamilton. 1989. A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica: Journal of the Econometric Society (1989), 357–384. 





[19] James D Hamilton. 1990. Analysis of time series subject to changes in regime. Journal of econometrics 45, 1-2 (1990), 39–70. 





[20] James D Hamilton and Raul Susmel. 1994. Autoregressive conditional heteroskedasticity and changes in regime. Journal of econometrics 64, 1-2 (1994), 307–333. 





[22] John A Hartigan and Manchek A Wong. 1979. Algorithm AS 136: A k-means clustering algorithm. Journal of the Royal Statistical Society. Series C (Applied Statistics) 28, 1 (1979), 100–108. 





[21] Mary R Hardy. 2001. A regime-switching model of long-term stock returns. North American Actuarial Journal 5, 2 (2001), 41–53. 





[23] Harold Hotelling. 1933. Analysis of a complex of statistical variables into principal components. Journal of educational psychology 24, 6 (1933), 417. 





[24] Ian Jolliffe. 2002. Principal component analysis. Springer Verlag, New York. 





[25] Ian T Jolliffe and Jorge Cadima. 2016. Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 374, 2065 (2016), 20150202. 





[26] Leonard Kaufman and Peter J Rousseeuw. 2009. Finding groups in data: an introduction to cluster analysis. Vol. 344. John Wiley & Sons. 





[27] Hans-Martin Krolzig et al. 1996. Statistical analysis of cointegrated VAR processes with Markovian regime shifts. Citeseer. 





[28] Shu-Hsien Liao, Hsu-hui Ho, and Hui-wen Lin. 2008. Mining stock category association and cluster on Taiwan stock market. Expert Systems with Applications 35, 1-2 (2008), 19–29. 





[29] Ying Ma, Leonard MacLean, Kuan Xu, and Yonggan Zhao. 2011. A portfolio optimization model with regime-switching risk factors for sector exchange traded funds. Pac J Optim 7, 2 (2011), 281–296. 





[30] KV Mardia, JT Kent, and JM Bibby. 1979. Multivariate analysis. Probability and Mathematical Statistics, London: Academic Press, 1979 (1979). 





[31] Timothy D Mount, Yumei Ning, and Xiaobin Cai. 2006. Predicting price spikes in electricity markets using a regime-switching model with time-varying parameters. Energy Economics 28, 1 (2006), 62–80. 





[32] SR Nanda, Biswajit Mahanty, and MK Tiwari. 2010. Clustering Indian stock market data for portfolio management. Expert Systems with Applications 37, 12 (2010), 8793–8798. 





[33] Andreas Park and Hamid Sabourian. 2011. Herding and contrarian behavior in financial markets. Econometrica 79, 4 (2011), 973–1026. 





[34] Raymond McTaggart, Gergely Daroczi, and Clement Leung. 2019. Quandl: API Wrapper for Quandl.com. https://CRAN.R-project.org/package=Quandl R package version 2.10.0. 





[35] Peter J Rousseeuw. 1987. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics 20 (1987), 53–65. 





[36] Lucio Sarno and Giorgio Valente. 2000. The cost of carry model and regime shifts in stock index futures markets: An empirical investigation. Journal of Futures Markets: Futures, Options, and Other Derivative Products 20, 7 (2000), 603–624. 





[37] Nop Sopipan, Pairote Sattayatham, Bhusana Premanode, et al. 2012. Forecasting volatility of gold price using markov regime switching and trading strategy. Journal of Mathematical Finance 2, 01 (2012), 121. 





[38] Lin Tan, Thomas C Chiang, Joseph R Mason, and Edward Nelling. 2008. Herding behavior in Chinese stock markets: An examination of A and B shares. Pacific-Basin Finance Journal 16, 1-2 (2008), 61–77. 





[40] Xiao Zhong and David Enke. 2017. A comprehensive cluster and classification mining procedure for daily stock market return forecasting. Neurocomputing 267 (2017), 152–168. 





[39] Zhipeng Yan, Yan Zhao, and Libo Sun. 2012. Industry herding and momentum. The Journal of Investing 21, 1 (2012), 89–96. 

