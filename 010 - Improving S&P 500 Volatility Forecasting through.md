# Improving S&P 500 Volatility Forecasting through Regime-Switching Methods

Ava C. Blake $^{*}$ Nivika A. Gandhi $^{\dagger}$ Anurag R. Jakkula $^{\ddagger}$ 

October 7, 2025 

## Abstract

Accurate prediction of financial market volatility is critical for risk management, derivatives pricing, and investment strategy. In this study, we propose a multitude of regime-switching methods to improve the prediction of S&P 500 volatility by capturing structural changes in the market across time. We use eleven years of SPX data, from May 1st, 2014 to May 27th, 2025, to compute daily realized volatility (RV) from 5-minute intraday log returns, adjusted for irregular trading days. To enhance forecast accuracy, we engineered features to capture both historical dynamics and forward-looking market sentiment across regimes. The regime-switching methods include a soft Markov switching algorithm to estimate soft-regime probabilities; a distributional spectral clustering method that uses XGBoost to assign clusters at prediction time; and a coefficient-based soft regime algorithm that extracts HAR coefficients from time segments segmented through the Mood test and clusters through Bayesian GMM for soft regime weights and uses XGBoost to predict regime probabilities. Models were evaluated across three time periods—before, during, and after the COVID-19 pandemic. The coefficient-based clustering algorithm outperformed all other models, including the baseline autoregressive model, during all time periods. Additionally, each model was evaluated on its recursive forecasting performance for 5 and 10 day horizons during each time period. The findings of this study demonstrate the value of regime-aware modeling frameworks and soft clustering approaches in improving volatility forecasting, especially during periods of heightened uncertainty and structural change. 

## 1 Introduction

Traditional econometric models such as the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model and the Heterogeneous Autoregressive (HAR) model rely on strong assumptions of stationarity and homogeneity, which often fail in the presence of structural breaks, financial crises, or rapidly changing market conditions. The HAR model, introduced by Corsi (2009) in 2009, remains a widely used benchmark for modeling realized volatility using high-frequency data, relying on past volatility values over different time horizons (daily, weekly, monthly) to predict future volatility. The model takes the following form, 

$$
R V _ {t} = \beta_ {0} + \beta_ {d} \cdot R V _ {t - 1} + \beta_ {w} \cdot \overline {{R V}} _ {t - 1} ^ {(w)} + \beta_ {m} \cdot \overline {{R V}} _ {t - 1} ^ {(m)} + \varepsilon_ {t}\tag{1}
$$

where $\beta_{0},\beta_{d},\beta_{w},\beta_{m}$ are the HAR parameters to be estimated, and $\varepsilon_{t}$ is the error term $^{1}$ . However, its fixed coefficient structure limits its ability to adapt to shifts between low and high volatility regimes, which are commonly observed in empirical financial data. Later findings suggest that this static dynamic structure does not fully capture the behavior of volatility, especially in regimes that switch between calm and turbulent periods. 

Recent literature has begun to address this limitation by incorporating regime-switching dynamics into volatility forecasting models. For example, Zhang et al. (2020) implement a Markov regime-switching model in which HAR model parameters depend on unobservable market states. Their framework uses a two-regime Markov chain with statistically significant transition probabilities, showing that distinct regression coefficients between high- and low-volatility regimes improve model performance. The authors concluded that such regime-aware modeling outperforms the traditional HAR model, providing a strong justification for our exploration of regime-switching frameworks. However, Markov models often rely on hard regime assignments that ignore uncertainty in state transitions. 

Our research builds on these findings by incorporating a more flexible soft-classification framework into the regime-switching architecture. Unlike Zhang et al. (2020), we use posterior regime probabilities—soft weights—to construct forecasts that better reflect uncertainty in regime identification and transition dynamics. In our approach, each regime-specific model learns from the full dataset but emphasizes different segments according to probabilistic weights. 

We further explore clustering as a form of regime-switching in our model architecture. Clustering aims to identify volatility regime labels for training and inference, raising key methodological questions: How is time-series data treated? How is the number of regimes determined? What clustering methods should be used? And how can clustering be validated beyond final regression accuracy? A notable challenge in this context is the breakdown of the i.i.d. assumption, since volatility data exhibit strong temporal dependencies. Nonetheless, clustering is employed precisely to capture these dependencies, as regimes emerge over time. 

Prakash et al. (2021) address this by using the Mood Test to segment time-series data based on variance shifts—a defining feature of regime transitions. They construct a distance matrix in Wasserstein space to quantify the effort required to morph one time-series segment into another, and apply spectral clustering to handle the nonlinear, sparse structure of financial data. Our second model architecture incorporates and advances these techniques by applying them directly to realized volatility forecasting. We extend their pipeline by training regime-specific HAR models on the clustered segments and using an XGBoost classifier to map current features to regime probabilities for forecasting. 

Finally, we propose a third, novel model that addresses limitations of distributional-based clustering by clustering on regression coefficients instead. Specifically, we segment the time series, extract Ordinary Least Squares (OLS) regression coefficients for each segment, and cluster based on these coefficients. These vectors reflect how input features relate to future volatility, providing a more interpretable and dynamic basis for identifying regimes. 

## 2 Data Scope and Collection

Our dataset comprises intraday price data in 5-minute intervals from the Standard & Poor's 500 (or S&P 500) Index. The 5-minute intervals were used in order to thoroughly capture intraday price dynamics and fluctuations. The data spans across the past 11 years from May 1st, 2014 to May 27th, 2025, accessed via the Bloomberg Terminal. Using 5-minute closing prices, we calculate intraday log-returns as: 

$$
r _ {t, i} = \ln \left(\frac {P _ {t , i}}{P _ {t , i - 1}}\right)\tag{2}
$$

where $P_{t,i}$ is the price at the $i^{th}$ 5-minute interval on day t. Log returns were used as opposed to raw returns to ensure time-additivity and stabilize variance, which is a standard approach in financial modeling. These returns are then aggregated to compute daily realized volatility (RV), following calculation and normalization techniques widely accepted and used by Zhang et al. (2020), Luo et al. (2022), Ding et al. (2025), Li et al. (2024), Hu et al. (2025), and many others. Additionally, RV values were adjusted to account for shorter trading days. 

$$
R V _ {t} = \sqrt {\frac {N}{n} \sum_ {i = 1} ^ {n} r _ {t , i} ^ {2}}\tag{3}
$$

Here, n is the number of intraday returns observed on day t (which may vary due to holidays), and N is the standard number of returns in a full trading day (lasting from 09:30 EST until 16:00 EST)—78 for intraday data only, or 79 including overnight returns. This scaling ensures comparability of RV across days with varying lengths. The square root of realized variance was taken to obtain realized volatility, aligning the scale with standard deviation and improving interpretability in forecasting models. To ensure alignment across all variables, only days with available SPX data were retained in the dataset. Feature values were included only if they corresponded to these days. 

For any missing feature values on retained days, linear interpolation was used to impute the missing entries. 

## 3 Assumptions and Simplifications

It is assumed that the 5-minute SPX closing prices are accurate and free of market microstructure noise, such as bid-ask bounce or discreteness of price changes. We also treat the closing price in each interval as a fair and consolidated measurement of market activity, ignoring any possible closing anomalies. Our calculation of daily real volatility as in Equation (3) is assumed to capture the full variability of asset prices when returns are sampled at a high frequency. 

The HAR model extends a linear regression framework to forecast realized volatility and thus inherits the core assumptions of linear regression: homoskedasticity, implying constant variance of the residuals, that residuals are independent, and that residuals are normally distributed with a mean of zero. The HAR model further relies on the long- memory behavior of volatility, implying that past high volatility tends to predict present high volatility. In order to utilize a regime-switching Markov model, we assume the following: volatility follows different statistical behaviors (regimes) at different times and that the transitions between these regimes follow a Markov process - the probability of transitioning to the next regime depends only on the present regime. More broadly, we assume that these market regimes can be inferred from observable data. For financial time-series data, distribution shifts of the inputs and outputs often do occur, which is one defect of fixed-regime HAR models. Our paper addresses this issue by incorporating regime switching to account for one such distribution shift. 

## 4 Methodology

The original HAR model was designed to capture realized volatility (RV) behavior across multiple time scales (daily, weekly, and monthly). These lags aim to address the idea that different types of investors operate on different time horizons, more accurately reflecting market dynamics. After building the HAR model, we assessed several statistics to evaluate the assumptions necessary for Ordinary Least Squares Regression. The Omnibus and Jarque-Bera tests strongly reject the null hypothesis of normally distributed residuals. The skewness and kurtosis values further reveal right-skewed and heavy-tailed residuals—features commonly observed in financial time series. Despite this, the Durbin-Watson statistic suggests no meaningful autocorrelation in the residuals. Additionally, a low condition number implies minimal multicollinearity among lagged volatility features. Overall, while the OLS HAR model captures some temporal structure, its failure to account for non-normality and fat tails highlights the need for regime-switching dynamics. Additionally, the ADF (Augmented Dickey-Fuller) test was performed on all input features to assess their stationarity. All features were found to be stationary. 

## 4.1 Feature Engineering

We extend the HAR model by introducing a dual-memory structure that captures both historical volatility patterns and forward-looking market sentiment. Our model applies HAR-style lags to implied volatility features (VIX), allowing the model to respond to shifts in investor expectations. As implied volatility reflects future expectations of market volatility, it represents a significant forward-looking measure and can provide information not contained in historical past volatility. 

We introduce three additional features to capture nonlinear and asymmetric market behavior. We used the daily closing prices previously obtained to determined daily values for Jump Variation and Realized Kurtosis. Our engineered structure thus incorporates: 

1. Lagged VIX values (daily, 5-day, 22-day): market fear over varying time horizons 

2. Realized kurtosis: Measures tail risk and extreme return behavior 

3. Jump variation: Isolates large discontinuous price moves from continuous volatility 

This feature design allows the model to adapt across regimes by integrating both behavioral and structural signals, improving forecast accuracy under changing market conditions. $^{2}$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/106d2e7b48fb484606fd9aa5af12e30ae2bc01d8f5517a463cde2e1f2fbd9247.jpg)



Figure 1: Feature-engineered standard HAR model over the eleven-year period.


We evaluated the standard HAR model against our feature-engineered model across the entire eleven-year period. Our findings establish that the feature-engineered model outperforms the standard HAR model in terms of Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE). Figure 1 illustrates the time series forecast from the feature-engineered HAR, demonstrating its robust ability to capture granular movements and some abrupt spikes. 

<table><tr><td>Model</td><td>MAPE</td><td>MSE</td></tr><tr><td>HAR</td><td>27.72</td><td>8.85</td></tr><tr><td>Feature-Engineered HAR</td><td>26.37</td><td>8.35</td></tr></table>


Table 1: Models predict in 5-day increments. MSE values have been divided by 1,000,000 to improve numerical interpretability.


Incorporating these additional features with a regime-switching model takes the following form: 

$$
\begin{array}{r} R V _ {t} = \beta_ {0, S _ {t}} + \beta_ {d, S _ {t}} \cdot R V _ {t - 1} + \beta_ {w, S _ {t}} \cdot \overline {{R V}} _ {t - 1} ^ {(w)} + \beta_ {m, S _ {t}} \cdot \overline {{R V}} _ {t - 1} ^ {(m)} + \gamma_ {d, S _ {t}} \cdot V I X _ {t - 1} \\ + \gamma_ {w, S _ {t}} \cdot V I X _ {t - 1} ^ {(w)} + \gamma_ {m, S _ {t}} \cdot V I X _ {t - 1} ^ {(m)} + \zeta_ {S _ {t}} \cdot K T S _ {t - 1} + \eta_ {S _ {t}} \cdot J M P _ {t - 1} + \varepsilon_ {t} \end{array}\tag{4}
$$

where $VIX$ is the Chicago Board of Exchange's implied volatility, $KTS$ is realized kurtosis, and $JMP$ is the jump variation. The model coefficients depend on the unobserved regime state $S_{t}$ . 

## 4.2 Rolling Window Implementation

For each model, we implement a rolling window framework to improve the model's adaptability to changing market conditions. Training on a fixed dataset can often lead to outdated parameter estimates when working with financial time series data. To address this limitation, we retrain our models on a fixed-length window of the past 441 observations (equivalent to 7 fiscal quarters) and generate forecasts for a shorter horizon of 5 or 10 days. We selected a window of 441 days to provide enough historical data for stable parameter estimation while still preserving responsiveness to regime changes. After generating these forecasts, we move the training window one step forward in time. Thus, the model continuously updates to reflect the most current market behavior. In particular, we employ a forecast horizon of 5 days for evaluation of our models during the Pre-COVID and Post-COVID periods, and a 10-day forecast horizon during the COVID period. 

## 4.3 Markov Regime-Switching Model

We aim to capture how volatility behaves differently across changing market conditions by first developing a soft-regime-switching model. Instead of training a single global model, we fit a separate HAR model for each regime. A Hidden Markov Model (HMM) identifies these regimes and assigns soft probabilities to each time point, reflecting the likelihood of being in each regime. Our final forecast is therefore a weighted blend of the regime-specific HAR predictions, where the weights depend on the inferred regime probabilities. 

The Hidden Markov Model (HMM) is a statistical model that represents a system with unobserved states, in this case, regimes, that evolve according to a Markov process. This framework looks to address the goal of accounting for latent structural shifts in market dynamics. Our algorithm aligns with previous work that incorporates the EM algorithm, such as Zhang et al. (2020). 

In each rolling training window, features X and target RV array y are z-score normalized, and a smoothed version of $y_{t}$ , denoted $\tilde{y}_{t}$ , is computed using a 5-day moving average. This computation occurs so that the HMM captures structural changes rather than noise. A Gaussian HMM with K regimes is fitted to $\tilde{y}_{1:t}$ . Thus, the HMM models volatility as a sequence generated by an unobserved regime process, with each regime emitting Gaussian-distributed observations: 

$$
\tilde {y} _ {t} \mid z _ {t} = k \sim \mathcal {N} (\mu_ {k}, \sigma_ {k} ^ {2})\tag{5}
$$

This HMM fitting process outputs two probabilistic components of the algorithm: posterior regime probabilities and transition matrix entries. The posterior regime probability at time t, defined by $\gamma_{t}(k)=P(z_{t}=k\mid\tilde{y}_{1:T})$ , is a backward-looking, data-informed probability that describes how likely a regime k is at time t given all observed data up to the last time in the training window, T. The posterior regime probabilities provide the soft weights for fitting the regime-specific weighted least squares models at training time, as seen in the following minimization function: 

$$
\min _ {\{\beta^ {(k)} \}} \sum_ {t = 1} ^ {T} \gamma_ {t} (k) \left(y _ {t} - x _ {t} ^ {\top} \beta^ {(k)}\right) ^ {2}\tag{6}
$$

If the data is sparse, the algorithm fallbacks to unweighted OLS. The transition matrix entries, with each entry expressed as $T_{ij} = P(z_t = j \mid z_{t-1} = i)$ , gives the probability of transitioning from regime i at time t - 1 to regime j at time t at prediction time. For each step-ahead forecast h, the final prediction is a regime-weighted average of the HAR model outputs: 

$$
\hat {y} _ {t + h} = \sum_ {k = 1} ^ {K} p _ {t + h} (k) \cdot x _ {t + h} ^ {\top} \beta^ {(k)}.\tag{7}
$$

The Markov switching mechanism is driven solely by the smoothed realized volatility series $\tilde{y}_{t}$ , thus allowing regime inference to focus on structural shifts in volatility dynamics. However, the subsequent regime-specific HAR models are fitted using a larger feature set which includes lagged RV averages, lagged VIX averages, jump variation, and kurtosis. These features help to capture a broader range of market characteristics within each inferred regime. 

## 4.4 Distributional Clustering

Unlike the Hidden Markov Model, which assumes that each regime follows a specific statistical distribution, our distributional clustering approach does not make assumptions about the underlying form of volatility behavior. Instead, this method identifies structural shifts in the data using a statistical test and then clusters based on the distribution of features. We fit a separate model for each identified regime and use a classifier to assign data points to the most likely regime. 

The input features $X$ and target RVs $y$ are first z-score standardized. We apply the Mood's Median Test, a statistical test that detects distributional changes in variance, to segment the data. For each time $t$ , we test for two-sample scale differences over a rolling window $w$ : 

$$
H _ {0}: \mathrm{Var} (\tilde {y} _ {t - w: t}) = \mathrm{Var} (\tilde {y} _ {t: t + w})\tag{8}
$$

$$
H _ {1}: \mathrm{Var} (\tilde {y} _ {t - w: t}) \neq \mathrm{Var} (\tilde {y} _ {t: t + w})\tag{9}
$$

The time points where the null is rejected are labeled as change points. These change points divide the time series into segments. Each segment $s_{i}$ is represented by the joint distribution over features and target: $D_{i} = \{(\tilde{x}_{t}, \tilde{y}_{t}) \mid t \in s_{i}\}$ . Then, we compute the pairwise squared 2-Wasserstein distances between time segments according to the following equation, 

$$
W _ {i j} ^ {2} = W _ {2} ^ {2} (D _ {i}, D _ {j}) = \min _ {\pi \in \Pi (a, b)} \sum \| u - v \| ^ {2} \pi (u, v)\tag{10}
$$

where $\pi$ is the optimal transport plan, and a, b are uniform weights over the two segments. The Wasserstein distance represents a similarity metric between two probability distributions. The result of this computation is a symmetric distance matrix W. Then, we convert pairwise distances into pairwise similarities using the equation below: 

$$
K _ {i j} = \exp \left(- \frac {W _ {i j} ^ {2}}{2 \sigma^ {2}}\right)\tag{11}
$$

We apply spectral clustering on the kernel matrix K. This process produces segment-level regime labels from 1 through number of regimes K. These labels are then projected to the individual time-series points in each time segment. 

For each regime k, we fit a separate OLS model for each cluster. We then train an XGBoost classifier on the rolling training window to map the feature values of the individual data points to the segment-level cluster labels $^{3}$ . This classifier is then used to determine hard cluster assignments at prediction time. The output is rescaled through the inverse of z-score normalization. 

This model is designed to cluster on the daily, weekly, and monthly RV and VIX averages, however, the prediction step incorporates a larger feature set. In particular, the HAR forecasting models and the XGBoost classifier both include the jump variation, which helps capture sudden market shocks that may not be reflected in the smoothed volatility features used for clustering. 

## 4.5 Coefficient-Based Soft Clustering

As opposed to our prior two methods, the coefficient-based soft clustering approach does not define regimes based on patterns in the observed data itself. Instead, this approach identifies regimes according to how the relationship between features and volatility evolves over time. We then segment the series based on variance shifts, extract regression coefficients from each segment, and cluster them to uncover distinct regimes. Forecasts are generated by weighting regime-specific predictions according to probabilities output by a trained classifier. 

The input features $X$ and target RVs $y$ are z-score standardized. We apply the Mood's Median Test to detect statistically significant changes in variance and thereby identify segment boundaries. For each time $t$ , we test for two-sample scale differences over a rolling window $w$ : 

$$
H _ {0}: \mathrm{Var} (\tilde {y} _ {t - w: t}) = \mathrm{Var} (\tilde {y} _ {t: t + w})\tag{12}
$$

$$
H _ {1}: \mathrm{Var} (\tilde {y} _ {t - w: t}) \neq \mathrm{Var} (\tilde {y} _ {t: t + w})\tag{13}
$$

The time points where the null is rejected are labeled as change points. These change points divide the time series into segments. Each segment $s_{i}$ is represented by the joint distribution over features and target: $D_{i} = \{(\tilde{x}_{t}, \tilde{y}_{t}) \mid t \in s_{i}\}$ . For each segment $s_{i}$ , we fit a linear OLS model: 

$$
y _ {t} = \beta_ {0} ^ {(i)} + \mathbf {x} _ {t} ^ {\top} \pmb {\beta} ^ {(i)} + \varepsilon_ {t}, \quad \mathrm{for} t \in s _ {i}\tag{14}
$$

The coefficient vectors serve as descriptors of local dynamics. If OLS fails due to rank deficiency, ridge regression is used as a fallback. Principal Component Analysis (PCA) is applied to reduce the dimensionality of the coefficient vectors before clustering. Next, we apply a Bayesian Gaussian Mixture Model to estimate posterior membership probabilities of each regime k: 

$$
\gamma_ {i} ^ {(k)} = P (z _ {i} = k \mid \theta^ {(i)}), \quad \text { with } \quad \sum_ {k = 1} ^ {K} \gamma_ {i} ^ {(k)} = 1\tag{15}
$$

We fit a Weighted Least Squares model on the full training window according to: 

$$
\min _ {\beta^ {(k)}} \sum_ {t = 1} ^ {T} \gamma_ {t} (k) \left(y _ {t} - x _ {t} ^ {\top} \beta^ {(k)}\right) ^ {2}\tag{16}
$$

for each regime k. For out-of-sample prediction, we train an XGBoost classifier on the mean feature vectors of each segment to map inputs to cluster assignments $^{4}$ . At prediction time, the classifier receives the features of a time point as input and outputs a soft probability vector, which 

is used to weight regime-specific predictions: 

$$
\hat {y} t + h = \sum k = 1 ^ {K} p _ {t + h} (k) \cdot x _ {t + h} ^ {\top} \beta^ {(k)}.\tag{17}
$$

The features used in this clustering method consist of lagged RV averages, lagged VIX averages, and realized kurtosis. These inputs capture both persistent volatility patterns and tail risk, enabling the segment-level models to differentiate between regimes with distinct higher-order dynamics. 

## 5 Results and Discussion

Each regime-switching model was tested on three different time periods: pre-COVID, COVID, and post-COVID. The pre-COVID time period ranged from 2014-06-02 to 2018-05-21, COVID ranged from 2018-05-21 to 2020-09-29, and post-COVID ranged from 2020-09-29 to 2025-04-29. Date ranges were determined based on major shifts in market behavior and volatility patterns associated with the onset and recovery phases of the COVID-19 pandemic. Model performance was compared within each time period and evaluated based on Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE). Predictions were made in 5-day increments for the pre-COVID and post-COVID time periods, and 10-day increments for the COVID time period in order to account for heightened volatility and reduced data stability during the crisis period, which warranted a slightly broader forecasting window. 

<table><tr><td>Model</td><td>MAPE</td><td>MSE</td><td># Reg.</td></tr><tr><td>HAR</td><td>27.09</td><td>3.40</td><td>1</td></tr><tr><td>Markov Soft EM</td><td>24.33</td><td>3.12</td><td>3</td></tr><tr><td>Distributional Clustering</td><td>25.89</td><td>3.55</td><td>2</td></tr><tr><td>Coefficient Clustering</td><td>23.92</td><td>3.11</td><td>2</td></tr></table>


Table 2: Models predict in 5-day increments. MSE values have been divided by 1,000,000 to improve numerical interpretability.


<table><tr><td>Model</td><td>MAPE</td><td>MSE</td><td># Reg.</td></tr><tr><td>HAR</td><td>30.13</td><td>35.35</td><td>1</td></tr><tr><td>Markov Soft EM</td><td>31.93</td><td>36.22</td><td>2</td></tr><tr><td>Distributional Clustering</td><td>32.63</td><td>33.59</td><td>2</td></tr><tr><td>Coefficient Clustering</td><td>30.62</td><td>31.91</td><td>2</td></tr><tr><td>HAR</td><td>23.31</td><td>8.63</td><td>1</td></tr><tr><td>Markov Soft EM</td><td>22.45</td><td>7.66</td><td>2</td></tr><tr><td>Distributional Clustering</td><td>23.30</td><td>7.91</td><td>2</td></tr><tr><td>Coefficient Clustering</td><td>22.70</td><td>7.60</td><td>2</td></tr></table>


Table 3: Models predict in 10-day increments. MSE values have been divided by 1,000,000 to improve numerical interpretability.


Table 4: Models predict in 5-day increments. MSE values have been divided by 1,000,000 to improve numerical interpretability. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/6632d32ebaf9c230f941598c49b730e1e9865e7c03675c6f777af3e98635a368.jpg)



(a) Pre-COVID: Coefficient-Based Clustering.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/969e49ada3e7bbde56dfb302adcc577ff8cd727be82733017fd22db1c98a8fad.jpg)



(b) COVID: Coefficient-Based Clustering.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/6f341c6b2c4e6ac15ec49f242c12977e3a1187c0cf7410b9fe5bcb7a3643da0e.jpg)



(c) Post-COVID: Coefficient-Based Clustering.



Figure 2: Forecasts from the Coefficient-Based Clustering model across all periods.


Across all periods, regime-switching models generally outperformed the baseline HAR model. In particular, the Coefficient-Based Clustering model achieved the lowest MSE during all time periods. It consistently reduced MSE compared to the standard HAR model, with improvements of $8.52\%$ (Pre-COVID), $9.73\%$ (COVID), and $11.9\%$ (Post-COVID). MAPE values showed a similar pattern in the pre- and post-COVID periods, confirming improved predictive accuracy for short-term volatility movements. The Markov regime-switching followed closely in the pre-COVID and post-COVID time periods, outperforming the standard non-regime-switching model in both time periods. The highly volatile COVID time period proved to be more difficult for the Markov HAR to predict, and the standard model was marginally more accurate. This may reflect the Markov model's limitations in adapting quickly to sharp, irregular market transitions during crisis periods. However, the Distributional Clustering model and Coefficient-Based Clustering model were better suited to handle abrupt structural breaks and regime shifts in volatility, leading to slightly improved performance during this unstable period. For most models, the optimal number of regimes was between two and three, suggesting that market dynamics can often be meaningfully described by a few distinct volatility states. Overall, the consistent reduction in MSE across all periods suggests that the regime-switching models not only improve average forecast accuracy but also better controls for large mispredictions, which is an important property to address for in volatility forecasting. We believe the Coefficient-Based Clustering method likely performed best because it captures the direct relationship between input features and volatility dynamics, leading to more informative regime distinctions. Figures 2a through 2c illustrate the time series forecasts from the top-performing models in each period. Here, we can see how the Coefficient-Based Clustering model visibly adapts to local structural shifts across time periods. 

## 5.1 Recursive Forecasting

We extend the HAR framework by generating multi-step-ahead forecasts of realized volatility through a recursive procedure, where each forecasted value is fed into the model as an input for predicting the next time step. This allows the model to capture compounding effects and evolving volatility patterns beyond one-day horizons. 

To more accurately capture the interdependent dynamics between realized and implied volatility, we develop a dual-recursive HAR-VIX framework. This approach jointly models and forecasts RV and VIX over time. At each recursive forecasting step, we first generate a forecast for VIX using both its own lags and the RV lags. This forecasted VIX is then used as an input, along with RV lags, to predict RV. The process repeats recursively, allowing the predictions for the two volatility measures to co-evolve and influence each other dynamically. This structure enables the model to capture feedback effects between market expectations (VIX) and realized outcomes (RV), which are often observed during periods of market stress or change. The only features in the recursive models are those which are being forecasted, as using feature's real values introduces look-ahead bias as the model forecasts future days based on previous predictions. The dual-recursive structure thus enables the use of VIX as a feature without introducing look-ahead bias, as it ensures that only forecasted values are used at each step while still allowing for features to enhance model accuracy. 

Both the single and dual-recursive HAR frameworks are implemented within our regime-switching modeling pipelines. For each regime, determined via segmentation and clustering methods previously discussed, we estimate regime-specific model parameters. The dual-recursive approach is particularly well-suited to this regime-aware setting, as it accounts for dynamic interactions between RV and VIX within each structural regime. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/53ee5da21407227d6b1fda7c211aca9779f9bb454e30bda948022b9fadac6d9c.jpg)



(a) Pre-COVID: Dual-Recursive Coefficient-Based Clustering.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/a55221d70b1595c740d63a34e3d66e24f6dd26cb5cbc3e421c8db16d2e98eca7.jpg)



(b) COVID: Dual-Recursive Distributional Clustering.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/e0686c929c27450793281432c4321a7bf56e458f3a5df09198204bf5257deb7d.jpg)



(c) Post-COVID: Dual-Recursive Coefficient-Based Clustering.



Figure 3: Forecasts from the optimally performing 5-day recursive forecasting model across all periods. Models correspond to results in Table 5.


<table><tr><td>Model</td><td colspan="3">Pre-COVID</td><td colspan="3">COVID</td><td colspan="3">Post-COVID</td></tr><tr><td>Single-Recursive</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td></tr><tr><td>Standard HAR</td><td>33.90</td><td>5.10</td><td>1</td><td>35.68</td><td>67.06</td><td>1</td><td>31.22</td><td>16.97</td><td>1</td></tr><tr><td>Markov Soft EM</td><td>33.89</td><td>5.10</td><td>2</td><td>35.34</td><td>64.70</td><td>2</td><td>31.02</td><td>16.67</td><td>2</td></tr><tr><td>Dist. Clustering</td><td>34.97</td><td>5.46</td><td>2</td><td>32.47</td><td>62.49</td><td>2</td><td>32.38</td><td>19.62</td><td>2</td></tr><tr><td>Coeff. Clustering</td><td>34.97</td><td>4.61</td><td>2</td><td>35.23</td><td>65.34</td><td>2</td><td>30.63</td><td>13.57</td><td>2</td></tr><tr><td>Dual-Recursive</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td></tr><tr><td>Standard HAR</td><td>33.28</td><td>4.96</td><td>1</td><td>42.91</td><td>80.53</td><td>1</td><td>28.49</td><td>13.37</td><td>1</td></tr><tr><td>Markov Soft EM</td><td>32.33</td><td>4.91</td><td>2</td><td>54.48</td><td>88.01</td><td>2</td><td>27.55</td><td>12.80</td><td>2</td></tr><tr><td>Dist. Clustering</td><td>32.21</td><td>5.23</td><td>2</td><td>41.70</td><td>57.28</td><td>2</td><td>29.32</td><td>15.00</td><td>2</td></tr><tr><td>Coeff. Clustering</td><td>33.01</td><td>4.46</td><td>2</td><td>41.52</td><td>116.5</td><td>2</td><td>28.67</td><td>12.17</td><td>2</td></tr></table>


Table 5: Model performance for 5-day forecasts during each time period. Divide MSE values by 1,000,000. Performance is compared based on MSE values.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/19a2cdb58604423543cfe5e3df663d21fd0c5ece49a7a84ccb9f933116b8b034.jpg)



(a) Pre-COVID: Dual-Recursive Coefficient-Based Clustering.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/52d676827f0fa6127db3cddb584a23d76fb2f3b1b8e75858d67be0a4ef75a78a.jpg)



(b) COVID: Dual-Recursive Distributional Clustering.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/7d36d148-5aa2-473f-93eb-dc3cf4a6583b/c7b622a402863a355acaf39e8d6d3de6166f49b30ecc7adf716ef52dd47bd95a.jpg)



(c) Post-COVID: Dual-Recursive Coefficient-Based Clustering.



Figure 4: Forecasts from the optimally performing 10-day recursive forecasting model across all periods. Models correspond to results in Table 6.


<table><tr><td>Model</td><td colspan="3">Pre-COVID</td><td colspan="3">COVID</td><td colspan="3">Post-COVID</td></tr><tr><td>Single-Recursive</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td></tr><tr><td>Standard HAR</td><td>38.19</td><td>6.22</td><td>1</td><td>40.03</td><td>101.7</td><td>1</td><td>34.90</td><td>21.48</td><td>1</td></tr><tr><td>Markov Soft EM</td><td>38.18</td><td>6.22</td><td>2</td><td>42.98</td><td>96.96</td><td>2</td><td>34.85</td><td>20.94</td><td>2</td></tr><tr><td>Dist. Clustering</td><td>39.39</td><td>6.47</td><td>2</td><td>38.46</td><td>98.20</td><td>2</td><td>36.70</td><td>23.69</td><td>2</td></tr><tr><td>Coeff. Clustering</td><td>39.45</td><td>5.51</td><td>2</td><td>39.40</td><td>101.7</td><td>2</td><td>33.91</td><td>15.59</td><td>2</td></tr><tr><td>Dual-Recursive</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td><td>MAPE</td><td>MSE</td><td># Reg.</td></tr><tr><td>Standard HAR</td><td>37.83</td><td>6.17</td><td>1</td><td>52.39</td><td>155.0</td><td>1</td><td>32.25</td><td>16.44</td><td>1</td></tr><tr><td>Markov Soft EM</td><td>37.04</td><td>6.15</td><td>2</td><td>60.39</td><td>149.7</td><td>2</td><td>31.41</td><td>16.04</td><td>2</td></tr><tr><td>Dist. Clustering</td><td>34.41</td><td>6.19</td><td>2</td><td>47.79</td><td>77.85</td><td>2</td><td>32.21</td><td>18.30</td><td>2</td></tr><tr><td>Coeff. Clustering</td><td>38.10</td><td>5.50</td><td>2</td><td>55.14</td><td>391.1</td><td>2</td><td>32.02</td><td>14.38</td><td>2</td></tr></table>


Table 6: Model performance for 10-day forecasts during each time period. Divide MSE values by 1,000,000. Performance is compared based on MSE values.


In the 5-day recursive forecasting models, visualized in Figure 3, the Coefficient-based Clustering model implemented with a dual-recursive structure achieves the lowest MSE of $4.46 \times 10^{-6}$ in the Pre-COVID period. This highlights the benefit of allowing regime-specific coefficients to reflect the stable relationships present during calm market conditions. During the COVID period, which is characterized by heightened volatility and structural breaks, the Distributional Clustering model with dual-recursive forecasting delivers the most accurate predictions. This result suggests that segmenting regimes based on distributional properties provides robustness in environments that are marked by sharp changes in volatility behavior. In the Post-COVID period, the Coefficient-based Clustering model again outperforms all others under the dual-recursive framework, with an MSE of $12.17 \times 10^{-6}$ , indicating that this model adapts well to normalized but structurally evolved post-crisis dynamics. 

In the 10-day recursive setting seen in Figure 4, forecasting accuracy generally declines, as expected, due to the increased uncertainty associated with longer horizons. However, the patterns across models remain consistent. The Coefficient-based Clustering model under the dual-recursive structure again achieves the best performance in the Pre-COVID period. During the COVID period, the Distributional Clustering model performs best, mirroring its relative advantage under stress conditions observed in the 5-day forecasts. In the Post-COVID period, the Coefficient-based Clustering model once again leads. 

Across both forecasting horizons, dual-recursive implementations consistently outperform their single-recursive counterparts. By allowing both series (realized volatility and implied volatility) to evolve together and by capturing interdependent dynamics over time, the dual-recursive framework provides a significant forecasting advantage. Notably, the Coefficient-based Clustering model performs best during the more stable Pre- and Post-COVID periods, while the Distributional Clustering model proves more resilient during the turbulent COVID period. Meanwhile, the Markov Soft EM model, while theoretically grounded, underperforms relative to clustering-based methods, especially during volatile periods. The superior performance of clustering-based models under a dual-recursive framework aligns with performances of the models in a non-recursive setting. These outcomes thus underscore the importance of flexible, adaptive structures that can effectively capture the nonlinear and regime-dependent properties of volatility. 

## 6 Limitations

Several methodological limitations should be considered when interpreting the results of these regime-switching frameworks. Firstly, forecast accuracy is sensitive to hyperparameter choices, including the number of regimes, the window size for the Mood test, the dimensionality retained through PCA, and the tuning of the XGBoost classifier. Although these were guided by prior literature and cross-validation, alternate configurations may yield different results. Second, the projection of segment-level regime labels onto individual time points in clustering assumes behavioral homogeneity within each segment, which may oversimplify regime transitions near boundaries. However, this is used as a type of smoothing, and therefore the assumption of behavioral homogeneity also functions as a technique for mitigating microstructure noise. Another limitation to be considered is that in coefficient-based clustering, PCA is used to reduce the dimensionality of local OLS coefficients before clustering. While this helps avoid overfitting, it may obscure complex feature-volatility relationships critical to effective regime identification. Our Markov and distributional regime detection methods also rely heavily on realized volatility dynamics, which, while informative, may omit other structural indicators represented through the HAR features. This is potentially why coefficient-based clustering outperforms in cases, as all three method's HAR models are fitting on the features, however, only the coefficient-based model clusters on feature information as well. Finally, the models exhibit limited robustness to extreme events. Although performance improves relative to baseline models during turbulent periods like COVID, methods such as the HMM may still lag in adapting to sudden structural breaks or nonlinear transitions, as reflected in reduced accuracy for the Markov Soft EM model. 

## 7 Conclusion

Our empirical results demonstrate that regime-aware HAR extensions consistently yield lower forecasting errors than the standard HAR model, confirming the value of incorporating regime-switching dynamics. Specifically, coefficient-based soft clustering effectively captures gradual shifts and identifies structural breaks in volatility distributions, achieving superior performance before, during, and after the COVID time period. The Markov Soft EM model marginally outperforms the HAR model before and after the COVID time period, yet struggles to accurately capture volatility behavior during the highly volatile COVID-19 time period. Distributional clustering likewise demonstrates slight improvements against the HAR model. Inclusion of the VIX as a forward-looking feature was also shown to enhance model responsiveness to shifts in market sentiment, thereby refining predictive accuracy. 

These outcomes illustrate that volatility dynamics exhibit regime-dependent statistical properties, and that flexible models including regime-specific parameters provide a meaningful advantage. In particular, clustering on the relationship between features and next-day volatility offers a novel and advantageous approach to regime-switching forecasting. Among the approaches evaluated, coefficient-based clustering stands out as the most effective method, due to its ability to adapt to nonstationarities in the data and maintain robustness across different market conditions. Its success suggests that modeling the evolving structure of the feature-volatility relationship—rather than static distributions—yields a more accurate and adaptive volatility forecasting framework. 

These results have practical relevance for financial forecasting and risk management. The ability of coefficient-based clustering to adapt to structural breaks and changing feature-volatility relationships suggests potential value in applications such as short-term risk estimation. Moreover, the clustering-based regime-switching approach—especially in its coefficient-driven form—shows promise for generalization across different financial assets, as it relies on relational dynamics rather than asset-specific assumptions. Given the structural similarities in volatility processes across asset classes, extending this framework to equities, commodities, or cryptocurrencies could further validate its utility. 

The regime-switching models show only marginal improvement over the feature-engineered HAR model, as it already performs fairly strongly. Future research could explore hybrid frameworks that integrate clustering and Markov switching, as well as advanced sequential models like LSTMs to capture more complex temporal dependencies and thereby see larger improvements. 

## 8 Acknowledgments

This research was conducted as part of the Columbia Summer Undergraduate Research Experiences in Mathematical Modeling (CSUREMM) program, hosted and supported by the Columbia University Department of Mathematics. We would like to thank our mentors–Professor George Dragomir, Vihan Pandey, and Professor Dobrin Marchev–for their guidance, insight, and support throughout the course of this project. 

## References



Corsi, F. (2009). A simple approximate long-memory model of realized volatility. Journal of Financial Econometrics 7(2), 174–196. 





Ding, Y., D. Kambouroudis, and D. G. McMillan (2025). Forecasting realised volatility using regime-switching models. International Review of Economics & Finance 101(104171). 





Hu, N., X. Yin, and Y. Yao (2025). A novel har-type realized volatility forecasting model using graph neural network. International Review of Financial Analysis 98(103881). 





Li, X., D. Li, Y. Cheng, and W. Li (2024). Forecasting the volatility of educational firms based on har model and lstm models considering sentiment and educational policy. Heliyon 10(19). 





Luo, J., T. Klein, Q. Ji, and C. Hou (2022). Forecasting realized volatility of agricultural commodity futures with infinite hidden markov har models. International Journal of Forecasting 38(1), 51–73. 





Prakash, A., N. James, M. Menzies, and G. Francis (2021, May). Structural clustering of volatility regimes for dynamic trading strategies. Applied Mathematical Finance 28(3), 236–274. 





Zhang, Y., L. Lei, and Y. Wei (2020). Forecasting the chinese stock market volatility with international market volatilities: The role of regime switching. The North American Journal of Economics and Finance 52(101145). 



## Appendix

## A Feature Description


Table 7: Model Feature Description and Calculation


<table><tr><td>Feature</td><td>Description</td><td>Calculation</td></tr><tr><td>Daily Realized Volatility (RV)</td><td>Measures the actual observed volatility of the S&amp;P 500 on a daily basis, computed from high-frequency intraday returns. Reflects short-term market risk and variability.</td><td>See Eq. 3.</td></tr><tr><td>Weekly Realized Volatility (<eq>\overline{RV}^{(w)}</eq>)</td><td>Averages daily realized volatility over the past trading week to capture medium-term volatility trends and smooth short-term noise.</td><td><eq>\frac{1}{5}\sum_{i=0}^{4}RV_{t-i}</eq></td></tr><tr><td>Monthly Realized Volatility (<eq>\overline{RV}^{(m)}</eq>)</td><td>Averages daily realized volatility over the past trading month to identify longer-term volatility patterns and regime shifts.</td><td><eq>\frac{1}{22}\sum_{i=0}^{21}RV_{t-i}</eq></td></tr><tr><td>Daily VIX (VIX)</td><td>The CBOE Volatility Index value for the day, representing the market&#x27;s expectation of near-term (30-day) volatility derived from option prices.</td><td>Sourced directly from VIX Index.</td></tr><tr><td>Weekly VIX (<eq>\overline{VIX}^{(w)}</eq>)</td><td>The average of daily VIX values over the past trading week, smoothing out daily fluctuations to better capture medium-term market sentiment.</td><td><eq>\frac{1}{5}\sum_{i=0}^{4}VIX_{t-i}</eq></td></tr><tr><td>Monthly VIX (<eq>\overline{VIX}^{(m)}</eq>)</td><td>The average of daily VIX values over the past trading month, used to capture longer-term implied volatility trends and investor fear/uncertainty levels.</td><td><eq>\frac{1}{22}\sum_{i=0}^{21}VIX_{t-i}</eq></td></tr><tr><td>Realized Kurtosis (KTS)</td><td>A statistical measure quantifying the “tailedness” or extremity of the return distribution, highlighting the presence of heavy tails or extreme events in returns.</td><td><eq>\frac{N\sum_{i=1}^{N}r_{t,i}^{4}}{\left(\sum_{i=1}^{N}r_{t,i}^{2}\right)^{2}}</eq>*</td></tr><tr><td>Jump Variation (JMP)</td><td>Quantifies discontinuous jumps in price returns separate from continuous volatility, capturing sudden market moves or shocks.</td><td><eq>RV_{t}-BV_{t}</eq> †</td></tr></table>


* $r_{t,i}$ is the $i$ -th intraday return on day $t$ , and $N$ is the total number of intraday returns for day $t$ . 


## B XGBoost Hyperparameters


Table 8: Hyperparameters Used in XGBoost for Distributional and Coefficient-Based Clustering


<table><tr><td>Hyperparameter</td><td>Description</td><td>Search Space</td></tr><tr><td>n_estimators</td><td>Number of trees (boosting rounds) in the ensemble. More trees can improve performance but may increase overfitting and training time.</td><td>{50, 100, 200, 300}</td></tr><tr><td>max_depth</td><td>Maximum depth of each decision tree. Controls model complexity; deeper trees can model more complex relationships but risk overfitting.</td><td>{3, 5, 7, 10}</td></tr><tr><td>learning_rate</td><td>Also called eta; scales the contribution of each tree. Lower values slow down learning for more robust fitting, often requiring more trees.</td><td>np.linspace(0.01, 0.3, 10)</td></tr><tr><td>subsample</td><td>Fraction of training samples randomly selected for each boosting round. Helps prevent overfitting by introducing randomness.</td><td>np.linspace(0.6, 1.0, 5)</td></tr><tr><td>colsample_bytree</td><td>Fraction of features (columns) randomly selected for each tree. Reduces correlation among trees and overfitting risk.</td><td>np.linspace(0.6, 1.0, 5)</td></tr><tr><td>gamma</td><td>Minimum loss reduction required to make a split. Larger values make the algorithm more conservative, pruning trees more aggressively.</td><td>{0, 0.1, 0.2, 0.3}</td></tr><tr><td>reg_alpha</td><td>L1 regularization term on weights (Lasso). Encourages sparsity in feature weights, helping feature selection and reducing overfitting.</td><td>{0, 0.01, 0.1, 1}</td></tr><tr><td>reg_lambda</td><td>L2 regularization term on weights (Ridge). Penalizes large weights, reducing model complexity and overfitting.</td><td>{1, 1.5, 2, 3}</td></tr></table>

## C Libraries Used


Table 9: Python Libraries Used in the Project


<table><tr><td>Library</td><td>Description</td></tr><tr><td>matplotlib</td><td>Used to visualize time series forecasts, model performance comparisons, and residual plots.</td></tr><tr><td>NumPy</td><td>Performs core numerical operations, array manipulation, and matrix mathematics.</td></tr><tr><td>pandas</td><td>Used for time series data manipulation, rolling window computations, and data handling via DataFrames.</td></tr><tr><td>SciPy</td><td>Enables statistical testing, including the Mood test for detecting distributional changes across segments.</td></tr><tr><td>statsmodels</td><td>Implements OLS regression and model diagnostics; used to fit and evaluate standard and regime-specific HAR models.</td></tr><tr><td>scikit-learn</td><td>Used for clustering, feature scaling, and model selection, including cross-validation and hyperparameter tuning.</td></tr><tr><td>xgboost</td><td>Implements gradient boosting framework; used for predicting soft regime probabilities.</td></tr><tr><td>hmmlearn</td><td>Implements Hidden Markov Models, used for inferring latent volatility regimes via Gaussian emissions.</td></tr><tr><td>joblib</td><td>Used to save and load Python objects such as scalers, models, and clustering outputs.</td></tr><tr><td>POT</td><td>Provides tools for optimal transport; used to compute Wasserstein distances between segment distributions.</td></tr></table>