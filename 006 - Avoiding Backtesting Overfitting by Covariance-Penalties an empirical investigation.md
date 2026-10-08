# Avoiding Backtesting Overfitting by Covariance-Penalties: an empirical investigation of the ordinary and total least squares cases

Adriano Koshiyama and Nick Firoozye
Department of Computer Science,
University College London
Gower Street, London WC1E 6BT, United Kingdom
[adriano.koshiyama.15,n.firoozye]@ucl.ac.uk 

May 14, 2019 

## Abstract

Systematic trading strategies are rule-based procedures which choose portfolios and allocate assets. In order to attain certain desired return profiles, quantitative strategists must determine a large array of trading parameters. Backtesting, the attempt to identify the appropriate parameters using historical data available, has been highly criticized due to the abundance of misleading results. Hence, there is an increasing interest in devising procedures for the assessment and comparison of strategies, that is, devising schemes for preventing what is known as backtesting overfitting. So far, many financial researchers have proposed different ways to tackle this problem that can be broadly categorised in three types: Data Snooping, Overestimated Performance, and Cross-Validation Evaluation. In this paper we propose a new approach to dealing with financial overfitting, a Covariance-Penalty Correction, in which a risk metric is lowered given the amount of parameters and data used to underpins a trading strategy. We outlined the foundation and main results behind the Covariance-Penalty correction for trading strategies. After that, we pursue an empirical investigation, comparing its performance with some other approaches in the realm of Covariance-Penalties across more than 1300 assets, using Ordinary and Total Least Squares. Our results suggest that Covariance-Penalties are a suitable procedure to avoid Backtesting Overfitting, and Total Least Squares provides superior performance when compared to Ordinary Least Squares. 

Keywords: Algorithmic Trading, Overfitting, Covariance-Penalty, Total Least Squares, Ordinary Least Squares, Quantitative Finance
MSC Numbers: 60G10, 62E15, 62P05, 62F99, 91G70, 91G80 

## 1 Introduction

Systematic trading strategies are rule-based procedures which choose portfolios and allocate assets, in order to deliver a set of returns which optimize some performance criterion, help to diversify a larger portfolio, or perhaps deliver returns specifically tailored to market conditions, such as providing insurance during rises of market volatility. In order to attain any of these desired return profiles, quantitative strategists must determine a large array of trading parameters, including determining the universe of assets being held, the appropriate trading horizons, the type of strategy being considered and many other possible parametrizations. Normally the analyst tries to identify the appropriate parameters by using all the historical data available, or by holding-out some of the most recent data in order to use it as an out-sample validation set. However, both backtesting strategies suffer from drawbacks that can render their findings misleading. 

Partly due to the abundance of misleading results produced from backtests, and partly due to the endemic abuse of backtested results, there is an increasing interest in devising procedures for the assessment and comparison of strategies $[22, 42, 3]$ . The goal is to devise schemes for preventing what is known as back-testing overfitting. Backtest overfitting can be understood as a collateral effect that occurs during the identification of strategies, resembling other practical modelling problems: overfitting in supervised learning $[14, 11, 29]$ , or data-snooping due to multiple hypothesis testing biases, (also known as p-hacking), a problem occurring frequently in the applied sciences $[25, 6]$ . 

So far, many financial researchers have proposed different ways to tackle this problem that can be broadly categorised in three types: Data Snooping, Overestimated Performance, and Cross-Validation Evaluation. In summary, the Data Snooping and Overestimated Performance techniques aim to identify spurious results by taking into account the number of trials and the performance from all attempted strategies in a multiple hypothesis testing framework. Conversely, Cross-Validation approaches provide more practical and generic mechanisms to individually fine-tune trading strategies. 

In this paper we propose a new approach to dealing with financial overfitting, a Covariance-Penalty Correction, in which a risk metric is lowered given the amount of parameters and data used to underpins a trading strategy. We outline the foundation and main results behind the Covariance-Penalty correction for trading strategies. After that, we pursue an empirical investigation, comparing its performance with some other approaches in the realm of Covariance-Penalties across more than 1300 assets, using Ordinary and Total Least Squares – with the last providing superior performance. Before delving into this novel perspective, we first briefly exhibit a historical context about the available methodologies. 

## 2 Literature Review

We can outline, chronologically, three distinct approaches in the literature to evaluate and deal with backtesting overfitting: Data Snooping, Overestimated Performance, and Cross-Validation Evaluation. The problem of overfitting cannot be understated, and innumerable references highlight the issues with p-hacking which has been an issue for considerable periods but making headlines more recently (see e.g., [35], [26], [7], [13]), and [24]) although it may not always be willful ([16]), and is pernicious in finance (see for instance, [44]). 

## 2.1 Data Snooping

Data snooping is a problem commonly encountered in the realm of econometrics, specifically when a researcher is exploring asset pricing models. Among several definitions in different contexts, in finance it is determined by the act of finding significant yet spurious risk premia factors, due to the unavoidable reuse of the same time series $[8, 48]$ . The first work in this direction can be attributed to Lo and MacKinlay $[33]$ attempted to highlight and quantified the effect of data snooping in risk premia studies. 

More precisely, the data-snooping problems/solutions tend to focus on the issue of Multiple Hypothesis Testing. Standard methods to adjust for snooping from multiple hypothesis testing biases, such as by Bonferroni and Holm to control family wise error rates (FWER $^{1}$ ) and by Benjamini-Hochberg-Yekutieli (BHY) for controlling False Discovery Rates (FDR $^{2}$ ) are in relatively common use in applied sciences [25, 6]. These methods adjust for the likelihood of false discoveries due to the correlation between multiple tests and are an effective way to prevent what is otherwise known as p-hacking in the applied sciences. Nonetheless, since they adjust to effectively worst-case scenarios, they are known to result in significant reductions of power. 

Due to the low power of the FWER adjustments, Westfall and Young $[47]$ introduced bootstrap methods to estimate the empirical correlation between hypothesis tests as a way of optimising the power of the resulting adjusted test statistics. Following on these, White $[48]$ outlined one of the first attempts to improve test statistics by applying bootstrap methods to identify spurious relationships in the realm of trading strategies, applying them to technical trading rules $[45]$ . Romano and Wolf $[40]$ advanced in procedures to more adequately correct multiple testing and resourcing in more adequate forms of bootstrapping for standard errors calculations; we point also to more recent works in this direction $[41, 39, 42, 23]$ . 

Since Romano and Wolf can be seen as an extension of White's work, we briefly outline its algorithm for data snooping diagnosis. Let $r_{it} = R_{it} - R_{bench,t}$ represent the excess return $(R)$ of the asset $i$ in relation to a benchmark bench at time $t$ . Each asset can represent a trading strategy with different parametrisation, and the benchmark can be the usual 3-month London Interbank Overnight Rate return or simply an offset. Given a differentiable performance function $P$ , a significance level $\alpha$ , and a number of bootstrap samples $B$ ( $b = 1, ..., B$ ), do: 

- Compute each asset performance: $p_i = P(r_{i1}, \dots, r_{it}, \dots, r_{iT})$ . 

- Set $\hat{V} = \sqrt{T} * \max(p_1, ..., p_n)$ as the maximum estimated performance across the available assets. 

- Set $k = 1$ , and while all assets have not been rejected: 

1. For $b = 1, \dots, B$ bootstrap samples, do: 

(a) Take a bootstrap sample $r_{.1}^{(b)}, \ldots, r_{.T}^{(b)}$ from the sequence $r_{.1}, \ldots, r_{.T}$ . The . notation is the abbreviation for "all assets". 

(b) Compute each asset performance: $p_{i}^{(b)} = P(r_{i1}^{(b)}, \ldots, r_{it}^{(b)}, \ldots, r_{iT}^{(b)})$ 

(c) Define the statistic $V_{b} = \sqrt{T} * \max(p_{i}^{(b)} - p_{1}, ..., p_{n}^{(b)} - p_{n})$ 

2. Find the critical value $c_k$ by estimating the quantile $(1 - \alpha / k)\%$ of $V_b$ empirical distribution. 

3. Reject all strategies that $(\sqrt{T} * p_{i}) > c_{k}$ ; remove these and place them in the echelon k. 

$$
4. k + = 1
$$

• Report all echelons to the user. 

Each echelon represent a subset of trading strategy that tend to perform similarly, but are superior/inferior to the followed/previous iteration. As it can be seen, this method is based on a stepdown approach, since we start with the null hypothesis that correspond to the largest test statistic. A final comment: for the first iteration it is possible to calculate the p-value for the Bootstrap Reality Check [48] by: $p_{BRC} = \frac{\sum_{b=1}^{B} V_{b} > \hat{V}}{B}$ . This statistic addresses the question whether the strategy that appears best in the observed data really beats the benchmark; it does not attempt to identify as many outperforming strategies as possible. We also highlight that this presentation took the non-studentised version of Romano and Wolf procedure; for the studentised version, which provide some improvements to the statistical procedure, refer to [40]. 

Finally, another related approached called Model Confidence Set $[21, 20]$ , acknowledges the limitations of the data, such that uninformative data yield a MCS with many models, whereas informative data yield a MCS with only a few models. By pairwise comparisons, this methodology creates confidence sets (analogous to confidence intervals) that contains the best model or models given a specified significance level. 

## 2.2 Overestimated Performance

In Harvey and Liu $[22]$ the authors exhibit a multiple hypothesis testing framework devoted to identify strategies incorrectly classified as profitable. Before delving into the $[22]$ work, it should be mentioned that similar works have also proposed ways to incorporate uncertainty and perform adjustments in the strategies performance (mainly for Sharpe Ratios). A section of these tends to favour hypothesis tests and confidence interval computations, starting from: (i) the classical i.i.d. Gaussian returns $[28]$ ; (ii) then dropping Gaussian assumption with the support of (independent) bootstrap $[46]$ ; (iii) more ahead being able to pinpoint an asymptotic distribution for the Sharpe Ratio under non-i.i.d and more mild assumptions $[32, 37]$ ; and finally (iv) dropping i.i.d. and underlying distribution assumptions by applying improved forms of bootstrapping $[30]$ . Another approach is by adjusting the Sharpe Ratio by computing the so-called Probabilistic Sharpe Ratio $[4]$ . This measure computes the probability that a strategy performance is greater than a benchmark, taking into account the skewness and kurtosis of the returns. 

However, all these works do not take into account the number of trials did to fully develop the strategy, or when a hypothesis test is being made to compare different strategies. This information is relevant, because some improvement is expected whenever the analyst have access to information coming from the backtest: increasing overall performance, removing drawdowns, etc. In this way, he can design stop/loss rules, change leveraging rules, with undoubtedly impact in the strategy performance when the backtest is remade. The authors $[22]$ treat this as a Multiple Hypothesis Testing situation $[25, 6]$ . To illustrate this concept, if we suppose that the (excess) returns are i.i.d. and follows a Normal distribution, then the Sharpe Ratio can be reinterpreted as t-scores: 

$$
t _ {s c o r e} = \frac {\bar {R}}{\sigma_ {R}} \times \sqrt {T} = \hat {S R} \times \sqrt {T}\tag{1}
$$

where $\hat{SR}$ is the estimated Sharpe Ratio and T is the backtest length. We can compute a t-test using this $t_{score}$ , aiming to check the probability for observing a departure from the null hypothesis ( $H_{0}: SR = 0$ ) as extreme as the difference present in our sample data. This is defined as: 

$$
p ^ {S} = P r (| t | > t _ {s c o r e})\tag{2}
$$

in which t denotes a random variable that follows a t-distribution with T-1 degrees of freedom, and $p^{S}$ is commonly dubbed as p-value, which in this case is computed for a single hypothesis (S). If a research have explored several strategies, then some overstatement will appear in the $\hat{S}\hat{R}$ , and we could interpret this as the researcher did tacitly multiple attempts to obtain such result. As a initial estimation of this overestatement, suppose that all attempts are independent (unreal, but we drop this more ahead), then we can rewrite 2 as: 

$$
\begin{array}{l} p ^ {M} = P r (m a x (| t _ {i} |, i = 1,..., n) > t _ {s c o r e}) = \\ = 1 - \prod_ {i = 1} ^ {n} P r (| t _ {i} | \leq t _ {s c o r e}) = 1 - (1 - p ^ {S}) ^ {n} \end{array}\tag{3}
$$

by following [22]: when n = 1 (single test) and $p^{S} = 0.05$ , $p^{M} = 0.05$ , so there is no multiple testing adjustment. If n = 10 and we observe a strategy with $p^{S} = 0.05$ , $p^{M} = 0.401$ , implying a probability of about 40% in finding an investment strategy that generates a SR that is at least as large as the observed $\hat{SR}$ , but much larger than the 5% probability for a single test. It is clear that multiple testing greatly reduces the statistical significance of single test. Hence, $p^{M}$ is the adjusted p-value, that reflects the likelihood of finding a strategy that is at least as profitable as the observed strategy after n attempts. By equating the p-value of a single test to $p^{M}$ , we obtain the defining equation for the multiple testing adjusted (haircut) Sharpe ratio $H\hat{SR}$ : 

$$
p ^ {M} = P r (| t | > H \hat {S} R \times \sqrt {T})\tag{4}
$$

since $p^{M}$ is larger than $p^{S}$ , $H\hat{S}R$ will be smaller than $\hat{S}R$ . However, when the test statistics are dependent, however, the approach in the example is no longer applicable as $p^{M}$ generally depends on the joint distribution of the n test statistics. For this more realistic case, the authors apply techniques that controls the FWER and FDR. These methods are Bonferroni and Holm p-value corrections to control FWER [10] and BHY procedure for FDR [6]. Below, in Figure 1 we display some charts with pre and post-adjustment of Sharpe Ratios, as well as the Haircuts $^{3}$ : 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/4015fdde099673dee326cfca1fabbb4a05e8d13d186839d26fb390b6bbbbc04b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/be902204190d29fb48f2418c083effe4201bb7db6fcf84943cafc2d0771421a4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/1361c4a20b430b6469fe58f3cadc6d6590cc09d98efb72f4dcfce9fe372536b9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/c2e5b66125a01e00a4c78e25dd44cc6900f806a2173594eb78255cac239aa949.jpg)



Figure 1: Original and Haircut Sharpe Ratios, as well as the Haircut levels for different methodologies and number of trials. Source: [22]


As it can be seen, the Haircut is a non-linear function and is highly affected by the number of trials, and the method of adjustment. The authors advocate by the BHY method in financial applications, as they argue that it is more reasonable to control for the rate of false discoveries, rather than controlling for not making any false discovery. Finally, as the authors points out, the main caveat of this method is that all the analysis is based with in-sample data, and as they observe a hybrid strategy might be useful as a double validation of the selected trading strategy. 

In line with multiple testing literature, there remain challenges associated with knowing which analyst observations of data are, in fact, uses of the data and thus, tests. Other challenges of multiple testing (in particular, when using brute-force penalisations such as BHY, Holm and Bonferroni) include: 

- What is the impact of common knowledge? 

- How does prior information impact hypotheses being tested? (and strategies being designed)? 

- If multiple analysts look at the same data, should they be penalized based on each other observations? 

- If the tests were conducted in a different order, the results would be different, so how does this impact the philosophy of the approach? 

- How do we know if we have been influenced by others who have seen the same data? 

- Is it even possible to have no knowledge of relevant financial history? 

While care and concern over the overuse of history is imperative, it is not altogether clear whether the philosophy of multiple testing, especially as done via BHY, Holm and Bonferroni, is reasonable for financial applications. Nonetheless, we believe the methodology is absolutely informative and can be used in conjunction with our own. As yet an unresolved area, preventing overfitting requires a multiplicity of approaches. 

## 2.3 Cross-Validation Evaluation

In Bailey et al. [3] the authors provide a set of definitions, tools and benchmarks to characterize the so-called backtest overfitting. In certain sense this work extend previous research and findings from the same group of authors, such as the probability that a Sharpe Ratio is inflated [4] and to determine the minimum track record for a Sharpe Ratio to be statistically different from a given threshold [2]. 

In terms of definitions, the authors establish that a strategy selection overfits if the expected performance in-sample is less than the median performance rank out-of-sample of all strategies. By strategy selection, it means a way of ranking the available strategies using a set of performance metrics (e.g., Sharpe Ratio). 

In relation to tools they propose the procedure called Combinatorially-Symmetric Cross-Validation (CSCV), which is briefly outlined below and represented in Figure 2. 

<table><tr><td colspan="2">In-Sample</td><td colspan="2">Out-of-Sample</td></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>A</td><td>C</td><td>B</td><td>D</td></tr><tr><td>A</td><td>D</td><td>B</td><td>C</td></tr><tr><td>B</td><td>C</td><td>A</td><td>D</td></tr><tr><td>B</td><td>D</td><td>A</td><td>C</td></tr><tr><td>C</td><td>D</td><td>A</td><td>B</td></tr></table>


Figure 2: Generating the CSCV folders, when S = 4. Each letter represents a block, and note that whenever the shuffles happen, it is necessary to reorder the blocks in each in-sample and out-of-sample sets.


1. Split the profit and loss of several strategies in S even size blocks over the backtested horizon (where S is a even natural number); 

2. Reshuffle these blocks in all feasible arrangements such that the natural ordering is as most preserved as possible; and 

3. For each first half (in-sample) of the arrangement, find the optimal strategy for a given performance measure and compare it to its own ranking in the out-of-sample set (second half). 

With these rankings and comparisons, the authors compute four complementary metrics: 

1. Probability of Backtest Overfitting: the probability that the model configuration selected as optimal in-sample will underperform the median of the N models configurations out-of-sample. 

2. Performance degradation: this determines to what extent greater performance in-sample leads to lower performance out-of-sample (discussed in [2]). 

3. Probability of loss: the probability that the model selected as optimal IS will deliver a loss OOS. 

4. Stochastic dominance: this analysis determines whether the procedure used to select a strategy IS is preferable to randomly choosing one model configuration among the N alternatives. 

Also, in terms of benchmarks the authors propose some synthetic and practical test cases. In this sense, the objective of these studies is to assess if the proposed framework is able to dodge from overfitted strategies with different set of hyperparameters (number of strategies, backtest length and overfit degree). 

We can highlight two main critics assessment towards this work: 

1. Preservation of temporal dependence: the authors claim that their method preserves the temporal dependence. In their support are the fact that their method reorders the blocks in-sample and out-of-sample and someone might argue that daily returns tends to present weak autocorrelation. However, this claim about preservation of time dependence is clearly not absolutely true: just check the third to fifth row at Figure 2. 

2. High overlap degree in the in-sample and out-of-sample metrics: it can be easily noticed that there is a high level of juxtaposition of blocks appearing several times in the out-of-sample part (and similarly in the in-sample). The authors in proposing the CSCV, using as inspiration the k-fold cross-validation, forgot to analyse deeper the impact of the highly correlated out-of-sample folders (something that do not occur in the case for the k-fold cross-validation). This correlation exerts an impact in the computation of standard errors and hypothesis tests $[29, 5]$ , and therefore decrease the out-of-sample assessment quality. 

## 2.4 Summary

Table 1 present a categorisation of all the previous lines of investigation – with the addition of Covariance-Penalty –, in terms of their breadth/generality of use in practice and the type of solution used to tackle backtesting overfitting. 


Table 1: Categorisation of all the four lines of investigations in backtesting overfitting.


<table><tr><td>Breadth / Solution</td><td>Multiple Hypothesis</td><td>In-Out Sample Assessment</td></tr><tr><td>Generic</td><td>Data Snooping</td><td>Cross-Validation</td></tr><tr><td>Assumptions</td><td>Overestimated Performance</td><td>Covariance-Penalty</td></tr></table>

The Data Snooping and Overestimated Performance authors identify that the root cause of any overfitting or spurious results is due to the multiple trials (hypothesis) employed by the analyst during the strategies fine-tuning step. However, the Data Snooping solutions tend to be more generic – it is based on bootstrapping returns –, with the expense of a increased computational burden. The Overestimated Performance solutions are based on a set of assumptions, allowing the authors to find closed-form solutions or simpler algorithms that make the performance correction aspect easier to compute. 

In contrast, the Cross-Validation Evaluation and Covariance-Penalty Correction are focused on avoiding the apparent in-sample performance, by uncovering the expected generalisation (out-sample) potential of a strategy. Similarly, they divert in terms of their amplitude: the Covariance-Penalty Correction methods require some assumptions on the joint behaviour of the asset and trading strategy, which is not the case for the majority of the Cross-Validation procedures. However, this lack of assumptions demand an extra computational processing in respect to the Covariance-Penalty approach. 

## 3 Covariance-Penalty for Trading Strategies

This section outlines the theoretical results and notations used along this paper. We start by enunciating single period linear strategies, and some of its properties. Then we move to optimisation of trading strategies by maximising correlation, and its links to Sharpe ratio utility function. This result/link allows us to conclude this section with the main result of this paper: Covariance-Penalty correction of Sharpe ratio. 

## 3.1 Single period linear strategies

We consider the (log) returns of a single asset, $R_{t} \sim \mathcal{N}(0, \sigma_{R}^{2})$ returns with autocovariance function at lag k, $\gamma(k) = E[R_{t} R_{t-k}]$ , together with corresponding auto-correlation function (ACF), $c(k) = \gamma(k)/\gamma(0)$ at lag k. 

Our main aim is to work with strategies based on linear portfolio weights (or signals) $X_{t} = \Sigma_{1}^{\infty} a_{k} R_{t-k}$ for coefficients $a_{k}$ generating the corresponding dynamic strategy returns $S_{t} = X_{t} \cdot R_{t}$ (here, and always, the signal, $X_{t}$ is assumed to only have appropriately lagged information). Example strategy weights include exponentially weighted moving averages $a_{k} \propto \lambda^{k}$ , simple moving averages $a_{k} = \frac{1}{T} 1_{[1,\ldots,T]}$ , forecasts from ARMA models, etc. Most importantly, the portfolio weights X are normal and jointly normal with returns R. 

We restrict our attention to return distributions over a single period. In the case of many momentum strategies, this period can be one day, if not longer. For higher-frequency intra-day strategies, this period can be much shorter. The pertinent concern is that the horizon (i.e., one period) is the same horizon over which the rebalancing of strategy weights is done. If weights are rebalanced every five minutes, then the single period should be five minutes. This is a necessary assumption in order to ensure the joint normality of (as yet indeterminate) signals and future returns. Moreover, this assumption will give some context to our results, which imply a maximal Sharpe ratio, maximal skewness and maximal kurtosis for dynamic linear strategies. 

We are interested in characterizing the moments of the strategy's unconditional returns, the corresponding standard errors on estimated quantities, and means of optimising various non-dimensional measures of returns such as the Sharpe ratio via the use of non-linear transformations of signals. Our goal is to look at unconditional properties of the strategy. It is important to avoid foresight in strategy design and this directly impacts the conditional properties of strategies (e.g., conditional densities involve conditioning on the currently observed signal to determine properties of the returns, which are just Gaussian). In the context of our study, we are concerned with one-period ahead returns of the unconditional returns distribution of our strategy, where both the signals and the returns are unobserved, and the resulting distributions (in our case, the product of two normals) are much richer and more realistic. 

## 3.2 Properties of linear strategies

Given the joint normality of the signal and the returns, we can explicitly characterise the one-period strategy returns (see $[9]$ ). To allow for greater extendibility, we prefer to only consider the moments of the resulting distributions. These can be characterized easily using Isserlis' theorem $[27]$ , which gives all moments for any multivariate normal random variable in terms of the mean and variance. We also refer to $[19]$ who meticulously produces both non-central and central moments for powers and products of Gaussians. While this is a routine application of Isserlis' theorem, the algebra can be tedious, so we quote the results. 

Theorem 1 (Isserlis (1918) [27]). If $X \sim \mathcal{N}(0, \Sigma)$ , then 

$$
E [ X _ {1} X _ {2} \dots X _ {2 n} ] = \sum_ {i = 1} ^ {2 n} \prod_ {i \neq j} E [ X _ {i} X _ {j} ]
$$

and 

$$
E [ X _ {1} X _ {2} \dots X _ {2 n - 1} ] = 0
$$

where the $\sum \prod$ is over all the $(2n)! / (2^n n!)$ unique partitions of $X_{1}, X_{2}, \ldots, X_{2n}$ into pairs $X_{i}X_{j}$ . 

Haldane's paper quotes a large number of moment-based results for various powers of each normal. We quote the relevant results. 

Theorem 2 (Haldane (1942) [19]). If $x, y \sim \mathcal{N}(0,1)$ with correlation $\rho$ then 

$$
\begin{array}{r l r} {E [ x y ] =} & {\rho} \\ {E [ x ^ {2} y ^ {2} ] =} & {1 + 2 \rho^ {2}} \\ {E [ x ^ {3} y ^ {3} ] =} & {3 \rho (3 + 2 \rho^ {2})} \\ {E [ x ^ {4} y ^ {4} ] =} & {3 (3 + 2 4 \rho^ {2} + 8 \rho^ {4})} \end{array}
$$

and thus the central moments of xy are 

$$
\begin{array}{r l r} \mu_ {1} = & & \rho \\ \mu_ {2} = & & 1 + \rho^ {2} \\ \mu_ {3} = & & 2 \rho (3 + \rho^ {2}) \\ \mu_ {4} = & & 3 (3 + 1 4 \rho^ {2} + 3 \rho^ {4}) \end{array}
$$

From these one period moments, (and a simple scaling argument giving the dependence on $\sigma(x)$ and $\sigma(y)$ ) we can characterise Sharpe ratio, skewness, etc., and can also define objective functions in order to determine some sense of optimality for a given strategy. 

Theorem 3 (Linear Gaussian). For single asset returns and a one period strategy, $R_{t} \sim \mathcal{N}(0, \sigma_{R}^{2})$ and $X_{t} \sim \mathcal{N}(0, \sigma_{X}^{2})$ jointly normal with correlation $\rho$ , the Sharpe ratio is given by 

$$
S R = \frac {\rho}{\sqrt {1 + \rho^ {2}}}\tag{5}
$$

the skewness is given as 

$$
\gamma_ {3} = \frac {2 \rho (3 + \rho^ {2})}{(1 + \rho^ {2}) ^ {\frac {3}{2}}},\tag{6}
$$

and the kurtosis is given by 

$$
\gamma_ {4} = \frac {3 (3 + 1 4 \rho^ {2} + 3 \rho^ {4})}{(1 + \rho^ {2}) ^ {2}}\tag{7}
$$

In the appendix, we extend equations (5) and (6) to the case of non-zero means. 

Proof. A simple application of Theorem 2 give us the following first two moments for our strategy $S_{t} = X_{t}\cdot R_{t}$ : $\mu_{1} = E[S_{t}] = E[X\cdot R] = \sigma_{X}\sigma_{R}\rho$ . and $\mu_{2} = Var[S_{t}] = \sigma_{X}^{2}\sigma_{R}^{2}(\rho^{2} + 1)$ . Thus we can derive the following results for the Sharpe ratio, 

$$
S R = \frac {\mu_ {1}}{\mu_ {2} ^ {1 / 2}} = \frac {\sigma_ {X} \sigma_ {R} \rho}{\sigma_ {X} \sigma_ {R} \sqrt {\rho^ {2} + 1}} = \frac {\rho}{\sqrt {\rho^ {2} + 1}}\tag{8}
$$

Moreover, we can see that the skewness, 

$$
\gamma_ {3} = \frac {\mu_ {3}}{\mu_ {2} ^ {3 / 2}} = \frac {2 \rho (3 + \rho^ {2})}{(1 + \rho^ {2}) ^ {3 / 2}}\tag{9}
$$

Finally, the kurtosis is given by 

$$
\gamma_ {4} = \frac {\mu_ {4}}{\mu_ {2} ^ {2}} = \frac {3 (3 + 1 4 \rho^ {2} + 3 \rho^ {4})}{(1 + \rho^ {2}) ^ {2}}\tag{10}
$$

□ 

If we restrict our attention to positive correlations, all three dimensionless statistics are monotonically increasing in $\rho$ . Consequently, strategies that maximize one of these statistics will maximize the others, although the impact of correlation upon Sharpe ratio, skewness and kurtosis is different. We illustrate the cross-dependencies in the following charts, depicting the relationships between the variables. In figure 3, the shaded blue histograms correspond to correlation ranges ( $\{[-1,-0.5],[-0.5,0],[0,0.5],[0.5,1]\}$ ). We note that a uniform distribution in correlations maps into a higher likelihood of extreme Sharpe ratios and an even higher likelihood of extreme skewness and kurtosis. 

Skewness ranges in $[-2^{3/2}, 2^{3/2}] \approx [-2.8, 2.8]$ . Unlike the Sharpe ratio, Skewness' dependence on correlation tends to flatten, so to achieve 90% peak skewness, one needs only achieve a 0.60 correlation, while for a 90% peak Sharpe, one needs a correlation of 0.85. Kurtosis is an even function and varies from a minimal value of 9 to a maximum of 15. In practice, correlations will largely be close to zero and the resulting skewness and kurtosis significantly smaller than the maximal values. 

Although we analyse the moments of the strategy $S_{t} = X_{t}R_{t}$ , the full product density is actually known in closed form (see appendix A, [9] and [36]). It is clear that the distribution of the strategy is leptokurtic even when it is not predictive (when the correlation is exactly zero, the strategy has a kurtosis of 9). In the limit as $\rho \rightarrow 1$ , the strategy's density approaches that of a non-central $\chi^2$ , an effective best-case density when considering the design of optimal linear dynamic strategies. 

An optimised strategy with sufficient lags (and a means of ensuring parsimony) may be able to capture both mean-reversion and trend and result in yet higher correlations. Annualised Sharpe ratios of between 0.5-1.5 are most common (i.e., correlations of between 3% to 9%) for single asset strategies in this relatively low-frequency regime. 

## 3.3 Optimisation: Maximal Correlation, Total least squares

Many algorithmic traders will explain how problematic strategy optimisation is, given the endless concerns of over-fitting, etc. Although these are a concern, the naïve use of strategies which are merely pulled out of thin air is equally problematic, where there is no explicit use of optimisation (and, in its place more eye-balling strategies or targeting Sharpe ratios rather loosely, effectively a somewhat loose mental optimisation exercise). Practical considerations abound and real-world returns are neither Gaussian nor stationary. We argue irrespectively that using optimisation and a well-specified utility function as a starting point is a means of preventing strategies from being just untested heuristics. Unlike most discretionary traders' heuristics (or rules of thumb) which have their place as a means of dealing with uncertainty (see for example [17]), heuristic quantitative trading strategies run the risk of being entirely arbitrary, or are subject to a large number of human biases, in marked contrast to the monniker quantitative investment strategies. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/b2bfb115f1a58cbcc6cf4fa1271a6db368cf0398720ab38bc1fb7c3e7cb101c7.jpg)



Figure 3: Correlation, Sharpe ratio, Skewness, and Kurtosis pairwise relationship. A uniform distribution in correlation is bucketed into four ranges $\{[-1,-0.5],[-0.5,0],[0,0.5],[0.5,1]\}$ as depicted in the bar charts in shades of blue. After transforming the correlation into SR, $\gamma_{3}$ and $\gamma_{4}$ the frequencies are no longer uniform.


Where optimisation is used, the most common optimisation method is to minimize the mean-squared error (MSE) of the forecast. Our results show that rather than to minimize the $L^{2}$ norm between our signal and the forecast returns (or to maximize the likelihood), if the objective is to maximize the Sharpe ratio, we must maximize the correlation. 

We can see in figures 4 and 5, a depiction of fits of strategies applied to S&P 500 using EWMA and HW filters for a variety of parameters. The relationship between MSE and Sharpe ratio is not monotone in MSE for the EWMA filter as we see in figure 4, while it is much closer to being linear in the case of the relationship between correlation and Sharpe. For the case of HW (with two parameters), in figure 5 any given MSE can lead to a non-unique Sharpe ratio, sometimes with a very broad range, leading us to conclude that the optimization is poorly posed. The relationship of correlation to Sharpe is obviously closer to being linear, with higher correlations almost always leading to higher Sharpe ratios. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/d791f09a61604b5e00e89f4fceeedfeb991806ac7f319331387f7a7c4d261876.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/c419819b05a4029893f05b4d08f2e3ae9ab1d48c12254fc22ac09ee36c3671ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/4754df665497f9ef6a0a884bec6fe55a4e6f77a054623363d8678097084f1f4f.jpg)



Figure 4: EWMA Strategy Sharpe Ratio vs $\alpha$ , MSE and correlation for S&P 500 reversal strategies


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/cfb3e71fa66e918a3730cb29731a85268af0f037f672e800c937a8fe082676d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/db88ffd86fc1bf80827e539f9ad7880400e2bde41ec6d3624bf1ee938aa197b5.jpg)



Figure 5: Holt-Winters Strategy Sharpe Ratio vs MSE and correlation for S&P 500 Reversal Strategies


In the case of a one-dimensional forecasting problem with (unconstrained) linear signals, optimizing the correlation amounts to using what is known as total least squares regression (TLS) or orthogonal distance regression, a form of principal components regression (see, e.g., [18] and [34]). In the multivariate case, it would be more closely related to canonical correlation analysis (CCA). 

Unlike OLS, where the dependent variable is assumed to be measured with error and the independent variables are assumed to be measured without error, in total least squares regression, both dependent and independent variables are assumed to be measured with error, and the objective function compensates for this by minimizing the sum squared of orthogonal distances to the fitted hyperplane. This is a simple form of errors-in-variables (EIV) regression and has been studied since the late 1870s, and is most closely related to principal components analysis. For k regressors, the TLS fit will produce weights which are orthogonal to the first k - 1 principal components. 

So, if we consider the signal $X = Z\beta$ to be a linear combination of features, 

with $Z \in R^{k}$ a k-dimensional feature space, then we note that 

$$
\hat {\beta} ^ {O L S} = (Z ^ {\prime} Z) ^ {- 1} Z ^ {\prime} R
$$

but 

$$
\hat {\beta} ^ {T L S} = (Z ^ {\prime} Z - \sigma_ {k + 1} ^ {2} I) ^ {- 1} Z ^ {\prime} R
$$

where $\sigma_{k + 1}$ is the smallest singular value for the $T\times (k + 1)$ dimensional matrix $\tilde{X} = [R,Z]$ (i.e., the concatenation of the features and the returns, see, e.g., [38]4).It is well known that, for the case of OLS, the smooth or hat matrix $\hat{R} = MR$ is given by 

$$
M ^ {O L S} = Z (Z ^ {\prime} Z) ^ {- 1} Z ^ {\prime}
$$

with $Tr(M^{OLS}) = k$ , the number of features. In contrast, 

$$
M ^ {T L S} = Z (Z ^ {\prime} Z - \sigma_ {k + 1} ^ {2} I) ^ {- 1} Z ^ {\prime}
$$

and effectively has a greater number of degrees of freedom than that of OLS, i.e., 

$$
T r (M ^ {T L S}) \geq T r (M ^ {O L S})
$$

with equality only when there is complete collinearity $^{5}$ For this reason, many people see TLS as an anti-regularisation method and may result in less-stable response to outliers (see for example, [49]). Consequently, there is extensive study of regularised TLS, typically using a weighted ridge-regression (or Tikhonov) penalty (see discussion in [49] for more detail on this large body of research). 

While maximizing correlation rather than minimizing the MSE seems a very minor change in objective function, the formulas differ from those of standard OLS. The end result is a linear fit which takes into account the errors in the underlying conditioning information. We believe that it should be of relatively little consequence when the features are appropriately normalized, as is the case for univariate time-series estimation, although some authors have suggested that optimising TLS is not appropriate for prediction (see, e.g., [15] section 1.6.3). When we seek to maximize the Sharpe ratio of a strategy, the objective should not be prediction, but rather optimal weight choice. 

## 3.4 Covariance-Penalty Correction

In order to partly address the issue of overfitting of dynamic models, we look to estimated objective functions, taking degrees of freedom into account. Most commonly used estimates include Akaike and Bayes' information criteria, but these require the use of a likelihood. In our case, we do not have an exact likelihood, but instead choose to optimise the Sharpe ratio, and noting the monotonicity in $\rho$ , this comes from maximizing $\hat{\rho}$ . Consequently, we wish to consider the means of optimising correlation taking into account penalisations for degrees of freedom. This type of penalised optimum can be found via Mallows' $C_{p}$ or via Cross validation. 

We mirror the discussion in [12] on Covariance penalties, Section 12.3, which unifies the discussion of Mallow's $C_p$ , Stein's Unbiased Risk Estimate (SURE), Akaike (AIC), and Bayesian Information Criteria (BIC). Each of these are methods for adjusting estimated in-sample goodness of fits measures, to produce better estimates of the out-of-sample performance. Effectively, they are a means of adjusting for the degrees of freedom of various models, akin to adjusted $R^2$ measures. 

We are particularly interested in the biases which occur due to in-sample estimates of measures which we optimize. Standard errors will seem lower in-sample, due to the fact that we have simultaneously optimized correlation, and measured this optimum to produce an estimated Sharpe ratio. We would like to have a more sober estimate of the potential performance of our optimised strategies out of sample. 

In our study, we do not fit probabilistic models or data-generating processes, but rather optimize the strategy. As a consequence, we do not have explicit MLE estimators (or Bayesian posterior), and no likelihood is explicitly used, so we cannot reproduce results for the AIC and the BIC. We are, however, able to find estimates for optimal $\hat{\rho}_{in}$ and the resulting out-of-sample adjustments for this $\hat{\rho}_{in}$ which give us a more reasonable expectation of the real $E[\rho_{out}]$ out of sample $^{5}$ . 

Maximizing $\hat{\rho}_{in}$ is of course similar to performing a normalized OLS, and our interest would be in the analogue of adjustments to MSE. The first is easily found in Mallow's $C_p$ . Mallow's formula states that 

$$
\begin{array}{r} E [ E r r o r _ {o u t} ^ {2} ] = E [ | R - X b | _ {o u t} ^ {2} ] \\ = \frac {1}{T} \sum | R _ {t} - X _ {t} b | _ {i n} ^ {2} - C o v (\mu_ {t}, \hat {\mu}) _ {t} \\ = \widehat {s s e} ^ {2} - \frac {2 \sigma_ {R} ^ {2}}{N} T r (M) \end{array}
$$

where we have confined our attention to linear predictive models, i.e., $\hat{m}u = My$ where M is the hat matrix or smooth matrix. 

We recognise that $sse = (1 - \rho^2)\sigma_R^2$ from which we can easily derive the following: 

Theorem 4 (Mallow's $C_p$ ). 

$$
E [ \rho_ {o u t} ^ {2} ] = \hat {\rho} _ {i n} ^ {2} - \frac {2}{N} T r (M)
$$

where 

$$
R = M X
$$

with hat-matrix or smooth M 

We do not prove this since it is a trivial extension of Mallow's. 

If $\hat{\rho}$ is estimated as a by-product of OLS, then $M = X(X'X)^{-1}X$ (see [12], (§12.5)) and $Tr(M) = n$ , where $k$ is the dimension of the column space of $X$ or the degrees of freedom for regressions (i.e. the total number of lags, where we assume that we have no problems with collinearity). The regularization term turns out to be $\frac{2 \cdot k}{N}$ if $\hat{\rho}$ is estimated in-sample by OLS. 

If, on the other hand, $\hat{\rho}$ is estimated to optimize the in-sample Sharpe ratio, i.e., through TLS, then it is well known that 

$$
M = X (X ^ {\prime} X - \sigma_ {n + 1} ^ {2} I) ^ {- 1} X ^ {\prime}
$$

where $\sigma_{n+1}$ is the smallest singular value of the matrix $T \times (n+1)$ dimensional matrix: $\tilde{X} = [R, X]$ ([38]). As is well known (see for example, [49]) TLS is in fact an anti-regularisation method and may result in less-stable response to outliers. 

In this case, it is also known that $Tr(M) = Tr(Z)$ where $Z = (X'X - \sigma_{n+1}^2 I)^{-1}X'X$ and we know that the singular values of $\sigma(Z) = \left\{\frac{\lambda_i^2}{(\lambda_i^2 - \sigma_{n+1}^2)}\right\}$ where $\lambda_i$ are the singular values of X (or correspondingly, $\lambda_i^2$ are the singular values of $X'X$ , $\lambda_1 \geq \cdots \geq \lambda_n > 0$ ([31]). 

By the Wilkinson interlacing theorem, $\lambda_{n} \geq \sigma_{n+1} \geq 0$ (see [38]). Consequently, 

$$
t r (M) = \sum_ {i} \frac {\lambda_ {i} ^ {2}}{(\lambda_ {i} ^ {2} - \sigma_ {n + 1} ^ {2})} > n
$$

with equality iff $\sigma_{n+1}^2 = 0$ (i.e., when there the $R^2 = 100\%$ and consequently, OLS and TLS coincide). 

Thus, while TLS can optimize the in-sample Sharpe-ratios over other predictive methods including OLS, the corresponding covariance penalization is larger than that of OLS and may result in an expected degradation of performance out-of-sample. $^{4}$ 

The corresponding penalization term can also be applied directly (via first-order conditions or the so-called Delta-method) to removing estimation bias from Sharpe ratio estimates, i.e., 

$$
{E [ S R _ {o u t} ]} \approx {\widehat {S R} _ {i n} - \frac {2 \cdot T r (M)}{N} \frac {\partial S R _ {i n}}{\partial \rho}}\tag{11}
$$

$$
\approx \widehat {S R} _ {i n} - \frac {2 \cdot T r (M)}{N} \frac {1}{(\hat {\rho} ^ {2} + 1) ^ {3 / 2}}\tag{12}
$$

$$
\approx \widehat {S R} _ {i n} - \frac {2 \cdot T r (M)}{N} (1 - \widehat {S R} _ {i n} ^ {2}) ^ {3 / 2}\tag{13}
$$

for all $|\widehat{SR}_{in}| < \frac{\sqrt{2}}{2}$ . 

We also can extend Stein's Unbiased Regression Estimates (SURE), i.e. 

Theorem 5. $SURE$ 

$$
E [ \rho_ {o u t} ^ {2} ] = \hat {\rho} _ {i n} ^ {2} - \frac {2}{N} \sum_ {t = 1} ^ {N} \frac {\partial \hat {X} _ {t}}{\partial R _ {t}}
$$

In both cases, the degrees of freedom are used to adjust down the estimated Sharpe ratios. Consequently, more highly parametrised models will generally have higher overoptimism. The penalisation terms help to adjust for this overoptimism. 

## 4 Empirical Assessment

In our empirical investigation, the quantitative strategist is required to build a trading strategy based on lagged information of a certain asset returns. Since the amount of lagged information is not given, he needs to discover it from the given in-sample data. The most naive way (called Naive method in our experiments) is to try different lags and to pick the one that maximizes the Sharpe ratio. Another traditional approach is to select the amount of lags using an information criteria, such as the AIC (Akaike Information Criteria $[1]$ ) for the OLS case. Nonetheless, we can also apply the Covariance-Penalty methods proposed in the previous section: the Implied Sharpe ratio (Imp SR - eq 13) and the R-Squared (eq 4) methods. 

In summary, our main hypothesis is: 

- Can the proposed approaches, Imp SR and R-Squared, outperform AIC and Naive in terms of out-sample/realised Sharpe ratio? 

In addition to this, we verified the level of alignment between the Expected/In-Sample and Realised/Out-sample Sharpe ratios. With that, our goal was to analyse if there was any additional benefit in terms of performance estimation that can be made by following a Covariance-Penalty approach. In the following subsections we present the data and methodology used to test such hypothesis, as well as the findings and hypothesis testing. 

## 4.1 Experimental Description

## 4.1.1 Datasets

Table 2 present aggregated statistics associated to the datasets used, whilst Figure 6 present the cumulative returns per asset pool. We have considered three main asset classes during our evaluation: equities, currencies, and fixed income. The data was obtained from Bloomberg, with the full list of 1361 assets tickers and period available at https://www.dropbox.com/s/oogzw8kaysx2qbp/data_list.csv?d1=0. 

## 4.1.2 Model Evaluation Scheme

We have established a testing procedure to assess different Covariance-Penalties methodologies. Figure 7 summarize the whole process. 

The process start by splitting a sequence of returns $r_{1}, \ldots, r_{T}$ in several in-sample (IS) and out-sample (OS) sets. The first pair, $IS_{1}$ and $OS_{1}$ , are determined by the trading horizon h and the first IS batch size I. After that, the whole sequence of $IS_{k}$ and $OS_{k}$ are created by simply accumulating $IS_{k} = \{IS_{k-1}, OS_{k-1}\}$ and rolling $OS_{k} = \{r_{I+(k-1)*h}, ..., r_{I+k*h}\}$ until $I + k * h \leq T$ . Every Covariance-Penalty methodology – Naive, Implied SR, R-squared and AIC – tap into the IS data to fine-tune a particular trading strategy. The measure of their success to avoid overfit is obtained when this strategy is applied in the OS set. Using the sequence of in-sample and out-sample results we are able to compute performance metrics to evaluate and test the different hypothesis. We have used h = 21 days and the initial period I = 1008 days in our experiments; this process is repeated per dataset and Covariance-Penalty. 


Table 2: Aggregated statistics of the assets used during our empirical evaluation.


<table><tr><td>Asset Pool*</td><td>N</td><td>Avg Return</td><td>Vol</td><td>Sharpe Ratio</td><td>Calmar Ratio</td><td>Monthly Skewness</td><td>VaR 95%</td></tr><tr><td>All Assets</td><td>1361</td><td>0.0173</td><td>0.0425</td><td>0.4064</td><td>0.0809</td><td>-0.9301</td><td>-0.9314</td></tr><tr><td>World Equity Indices</td><td>18</td><td>0.0152</td><td>0.0717</td><td>0.2114</td><td>0.0504</td><td>-0.8170</td><td>-1.0048</td></tr><tr><td>S&amp;P, SX5E, FTSE, DJIA, Russel and Nikkei Equities</td><td>1289</td><td>0.0172</td><td>0.0439</td><td>0.3915</td><td>0.0796</td><td>-0.8816</td><td>-0.9316</td></tr><tr><td>World Swaps Indices</td><td>14</td><td>-0.0197</td><td>0.0624</td><td>-0.3149</td><td>-0.0433</td><td>-0.1039</td><td>-0.9955</td></tr><tr><td>Rates Aggregate Indices</td><td>16</td><td>0.1220</td><td>0.0637</td><td>1.9135</td><td>0.5504</td><td>-0.8227</td><td>-0.9440</td></tr><tr><td>World Currencies</td><td>24</td><td>-0.0025</td><td>0.0315</td><td>-0.0798</td><td>-0.0157</td><td>-1.0052</td><td>-0.8856</td></tr></table>


* Before being averaged, each individual asset was volatility scaled to 10% 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/36a1804e26aac49237812685ea05386ddcb377bdb24dc7e9aec1e93d6c4b0e89.jpg)



Figure 6: Cumulative returns aggregated across asset pool. Before being averaged, each individual asset was volatility scaled to 10%


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/6dbd18863dc5f53ca2135f47bbdda86c5a8c856aafa8adcc600c447771782a30.jpg)



Figure 7: Window-based approach to assess covariance-penalties methodologies.


## 4.1.3 Trading Strategy and Performance Metrics

Below we present the steps taken to backtest the different trading strategies. For every pair of in-sample and out-sample data $(IS_{k}, OS_{k})$ , run: 

- For every lag $p \in \{3, 5, 7, 9, 12, 15, 18, 21, 26, 31, 36, 42, 49, 56, 63, 84, 105, 126\}$ , do: 

1. Fit the linear model using the in-sample set available $(IS_{k})$ : 

$$
r _ {t} = \beta_ {0} + \beta_ {1} r _ {t - 1} + \beta_ {2} r _ {t - 2} \dots + \beta_ {p} r _ {t - p} + \varepsilon_ {t} = \hat {r} _ {t} + \varepsilon_ {t}
$$

with the coefficients $\beta_{0}, \beta_{1}, \ldots$ identified via Ordinary or Total Least Squares. 

2. Compute the risk of choosing p using the in-sample set: 

$$
L _ {p} = \mathcal {L} (\dots , (\hat {r} _ {t - 1}, r _ {t - 1}), (\hat {r} _ {t}, r _ {t}), (\hat {r} _ {t + 1}, r _ {t + 1}), \dots ; M)
$$

with hat matrix M and P as a performance function, representing the different types of penalization used (AIC, Naive, Imp SR and R-squared). For the AIC and R-Squared it was also computed the expected Sharpe ratio $(\hat{S}R_{p})$ . 

- Pick $p^* = \text{argmax}_p\{L_3, L_5, ..., L_{126}\}$ , multiplying by -1 in the AIC case. 

- Fit the linear model using $p^*$ and the in-sample set available ( $IS_k$ ): 

$$
r _ {t} = \beta_ {0} + \beta_ {1} r _ {t - 1} + \beta_ {2} r _ {t - 2} \dots + \beta_ {p ^ {*}} r _ {t - p ^ {*}} + \varepsilon_ {t} = \hat {r} _ {t} + \varepsilon_ {t}
$$

- Test this model against the out-sample data $(OS_{k})$ . Store out-sample realised strategy returns, $L_{p*}^{(k)}$ and $\hat{S}\hat{R}_{p*}^{(k)}$ . 

With the sequence of results per asset in batches of in and out-sample, $\{(\hat{r}_{I}, r_{I}), (\hat{r}_{I+1}, r_{I+1}), \ldots, (\hat{r}_{I+h*k}, r_{I+h*k}); L_{p*}^{(k)}; \hat{S}R_{p*}^{(k)}\}_{k=1}^{K-1}$ , we computed some metrics related to the alignment between what was Expected and Realised: 

• Correlation ( $\rho$ as the Pearson correlation coefficient): 

$$
C o r r e l a t i o n = (\frac {1}{K - 1}) \sum_ {i = 1} ^ {K - 1} \rho (S R ^ {(1: i)}, \hat {S R} _ {p *} ^ {(k)})\tag{14}
$$

• Mean Absolute Difference (MAD) 

$$
M A D = (\frac {1}{K - 1}) \sum_ {i = 1} ^ {K - 1} | S R ^ {(1: i)} - \hat {S R} _ {p *} ^ {(k)} |\tag{15}
$$

where $SR^{(1:i)}$ is the Out-sample/Realised Sharpe ratio of a trading strategy until out-sample set i. Hence, the Correlation measures the association between the Expected and Realised Sharpe ratios, whilst MAD is measuring its distance; the best case being MAD = 0 and Correlation = 1. 

Finally, using the sequence as a whole per asset, $\{(\hat{r}_{I}, r_{I}), (\hat{r}_{I+1}, r_{I+1}), \ldots, (\hat{r}_{I+h*(K-1)}, r_{I+h*(K-1)})\}$ , we computed the Realised Sharpe Ratio. In all computations made, the Sharpe ratio was computed considering a 3-month Libor rate as the benchmark rate. 

## 4.2 Results and Discussions

Figure 8 present the aggregated results for the different Covariance-Penalty methods, and types of Least Squares across the 1361 assets. In terms of Realised Out-Sample (ROS) Sharpe ratio, using some Covariance-Penalty method provides a boost in the unconditional average of 80-100% percent in relation to the Naive approach. Our two proposed approaches, Imp SR and R-squared, also improves 20-30% in relation to the AIC lag selection. Overall, this pattern is also visible in terms of alignment between Expected and Realised Sharpe ratios: MAD is reduced and Correlation is increased. 

To check the main hypothesis, we have performed a statistical comparison between all approaches in terms of their ROS Sharpe ratio. Table 3 present the results of paired t-tests, controlling for multiple hypothesis testing using Bonferroni correction. Overall, three main patterns can be spotted: 

- All Covariance-Penalty approaches outperformed significantly the Naive approach; and our proposed two, Imp SR and R-Squared, also compares favourably in relation to the AIC. 

- R-Squared provided slightly better results than Imp SR, but its effect-size is small when compared to AIC and Naive cases. 

- TLS improves ROS Sharpe ratio in relation to OLS; as we have outlined before, since Sharpe ratio and Correlation are strongly associated, as TLS is maximizing Correlation, it will tend to perform better than OLS in relation to this metric. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/e987d7bdb619bc5b82705a37224288ea7829f64465cbd86304a40b57eba2aa89.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/ff91173feb91adb349767129a2447d1c0170eac8636b326eb0b0b061f80b0495.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/cbcbf53e07fe9bda05ddb48d84c130bc00093a54e92b2f5e9d4f10af1d71fb12.jpg)



Figure 8: Aggregated results for the different Covariance-Penalty methods, and types of Least Squares: Realised Out-Sample (ROS) Sharpe ratio, MAD and Correlation. The dot and bar represent the mean and standard error, respectively, across 1361 assets.



Table 3: Statistical comparisons between different Covariance-Penalty approaches and the baseline (Naive), using paired t-test and Bonferroni Correction for multiple hypothesis test. The boldfaced pairs are the ones not significantly different at 5% after Bonferroni correction.


<table><tr><td colspan="2">Val Method</td><td colspan="2">ROS Sharpe ratio</td><td colspan="2">p-value</td></tr><tr><td>Left</td><td>Right</td><td>Mean diff</td><td>t-stat</td><td>Value</td><td>Bonferroni</td></tr><tr><td>OLS-AIC</td><td>OLS-Imp SR</td><td>-0.0235</td><td>-3.7573</td><td>0.00018</td><td>0.00376</td></tr><tr><td>OLS-AIC</td><td>OLS-Naive</td><td>0.0575</td><td>10.061</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-AIC</td><td>OLS-R-Squared</td><td>-0.05351</td><td>-8.6793</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-AIC</td><td>TLS-Imp SR</td><td>-0.1036</td><td>-12.0062</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-AIC</td><td>TLS-Naive</td><td>0.03583</td><td>3.7806</td><td>0.000163</td><td>0.003430</td></tr><tr><td>OLS-AIC</td><td>TLS-R-Squared</td><td>-0.1048</td><td>-12.1588</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-Imp SR</td><td>OLS-Naive</td><td>0.08118</td><td>13.8264</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-Imp SR</td><td>OLS-R-Squared</td><td>-0.0299</td><td>-4.3723</td><td>0.00002</td><td>0.00028</td></tr><tr><td>OLS-Imp SR</td><td>TLS-Imp SR</td><td>-0.0800</td><td>-9.4425</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-Imp SR</td><td>TLS-Naive</td><td>0.05942</td><td>6.2843</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-Imp SR</td><td>TLS-R-Squared</td><td>-0.08122</td><td>-9.5304</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-Naive</td><td>OLS-R-Squared</td><td>-0.1111</td><td>-14.7347</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-Naive</td><td>TLS-Imp SR</td><td>-0.1612</td><td>-17.9283</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-Naive</td><td>TLS-Naive</td><td>-0.02176</td><td>-2.4326</td><td>0.01512</td><td>0.31753</td></tr><tr><td>OLS-Naive</td><td>TLS-R-Squared</td><td>-0.1624</td><td>-17.9653</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-R-Squared</td><td>TLS-Imp SR</td><td>-0.0501</td><td>-6.3812</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-R-Squared</td><td>TLS-Naive</td><td>0.08934</td><td>9.0067</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>OLS-R-Squared</td><td>TLS-R-Squared</td><td>-0.05129</td><td>-6.5487</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>TLS-Imp SR</td><td>TLS-Naive</td><td>0.1394</td><td>14.8044</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr><tr><td>TLS-Imp SR</td><td>TLS-R-Squared</td><td>-0.00118</td><td>-0.4581</td><td>0.64694</td><td>1.00000</td></tr><tr><td>TLS-Naive</td><td>TLS-R-Squared</td><td>-0.1406</td><td>-14.7307</td><td>&lt; 0.00001</td><td>&lt; 0.00001</td></tr></table>

There is also another explanation of why TLS has outperformed OLS: Figure 4.2 outline the similarity between approaches in terms of ROS Sharpe ration, and the most frequent (mode) lag selected across the in-out sample sets, averaged across 1361 assets. Overall, TLS penalized harder than OLS, making it many setups to adopt smaller lags, which in some sense, mimic a mean-reversion strategies; similarly, whenever OLS preferred smaller Hypothesis Spaces (less parameters), ROS Sharpe ratio rose. Overall ROS Sharpe ratio are quite correlated across the different methods and types, but some diversification might 


still be attained by pooling TLS and OLS models.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/3ace1b8b52d57de8ebf3e4858304657b5fdb13a4bcfbe001c9533918431e78fe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/b1d84f582d614398ca4f50aca99e48a92037d8be251a72ca1265dc8e91f1e02e.jpg)



Figure 9: Correlation matrix between ROS Sharpe ratio across methods and types, with the average most frequent (mode) lag selected across 1361 assets.


## 5 Conclusions

In this paper we proposed a new approach to dealing with financial overfitting, a Covariance-Penalty Correction, in which a risk metric is lowered given the amount of parameters and data used to underpins a trading strategy. We outlined the foundation and main results behind the Covariance-Penalty correction for trading strategies. After that, we pursued an empirical investigation, comparing its performance with some other approaches in the realm of Covariance-Penalties across more than 1300 assets, using Ordinary and Total Least Squares – with the last providing superior performance. Before delving into this novel perspective, we first briefly exhibit a historical context about the available methodologies. 

We outlined, chronologically, three distinct approaches in the literature to evaluate and deal with backtesting overfitting: Data Snooping, Overestimated Performance, and Cross-Validation Evaluation. The Data Snooping and Overestimated Performance authors identify that the root cause of any overfitting or spurious results is due to the multiple trials (hypothesis) employed by the analyst during the strategies fine-tuning step. In contrast, the Cross-Validation Evaluation and Covariance-Penalty Correction are focused on avoiding the apparent in-sample performance, by uncovering the expected generalisation (outsample) potential of a strategy. The Covariance-Penalty Correction methods require some assumptions on the joint behaviour of the asset and trading strategy, which is not the case for the majority of the Cross-Validation procedures. However, this lack of assumptions demand an extra computational processing in respect to the Covariance-Penalty approach. 

We then focused on to address the issue of overfitting of dynamic models, we look to estimated objective functions, taking degrees of freedom into account. Most commonly used estimates include Akaike and Bayes' information criteria, but these require the use of a likelihood. In our case, we do not have an exact likelihood, but instead choose to optimise the Sharpe ratio, and noting the monotonicity in $\rho$ , this comes from maximizing $\hat{\rho}$ . Consequently, we wish to consider the means of optimising correlation taking into account penalisations for degrees of freedom. In this sense, we presented formulas for adjusting estimated in-sample goodness of fits measures, to produce better estimates of the out-of-sample Sharpe ratio performance. Effectively, we would like to have a more sober estimate of the potential performance of our optimised strategies out of sample. 

We then performed an empirical investigation, in which the quantitative strategist is required to build a trading strategy based on lagged information of a certain asset returns. The most naive way (called Naive method in our experiments) is to try different lags and to pick the one that maximizes the Sharpe ratio. We compared it with the Akaike Information Criteria (AIC), as well as with the Covariance-Penalty methods proposed in this work: the Implied Sharpe ratio (Imp SR) and the R-Squared methods. In summary, our main hypothesis is: Can the proposed approaches, Imp SR and R-Squared, outperform AIC and Naive in terms of out-sample/realised Sharpe ratio? In addition to this, we verified the level of alignment between the Expected/In-Sample and Realised/Out-sample Sharpe ratios. With that, our goal was to analyse if there was any additional benefit in terms of performance estimation that can be made by following a Covariance-Penalty approach. 

Our findings suggest that Realised Out-Sample (ROS) Sharpe ratio, using some Covariance-Penalty method provides a boost in the unconditional average of 80-100% percent in relation to the Naive approach. Our two proposed approaches, Imp SR and R-squared, also improves 20-30% in relation to the AIC lag selection. Overall, this pattern is also visible in terms of alignment between Expected and Realised Sharpe ratios: MAD is reduced and Correlation is increased. Overall, all Covariance-Penalty approaches outperformed significantly the Naive approach; and our proposed two, Imp SR and R-Squared, also compares favourably in relation to the AIC. Total Least Squares (TLS) improves ROS Sharpe ratio in relation to Ordinary Least Squares (OLS). One of the explanations why TLS outcompeted OLS was that TLS penalized harder than OLS, making it many setups to adopt smaller lags, which in some sense, mimic a mean-reversion strategies. Similarly, whenever OLS preferred smaller Hypothesis Spaces (less parameters), ROS Sharpe ratio rose. Overall ROS Sharpe ratio are quite correlated across the different methods and types, but some diversification might still be attained by pooling TLS and OLS models. 

## Acknowledgement

Adriano Koshiyama would like to acknowledge the National Research Council of Brazil for his PhD scholarship. 

## References



[1] Hirotugu Akaike. A new look at the statistical model identification. IEEE transactions on automatic control, 19(6):716-723, 1974. 





[2] David H Bailey, Jonathan M Borwein, Marcos López de Prado, and Qiji Jim Zhu. Pseudomathematics and financial charlatanism: The effects of backtest over fitting on out-of-sample performance. Notices of the AMS, 61(5):458–471, 2014. 





[3] David H Bailey, Jonathan M Borwein, Marcos Lopez de Prado, and Qiji Jim Zhu. The probability of backtest overfitting. Journal of Computational Finance (Risk Journals), 2015. 





[4] David H Bailey and Marcos Lopez de Prado. The sharpe ratio efficient frontier. Journal of Risk, 15(2):13, 2012. 





[5] Yoshua Bengio and Yves Grandvalet. No unbiased estimator of the variance of k-fold cross-validation. Journal of Machine Learning Research, 5(Sep):1089–1105, 2004. 





[6] Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the royal statistical society. Series B (Methodological), pages 289-300, 1995. 





[7] Open Science Collaboration et al. Estimating the reproducibility of psychological science. Science, 349(6251):aac4716, 2015. 





[8] Timothy Falcon Crack. A classic case of "data snooping" for classroom discussion. Journal of Financial Education, pages 92-97, 1999. 





[9] Guolong Cui, Xianxiang Yu, Salvatore Iommelli, and Lingjiang Kong. Exact distribution for the product of two correlated gaussian random variables. IEEE Signal Processing Letters, 23(11):1662–1666, 2016. 





[10] Janez Demšar. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7(Jan):1-30, 2006. 





[11] Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toni Pitassi, Omer Reingold, and Aaron Roth. Generalization in adaptive data analysis and hold-out reuse. In Advances in Neural Information Processing Systems, pages 2350-2358, 2015. 





[12] Bradley Efron and Trevor Hastie. Computer age statistical inference, volume 5. Cambridge University Press, 2016. 





[13] Leonard P Freedman, Iain M Cockburn, and Timothy S Simcoe. The economics of reproducibility in preclinical research. PLoS biology, 13(6):e1002165, 2015. 





[14] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. The elements of statistical learning, volume 1. Springer series in statistics Springer, Berlin, 2001. 





[15] Wayne A Fuller. Measurement error models, volume 305. John Wiley & Sons, 2009. 





[16] Andrew Gelman and Eric Loken. The garden of forking paths: Why multiple comparisons can be a problem, even when there is no fishing expedition or p-hacking and the research hypothesis was posited ahead of time. Department of Statistics, Columbia University, 2013. 





[17] Gerd Gigerenzer and Peter M Todd. Simple heuristics that make us smart. Evolution and Cognition (Paper, 1999. 





[18] Gene H Golub and Charles F Van Loan. An analysis of the total least squares problem. SIAM journal on numerical analysis, 17(6):883-893, 1980. 





[19] JBS Haldane. Moments of the distributions of powers and products of normal variates. Biometrika, 32(3/4):226-242, 1942. 





[20] Peter R Hansen, Asger Lunde, and James M Nason. The model confidence set. Econometrica, 79(2):453-497, 2011. 





[21] Peter Reinhard Hansen, Asger Lunde, and James M Nason. Choosing the best volatility models: the model confidence set approach. Oxford Bulletin of Economics and Statistics, 65(s1):839–861, 2003. 





[22] Campbell R Harvey and Yan Liu. Backtesting. The Journal of Portfolio Management, pages 12–28, 2015. 





[23] Campbell R Harvey and Yan Liu. Lucky factors. Available at SSRN 2528780, 2015. 





[24] Megan L Head, Luke Holman, Rob Lanfear, Andrew T Kahn, and Michael D Jennions. The extent and consequences of p-hacking in science. PLoS biology, 13(3):e1002106, 2015. 





[25] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian journal of statistics, pages 65-70, 1979. 





[26] John PA Ioannidis. Why most published research findings are false. PLoS medicine, 2(8):e124, 2005. 





[27] Leon Isserlis. On a formula for the product-moment coefficient of any order of a normal frequency distribution in any number of variables. Biometrika, 12(1/2):134-139, 1918. 





[28] J Dave Jobson and Bob M Korkie. Performance hypothesis testing with the sharpe and treynor measures. The Journal of Finance, 36(4):889–908, 1981. 





[29] Ron Kohavi et al. A study of cross-validation and bootstrap for accuracy estimation and model selection. In IJCAI, volume 14, pages 1137-1145, 1995. 





[30] Oliver Ledoit and Michael Wolf. Robust performance hypothesis testing with the sharpe ratio. Journal of Empirical Finance, 15(5):850–859, 2008. 





[31] Wang Leyang. Properties of the total least squares estimation. Geodesy and geodynamics, 3(4):39-46, 2012. 





[32] Andrew W Lo. The statistics of sharpe ratios. Financial analysts journal, 58(4):36–52, 2002. 





[33] Andrew W Lo and A Craig MacKinlay. Data-snooping biases in tests of financial asset pricing models. Review of financial studies, 3(3):431–467, 1990. 





[34] Ivan Markovsky and Sabine Van Huffel. Overview of total least-squares methods. Signal processing, 87(10):2283-2302, 2007. 





[35] Graham Martin. Munchausen's statistical grid, which makes all trials significant. The Lancet, 324(8417):1457, 1984. 





[36] Saralees Nadarajah and Tibor K Pogány. On the distribution of the product of correlated normal random variables. Comptes Rendus Mathematique, 354(2):201-204, 2016. 





[37] John Douglas JD Opdyke. Comparing sharpe ratios: so where are the p-values? Journal of Asset Management, 8(5):308–336, 2007. 





[38] MD Rahman and Kai-Bor Yu. Total least squares approach for frequency estimation using linear prediction. IEEE Transactions on Acoustics, Speech, and Signal Processing, 35(10):1440–1454, 1987. 





[39] Joseph P Romano, Azeem M Shaikh, and Michael Wolf. Formalized data snooping based on generalized error rates. Econometric Theory, 24(02):404-447, 2008. 





[40] Joseph P Romano and Michael Wolf. Exact and approximate stepdown methods for multiple hypothesis testing. Journal of the American Statistical Association, 100(469):94–108, 2005. 





[41] Joseph P Romano and Michael Wolf. Stepwise multiple testing as formalized data snooping. Econometrica, 73(4):1237-1282, 2005. 





[42] Joseph P Romano and Michael Wolf. Efficient computation of adjusted p-values for resampling-based stepdown multiple testing. Statistics & Probability Letters, 113:38-40, 2016. 





[43] Marvin K Simon. Probability distributions involving Gaussian random variables: A handbook for engineers and scientists. Springer Science & Business Media, 2007. 





[44] Antti Suhonen, Matthias Lennkh, and Fabrice Perez. Quantifying backtest overfitting in alternative beta strategies. Journal of Portfolio Management, 43(2):90, 2017. 





[45] Ryan Sullivan, Allan Timmermann, and Halbert White. Dangers of data mining: The case of calendar effects in stock returns. Journal of Econometrics, 105(1):249–286, 2001. 





[46] HD Vinod and Matthew R Morey. Confidence intervals and hypothesis testing for the sharpe and treynor performance measures: A bootstrap approach. Computational Finance, 99, 1999. 





[47] Peter H Westfall and S Stanley Young. Resampling-based multiple testing: Examples and methods for p-value adjustment, volume 279. John Wiley & Sons, 1993. 





[48] Halbert White. A reality check for data snooping. Econometrica, 68(5):1097-1126, 2000. 





[49] Xian-Da Zhang. Matrix analysis and applications. Cambridge University Press, 2017. 



## A Full distributions for single period

In general, for X and R jointly normal, we have $X_{t}R_{t}$ is known to have a pdf, 

$$
p _ {x} = \frac {1}{\pi \sigma_ {R} \sigma_ {X}} \exp \bigl (\frac {\rho x}{\sigma_ {R} \sigma_ {X} (1 - \rho^ {2})} \bigr) K _ {0} \bigl (\frac {| x |}{\sigma_ {R} \sigma_ {X} (1 - \rho^ {2})} \bigr)
$$

where $K_{0}(\cdot)$ is a modified Bessel function of the $2^{nd}$ kind ([43], p 51, eq 6.15). The more general density for non-zero means, is given in [9] as an infinite series. In the special cases of independence and of correlated but zero mean, the expressions become much simpler and we choose to focus on the zero-mean case here. The density is unbounded at zero and has fat tails and positive skewness, becoming more pronounced with higher correlation. We can see the distribution for a variety of correlations in figure (10), with the skewness becoming increasingly pronounced for higher $\rho$ . In the limit as $\rho \rightarrow 1$ the distribution converges to that of the central $\chi^{2}$ distribution with one degree of freedom. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/4d36e27d-0c92-4c8f-81d7-95b26d35318c/1f4e1e944181cf21e6904c4c0e7f9c679cbddee602eeea80aa16fab62d6d34fa.jpg)



Figure 10: Complete product distributions for $\rho \in \{0, 0.2, 0.4, 0.6., 0.8, 1.0\}$ , normalised to have unit variance


## B Nonzero means: Sharpe ratios and Skewness

By an abuse of notation, we define $SR[R]$ to be $\mu_R / \sigma_R$ and by an abuse of notation, we define $SR[X] = \mu_X / \sigma_X$ (for $X$ the signal), 

Corollary 1: If $R \sim \mathcal{N}(\mu_R, \sigma_R^2)$ and $X \sim \mathcal{N}(\mu_X, \sigma_X^2)$ then 

$$
S R [ X \cdot R ] = \frac {S R [ R ] \cdot S R [ X ] + \rho}{(S R [ R ] ^ {2} + S R [ X ] ^ {2} + 2 \rho S R [ R ] \cdot S R [ S ] + \rho^ {2} + 1) ^ {1 / 2}}
$$

Corollary 2: If $R \sim \mathcal{N}(\mu_R, \sigma_R^2)$ and $X \sim \mathcal{N}(\mu_X, \sigma_X^2)$ then 

$$
\gamma_ {3} [ X \cdot R ] = \frac {2 \rho (\rho^ {2} + 3 + 3 S R [ R ] ^ {2} + 3 S R [ X ] ^ {2})}{(S R [ R ] ^ {2} + S R [ X ] ^ {2} + 2 \rho S R [ R ] \cdot S R [ X ] + \rho^ {2} + 1) ^ {3 / 2}}
$$

We note the one period Sharpe ratio of the strategy may depend on both the interaction between the Sharpe ratios of the Signals (weights) and the Returns, in particular whether they have the same sign or not, together with the sign of the correlation. In fact, the amplitude of the resulting strategy SR may be more dependent on the respective Sharpe ratios rather than $\rho$ since after all, $-1 \leq \rho \leq 1$ , while SR[R] and SR[X] may individually be above 1. 