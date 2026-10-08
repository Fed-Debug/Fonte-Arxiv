# Structural clustering of volatility regimes for dynamic trading strategies

Arjun Prakash $^{a*}$ , Nick James $^{b*}$ , Max Menzies $^{c*}$ and Gilad Francis $^{d}$ 

$^{a}$ School of Computer Science, University of Sydney, NSW, Australia; $^{b}$ School of Mathematics and Statistics, University of Melbourne, VIC, Australia; $^{c}$ Yau Mathematical Sciences Center, Tsinghua University, Beijing, China; $^{d}$ School of Mathematics and Statistics, University of Sydney, NSW, Australia. 

ARTICLE HISTORY 

Compiled November 25, 2021 

## ABSTRACT

We develop a new method to find the number of volatility regimes in a nonstationary financial time series by applying unsupervised learning to its volatility structure. We use change point detection to partition a time series into locally stationary segments and then compute a distance matrix between segment distributions. The segments are clustered into a learned number of discrete volatility regimes via an optimization routine. Using this framework, we determine a volatility clustering structure for financial indices, large-cap equities, exchange-traded funds and currency pairs. Our method overcomes the rigid assumptions necessary to implement many parametric regime-switching models, while effectively distilling a time series into several characteristic behaviours. Our results provide significant simplification of these time series and a strong descriptive analysis of prior behaviours of volatility. Finally, we create and validate a dynamic trading strategy that learns the optimal match between the current distribution of a time series and its past regimes, thereby making online risk-avoidance decisions in the present. 

Volatility modelling; regime-switching model; change point detection; spectral clustering; trading strategies 

## 1. Introduction

Modelling the volatility of a financial asset is an important task for traders and economists. Volatility may be modelled from an individual stock level to an index level; the latter can represent the uncertainty or systemic risk of an entire sector or economy. Financial markets are both significant in their own right and have substantial flow-on effects on the rest of society, as seen during the global financial crisis, US-China trade war, and COVID-19 pandemic. 

Statistical methods for volatility modelling have long been popular in the literature (Shah, Isah, and Zulkernine 2019; Kirchler and Huber 2007; Baillie and Morana 2009). Long-standing parametric methods such as ARCH (Engle 1982; Zakoian 1994) and GARCH (Bollerslev 1986) model the volatility of individual stocks, obeying assumptions such as stylized facts (Guillaume et al. 1997), and appropriately choosing parameters to best fit past data. These methods allow traders to model future returns, provided that their founding assumptions continue to hold. Volatility may also be studied indirectly through options pricing (Gonçalves and Guidolin 2006; Cont and da Fonseca 2002; Tompkins 2001). 

For many years, statisticians have noted that many real-world processes may exhibit significant nonstationarity. These switches in behaviour may disrupt the use of parametric or nonparametric models to predict future behaviour and so must be accounted for. For this purpose, a broad literature has been developed that aims to study nonstationary time series by partitioning them into locally stationary segments. Early research in a purely statistical context was carried out by (Priestley 1965; Priestley and Tong 1973; Ozaki and Tong 1975; Tong and Lim 1980) with significant advances by Dahlhaus (1997). 

Such concepts have been applied to financial time series to develop regime-switching models. These combine the aforementioned statistical techniques with the observation that financial assets exhibit such switching patterns, moving between periods of heightened volatility and ease (Hamilton 1989; Lavielle and Teyssière 2007; Lamoureux and Lastrapes 1990). Numerous regime-switching models have been developed to model such patterns (Hamilton 1989; Klaassen 2002; Guidolin and Timmermann 2007; Yang et al. 2018; de Zeeuw and Zemel 2012); these are also generally parametric, building on ARCH and GARCH, and must a priori specify the number of regimes and underlying distributions. Typically, the number of regimes is assumed to be two (Taylor 1999, 2005), one for high volatility and one for low volatility (Guidolin 2011). However, market crises such as the global financial crisis or COVID-19 may require more flexibility in volatility modelling, through varied assumptions regarding both individual distributions and the number of regimes (Arouri et al. 2016; Song, Ryu, and Webb 2016; Balcombe and Fraser 2017; Carstensen et al. 2020; Baiardi et al. 2020; Campani, Garcia, and Lewin 2021). For example, Baba and Sakurai (2011) describes three regimes in the VIX index from 1990-2010: tranquil, turmoil and crisis. Most recent regime-switching models have been statistical in nature, for instance building on the Heston model (Goutte, Ismail, and Pham 2017; Papanicolaou and Sircar 2013; Song, Ryu, and Webb 2018). 

This paper introduces a new method from an alternative perspective to analyze the regime-switching structure of a financial time series and use this for risk-avoidance in the future. Our method flexibly determines the number of volatility regimes, making no assumptions about the underlying data generating process. We use the entire history of the time series to learn its fundamental structure over time and how it behaves in volatile and non-volatile periods alike, improving on most parametric methods that only use recent behaviour. By not assuming the number of regimes a priori, we ameliorate a typical shortcoming of such models (Ang and Timmermann 2012). Moreover, our determination of the number of volatility regimes fits well within the existing literature as one could conceivably take our determined number for a given time series and input this learned number in an alternative regime-switching model. 

Subsequently, we draw on our findings and use additional learning procedures to design a dynamic trading strategy that can identify volatile periods and allocate capital in real time. We learn the volatility structure of the S%P 500 and determine whether the present period should be avoided by comparing its empirical distribution with past segments and determining the closest match by distance minimization. We show that it provides superior risk-adjusted returns to the S&P 500 index in various market conditions and suitably avoids the index during periods of higher volatility, generally associated with market downturns. Our procedure is inspired by both machine learning and metric geometry and contains additional optimization relative to previous trading strategies (Nystrup et al. 2016). 

In Section 2, we describe our methodology in detail, together with mathematical proof of its efficacy under specific circumstances. In Section 3, we validate our methodology on synthetic data, comparing several variants, and show a reasonable clustering structure can be determined for major indices, stocks, popular exchange-traded funds (ETFs) and currency pairs. In Section 4, we introduce our dynamic trading strategy, incorporating our insights and additional learning procedures. Section 5 summarizes our findings and describes future research and applications of our methods. 

## 2. Mathematical model

In mathematical statistics, a time series $(X_{t})$ is a sequence of random variables - measurable functions from a probability space to the real numbers - indexed by time. In finance, one generally conflates the random variable with the observed data point at each time. As such, a financial time series is a sequence of price data. In this paper, we will examine the time series of closing prices (across a range of assets) $(p_{t})_{t\geq0}$ at time t, and the associated log returns $(R_{t})_{t\geq1}=\log\left(\frac{p_{t}}{p_{t-1}}\right)$ . 

In this section, we describe our method in detail. We begin by assuming our nonstationary time series are generated from Dahlhaus locally stationary processes (Dahlhaus 1997) and proceed to partition the time series into stationary segments; specifically, we detect changes in the volatility of a time series via the Mood test change point method. Next, using the empirical cumulative density function of each segment, we use the Wasserstein metric to quantify distance between these distributions. We determine an allocation into an appropriate number of clusters via self-tuning spectral clustering (Zelnik-Manor and Perona 2004). Thus, we classify our segments of volatility into discrete classes without assuming the number of classes a priori. Finally, we record the number of clusters and their structure. 

Our methodology differs from usual regime-switching models in that it is inspired by unsupervised learning rather than stochastic volatility models. We use spectral clustering rather than simply testing for low or high volatility in the different segments precisely because two volatility regimes are not suitable in all contexts (Campani, Garcia, and Lewin 2021). As we study long histories of these time series, rather than recent behaviour for use in probabilistic models, we must account for the possibility of a varying number of regimes between different assets. Periods of ordinary market behaviour, turmoil, and crises, could potentially manifest themselves in three regimes for equities (Baba and Sakurai 2011), but perhaps the behaviour of currency pairs in these times may be quite different. 

The precise method that we describe below, applicable to volatility clustering, is not exhaustive. As long as there is consistency between the regime characteristic of interest, the change point algorithm (and its test statistic if applicable), and the distance metric between distributions, the method below could easily be reworked for detection and classification of regimes of alternative characteristics. For example, we could substitute the Jensen–Shannon divergence for the Wasserstein metric, substitute the Bartlett test for the Mood test, or smooth out the empirical distributions via kernel density estimation. Below, we outline the detailed implementation to model volatility and discuss some of the aforementioned alternatives in the process. 

## 2.1. Partition of the time series

Given time series price data, begin by forming the log return time series $(R_{t})_{t=1,\ldots,T}$ over a particular time interval. It is generally appropriate to assume the log returns independent random variables with mean close to 0, but not appropriate to assume they have any particular distribution (Ross 2013). 

With this in mind, we apply the nonparametric Mood test, performed in the CPM package of Ross (2015), to detect changes in the volatility of a time series. Although this is commonly known as a median test, it is also appropriate for detecting change in the variance between two distributions, as described in Section 4 of Mood (1954). More details on the change point framework and our specific implementation can be found in Appendix A. This yields a collection of change points $\tau_{1}, \ldots, \tau_{m-1}$ . For notational convenience, set $\tau_{0} = 1, \tau_{m} = T$ . The stationary segments according to this partition are then 

$$
(R _ {t}) _ {t \in [ \tau_ {j - 1}, \tau_ {j} ]}, j = 1, 2, \dots , m.
$$

This yields m stationary segments. Now let $(Y^{(j)})$ be the restricted time series whose entries are taken from the time interval $[\tau_{j-1},\tau_{j}]$ . That is, $(Y_{t}^{(j)})$ consists of the values $R_{t}$ where t ranges from $\tau_{j-1}$ to $\tau_{j}$ . Each $(Y^{(j)})$ has been determined by the algorithm to be sampled from a consistent distribution. 

## 2.2. Computation of the Wasserstein distance

Next, we compute the Wasserstein distance between the empirical distributions of each stationary segment $(Y^{(j)})$ . The Wasserstein metric (Kolouri et al. 2017), also known as the earth mover's distance, is the minimal work to move the mass of one probability distribution into another. Given probability measures $\mu,\nu$ on Euclidean space $R^{d}$ , we define 

$$
W _ {p} (\mu , \nu) = \inf _ {\gamma} \left(\int_ {(x, y) \in \mathbb {R} ^ {d} \times \mathbb {R} ^ {d}} | | x - y | | ^ {p} d \gamma (x, y)\right) ^ {\frac {1}{p}}.
$$

This infimum is taken over all joint probability measures $\gamma$ on $R^{d} \times R^{d}$ with marginal probability measures $\mu$ and $\nu$ . In the case where d = 1, this distance can be computed relatively simply in terms of the cumulative distribution functions associated to the two distributions. Given probability measures $\mu, \nu$ with cumulative distribution functions F, G, the distance $W_{p}(\mu, \nu)$ may be computed (del Barrio, Giné, and Matrán 1999) as 

$$
\left(\int_ {\mathbb {R}} | F - G | ^ {p} d x\right) ^ {\frac {1}{p}}.
$$

This allows us to form a $m \times m$ distance matrix of Wasserstein distances 

$$
D _ {i j} = W _ {p} (\mu_ {i}, \mu_ {j}) = W _ {p} (F _ {i}, F _ {j}), i, j = 1, \dots , m
$$

between the m distributions of each locally stationary segment of the log return time series. In our implementation, we set p = 1. 

## 2.3. Implementation of spectral clustering

In this section, we cluster the distributions by applying spectral clustering directly to the matrix D. This is a natural choice - alternative methods such as K-means would require the data to lie in Euclidean space. Spectral clustering often proceeds with the number of clusters k chosen a priori - we explore two methods for making this selection. 

We proceed according to the self-tuning spectral clustering methodology proposed by Zelnik-Manor and Perona (2004). From the distance matrix D, an affinity matrix A is defined according to 

$$
A _ {i, j} = \exp \left(\frac {- D _ {i j} ^ {2}}{\sigma_ {i} \sigma_ {j}}\right),
$$

where $\sigma_{i}$ are parameters to be chosen. In their original implementation, Zelnik-Manor and Perona (2004) select $\sigma_{i}=D_{i,7}$ to be the K=7th neighbour of the point i. To adapt this method to our needs, since occasionally there are fewer than 7 segments, we follow Hassanat et al. (2014) and set $K=\lceil\sqrt{m}\rceil$ , where m is the number of segments and set $\sigma_{i}=D_{i,K}$ . 

Next, one forms the Laplacian L and normalized Laplacian $L_{sym}$ following von Luxburg (2007). We define the diagonal degree matrix $Deg_{ii} = \sum_{j} A_{ij}$ , and then 

$$
\begin{array}{c} {L = \mathrm{Deg} - A;} \\ {L _ {\mathrm{sym}} = \mathrm{Deg} ^ {- 1 / 2} L \mathrm{Deg} ^ {- 1 / 2}.} \end{array}
$$

$L, L_{\mathrm{sym}}$ are $m \times m$ symmetric matrices, and hence are diagonalizable with all real eigenvalues. By the definition of $L$ and the normalization $L_{\mathrm{sym}}$ , all their eigenvalues are non-negative, $0 = \lambda_1 \leq \ldots \leq \lambda_m$ . 

We now describe two methods of determining the number of clusters k, which we will empirically investigate in Section 3. The first method is that described by Zelnik-Manor and Perona (2004). Obtaining the number of clusters proceeds by varying the number of eigenvectors $c \in [1, m]$ : 

(1) Find $x_{1}, \ldots, x_{c}$ the eigenvectors of $L$ corresponding to the $c$ largest eigenvalues. Form matrix $X = [x_{1}, \ldots, x_{c}] \in \mathbb{R}^{n \times c}$ . 

(2) Recover the rotation $\mathcal{R}$ which best aligns $X$ 's columns with the canonical coordinate system using the incremental gradient descent scheme. 

(3) Grade the cost of the alignment for each cluster number, $c \in [1, m]$ , according to (Zelnik-Manor and Perona 2004, p. 5, Eq 3). 

The final number of clusters $k^{ZP}$ is chosen to be the value of c with the minimal alignment cost. The second method chooses k to maximize the eigengap between successive eigenvalues. We denote this as $k^{e}$ , defined by 

$$
k ^ {e} = \operatorname * {a r g m a x} _ {c = 1, \ldots , m} \lambda_ {c} - \lambda_ {c - 1}.
$$

Having determined the number of clusters k, spectral clustering proceeds as follows. We compute the normalized eigenvectors $u_{1}, \ldots, u_{k}$ corresponding to the k smallest eigenvalues of $L_{sym}$ . We form the matrix $U \in R^{m \times k}$ whose columns are $u_{1}, \ldots, u_{k}$ . Let $v_{i} \in R^{k}$ be the rows of U, $i = 1, \ldots, m$ . These rows are clustered into clusters $C_{1}, \ldots, C_{k}$ according to k-means. Finally, we output clusters $A_{l} = \{i : v_{i} \in C_{l}\}, l = 1, \ldots, k$ to assign the original m elements, in this case segment distributions, into the corresponding clusters. 

## 2.4. Theoretical justification of model

In this section, we introduce a mathematical framework: under certain conditions (introduced in Definition 2.2), our aforementioned methodology is proven to accurately cluster segments by high or low volatility. Propositions 2.3 and 2.5 are intermediate technical results that allow us to establish this claim in Theorem 2.6. 

For the remainder of this section, let $f, g, f_i$ denote continuous probability density functions (pdf's) and let $F, G, F_i$ denote their respective cumulative density functions (cdf's). We recall the necessary properties. Such a function $f$ is continuous $f: \mathbb{R} \to \mathbb{R}$ , non-negative everywhere, with $\int_{-\infty}^{\infty} f(t) dt = 1$ . The cumulative density function $F$ is defined by $F(x) = \int_{-\infty}^{x} f(t) dt$ . 

Definition 2.1. Say a continuous probability density function f is median-zero if one of three equivalent conditions hold: 

$$
\begin{array}{l} (1) \int_ {- \infty} ^ {0} f (x) d x = \frac {1}{2}; \\ (2) \int_ {0} ^ {\infty} f (x) d x = \frac {1}{2}; \\ (3) F (0) = \frac {1}{2}. \end{array}
$$

In the next definition, we define a partial ordering $\prec$ that describes when one probability distribution is of consistently greater spread than another. 

Definition 2.2. Let $f, g$ be two median-zero continuous pdf's. Say $g$ has greater volatility than $f$ everywhere, denoted $f \prec g$ if $F(x) < G(x)$ for $x \in (-\infty, 0)$ while $F(x) > G(x)$ for $x \in (0, \infty)$ . 

We make several remarks on this definition. First, this is a partial rather than a total ordering, as two arbitrary cdf's $F, G$ may frequently intersect for $x \in \mathbb{R}$ . Second, we note that this notation is only used in Section 2.4. Third, this is quite a strong definition - while volatility may sometimes refer to only the second moment, this definition carries information about higher moments as well - broadly speaking if $f \prec g$ then $g$ has all higher moments greater than that of $f$ . Finally, the utility of this definition is predicated upon the following proposition and corollary. For an example of two pdf's that fit the scope of both Proposition 2.3 and Corollary 2.4, we include Figure 1. 

Proposition 2.3. Let $f, g$ be two median-zero continuous pdf's. Suppose there exist $\tau_1, \tau_2 > 0$ such that $f(x) > g(x)$ for $x \in (-\tau_1, \tau_2)$ but $f(x) < g(x)$ if $x < \tau_1$ or $x > \tau_2$ . Then $f \prec g$ . 

This proposition gives a means to establish that $f \prec g$ by quick inspection. 

Proof. First, the continuity of f and g implies that $f(\tau_{1}) = g(\tau_{1})$ and $f(\tau_{2}) = g(\tau_{2})$ . Next, let $x \leq \tau_{1}$ . By integrating f and g over the interval $t \in (-\infty, x)$ , it is immediate that $F(x) < G(x)$ . Now, consider the interval $[\tau_{1}, 0]$ and the function h = F - G on this interval. We have established that $h(\tau_{1}) < 0$ while $h(0) = 0$ by the median-zero property. By the fundamental theorem of calculus, $h$ is differentiable on the interval with $h'(t) = f(t) - g(t) > 0$ for $t \in (\tau_1, 0)$ . Thus $h$ is increasing on the interval $[\tau_1, 0]$ . Since $h(0) = 0$ , this implies $h(t) < 0$ for $t \in (\tau_1, 0)$ , that is $F(t) < G(t)$ on the interval $t \in (\tau_1, 0)$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/5e1b665915bd4cf97957bd102de5b95069c976dd87896f8d2af857f18f73168e.jpg)



Figure 1. Normal distributions satisfying conditions of Proposition 2.3 and Corollary 2.4. These results prove that $N(0,1) \prec N(0,2)$ .


Together with the earlier fact that $F(x) < G(x)$ for $x \leq \tau_{1}$ , we deduce that $F(x) < G(x)$ for all x < 0. By an identical argument, we deduce $F(x) > G(x)$ for all x > 0. That is, $f \prec g$ , which concludes the proof. ☐ 

Corollary 2.4. Let $0 < \sigma_{1} < \sigma_{2}$ and let $f_{1}, f_{2}$ be the pdf's associated to normal distributions $N(0, \sigma_{1}), N(0, \sigma_{2})$ respectively. Then $f_{1} \prec f_{2}$ . 

Proof. We have $f_{i}(x) = \frac{1}{\sqrt{2\pi}\sigma_{i}} \exp\left(-x^{2}/2\sigma_{i}^{2}\right)$ for i = 1, 2. By simplifying, $f_{1}(x) > f_{2}(x)$ if and only if 

$$
x ^ {2} <   \frac {2 \sigma_ {1} ^ {2} \sigma_ {2} ^ {2}}{\sigma_ {2} ^ {2} - \sigma_ {1} ^ {2}} \log \left(\frac {\sigma_ {2}}{\sigma_ {1}}\right).\tag{1}
$$

Let $\tau$ be the square root of the right hand side of (1). Then $f_{1}(x) > f_{2}(x)$ precisely for $x \in (-\tau, \tau)$ , which establishes $f_{1} \prec f_{2}$ . 

The following lemma is a technical result for evaluating Wasserstein distances between pdf's that can be ordered with $\prec$ . 

Proposition 2.5. Let f < g. Then the Wasserstein distance $W(f, g)$ can be computed 

$$
W (f, g) = \int_ {- \infty} ^ {0} (G - F) d x + \int_ {0} ^ {\infty} (F - G) d x.\tag{2}
$$

Proof. In general, $W(f,g)$ may be computed (del Barrio, Giné, and Matrán 1999) as 

$$
\int_ {\mathbb {R}} | F - G | d x.
$$

In the case where $f \prec g$ , F - G is non-negative on $[0, \infty)$ and negative on $(\infty, 0)$ . Thus this integral can be expressed as a sum 

$$
\int_ {- \infty} ^ {0} | F - G | d x + \int_ {0} ^ {\infty} | F - G | d x = \int_ {- \infty} ^ {0} (G - F) d x + \int_ {0} ^ {\infty} (F - G) d x,
$$

as required. 

Theorem 2.6. Suppose $f_1, \ldots, f_n$ is a collection of median-zero pdf's that can be totally ordered by $\prec$ . Then our method described in Section 2 partitions the segments into intervals under $\prec$ . That is, if $f_1 \prec \cdots \prec f_n$ then the methodology necessarily segments into clusters $C_1 = \{f_1, \ldots, f_{n_1}\}, C_2 = \{f_{n_1+1}, \ldots, f_{n_2}\}, \ldots, C_k = \{f_{n_{k-1}+1}, \ldots, f_{n_k}\}$ . Moreover, our methodology is also resistant to deformations: if $f_1, \ldots, f_n$ are a sufficiently small deformation away from totally ordered median-zero pdf's, then the result still holds. 

Proof. First we assume the collection can be totally ordered under $\prec$ . By relabelling, without loss of generality $f_{1} \prec \ldots f_{n}$ . Then, if i < j, 

$$
W (f _ {i}, f _ {j}) = \int_ {- \infty} ^ {0} (F _ {j} - F _ {i}) d x + \int_ {0} ^ {\infty} (F _ {i} - F _ {j}) d x\tag{3}
$$

$$
= \left(\int_ {- \infty} ^ {0} (F _ {j} - F _ {1}) d x + \int_ {0} ^ {\infty} (F _ {1} - F _ {j}) d x\right)\tag{4}
$$

$$
- \left(\int_ {- \infty} ^ {0} (F _ {i} - F _ {1}) d x + \int_ {0} ^ {\infty} (F _ {1} - F _ {i}) d x\right).\tag{5}
$$

Thus, let $a_{i}=\int_{-\infty}^{0}(F_{j}-F_{1})dx+\int_{0}^{\infty}(F_{1}-F_{j})dx$ . Then $0=a_{1}<a_{2}<\cdots<a_{n}$ and $W(f_{i},f_{j})=a_{j}-a_{i}$ by (3), (4) and (5) above. That is, the distance matrix D between the segments $f_{1},\ldots,f_{n}$ is identical to a distance matrix between real numbers $a_{1}<\cdots<a_{n}$ . Spectral clustering applied to real numbers partitions them into intervals, that is $C_{1}=\{a_{1},\ldots,a_{n_{1}}\},C_{2}=\{a_{n_{1}+1},\ldots,a_{n_{2}}\},\ldots,C_{k}=\{a_{n_{k-1}+1},\ldots,a_{n_{k}}\}$ for some integers $1\leq n_{1}<\cdots<n_{k}=n$ . Since $D_{ij}=a_{j}-a_{i}$ the results of spectral clustering are identical to applying it on the corresponding real numbers. Hence the segments are partitioned into corresponding clusters $C_{1}=\{f_{1},\ldots,f_{n_{1}}\},C_{2}=\{f_{n_{1}+1},\ldots,f_{n_{2}}\},\ldots,C_{k}=\{f_{n_{k-1}+1},\ldots,f_{n_{k}}\}$ as required. 

Finally, if the initial collection of pdf's were not precisely median-zero and totally ordered under $\prec$ but a sufficiently small deformation away from a median-zero totally ordered collection, then the result still holds. By the continuity of the Wasserstein distance, the distance matrix D would be a small deformation away from a distance matrix between real numbers. By the continuity of the procedures within spectral clustering, once again the $f_{i}$ would be partitioned into clusters of increasing volatility. 

## 3. Results

## 3.1. Synthetic data experiments

In this section, we validate our method on 200 synthetic time series. In two distinct experiments, we randomly generate 100 times series each with artificially pronounced breaks in volatility by concatenating different segments. Each segment is randomly drawn from various data generating processes and randomly chosen between 200 and 300 in length, together with added Gaussian random noise $\delta, \epsilon$ to ensure none of the data generating processes are identical. Thus, we create 200 unique time series where a ground truth number of clusters and the segment memberships are known - to these we apply our methodology to validate it. In a small number of instances, our methodology does not correctly identify the number of segments at the change point detection phase - we term this a mismatch. Excluding these, when the change point detection step correctly identifies the number of segments, we evaluate the quality of the clustering using the Fawlkes-Mallows index (Fowlkes and Mallows 1983; Halkidi, Batistakis, and Vazirgiannis 2001): 

$$
F M I = \sqrt {\frac {T P}{T P + F P} \cdot \frac {T P}{T P + F N}},
$$

where TP, FP, FN are the number of true positives, false positives and false negatives, respectively. The score is bounded between 0 and 1, where 1 indicates a perfect cluster assignment. This FMI can only be applied when the change point detection step correctly identifies the number of segments, which we call correctly matched. 

In the first experiment, we draw from five different normal distributions, as follows: 

$$
\begin{array}{r l} & X _ {1} \sim \mathcal {N} (0 + \delta , (0. 2 5 + \epsilon) ^ {2}) \\ & X _ {2} \sim \mathcal {N} (0 + \delta , (0. 5 + \epsilon) ^ {2}) \\ & X _ {3} \sim \mathcal {N} (0 + \delta , (1 + \epsilon) ^ {2}) \\ & X _ {4} \sim \mathcal {N} (0 + \delta , (2 + \epsilon) ^ {2}) \\ & X _ {5} \sim \mathcal {N} (0 + \delta , (4 + \epsilon) ^ {2}) \end{array}
$$

They must be approximately mean-centred to mimic the properties of the log returns time series and for the Mood test to detect change points in the variance, as detailed in Appendix A. 

In this first experiment, there were 4 out of 100 mismatches. In all these instances, the algorithm was off by just one in detecting the correct number of segments. Among the 96 remaining time series, we are able to apply the two different methods to select the number of clusters and apply spectral clustering, as described in Section 2.3. We refer to the two methods as the eigengap method to generate $k^{e}$ and the gradient descent method to generate $k^{ZP}$ , as defined in 2.3. To validate our methodology, we record both the mean FMI score for these two methods as well as smoothed histograms 

over the 96 experiments. 

In Figure 2, we present one example of such a time series. Figure 2(a) displays the change point partitions; the detection times, as described in Appendix A, are not visible. Figure 2(b) displays the kernel density estimations of the distributions, coloured according to their membership in five detected clusters. Figure 2(c) shows the final clustering of the segments of the synthetic time series. This whole procedure correctly identifies the change in variance, as well as the existence of five regimes (clusters) of volatility. 

Across all 96 correctly matched iterations of this first experiment, the mean FMI score for the eigengap method $(k^{e})$ was 0.93, while the mean FMI score for the gradient descent method $(k^{ZP})$ was 0.86. The smoothed histogram in Figure 3(a) shows that the eigengap method has systematically higher FMI scores. 

For the second experiment, we repeated the same procedure, however we drew from five Laplace distributions: 

$$
\begin{array}{r} X _ {1} \sim \mathcal {F} _ {\mathcal {L}} (0 + \delta , 0. 2 5 + \epsilon) \\ X _ {2} \sim \mathcal {F} _ {\mathcal {L}} (0 + \delta , 0. 5 + \epsilon) \\ X _ {3} \sim \mathcal {F} _ {\mathcal {L}} (0 + \delta , 1 + \epsilon) \\ X _ {4} \sim \mathcal {F} _ {\mathcal {L}} (0 + \delta , 2 + \epsilon) \\ X _ {5} \sim \mathcal {F} _ {\mathcal {L}} (0 + \delta , 4 + \epsilon) \end{array}
$$

In this second experiment, there were 11 out of 100 mismatches. Once again, the algorithm was off by just one in detecting the correct number of segments in all instances. Across all 89 correctly matched iterations of this second experiment, the mean FMI score for the eigengap method ( $k^{e}$ ) was 0.94, while the mean FMI score for the gradient descent method ( $k^{\bar{Z}P}$ ) was 0.85. Again, the smoothed histogram in Figure 3(b) shows that the eigengap method has systematically higher FMI scores than the alternative. 

Manually examining the mismatches shows that they tend to occur when the one segment is sandwiched between two segments drawn from the next most similar distribution, for example when a $\mathcal{N}(0,1)$ segment is sandwiched between two $\mathcal{N}(0,2)$ segments. When this happens, the algorithm tends to split the segments into four parts instead of three, leading to the overestimation. It is also worth noting that we repeated both sets of experiments using the Bartlett test instead of the Mood test to help reduce mismatches. While both tests performed similarly in the first experiment, the Bartlett test had a much higher mismatch rate in the second experiment, leading to much higher estimations of the number of clusters. This is to be expected as the Barlett test is a parametric test for changes in normal distributions. The better performance of the Mood test in both the normal and Laplace experiments indicates that it is better suited to real-world log returns data, which are not necessarily normally distributed. Having observed few mismatches and better FMI scores with the eigengap method, we proceed by exclusively applying the eigengap method to real data in subsequent sections. 

## 3.2. SPY

In this section, we apply our method to SPY, an ETF of the S&P 500, and analyze our volatility clustering results. We study adjusted closing price data from Yahoo! Finance, https://finance.yahoo.com, from 1 January 2008 to 31 December 2020, and compute the log returns before applying our methodology. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/7972fb48c6b25de7d42bd25c7825b3fc490312d0e7d3442246c28075d5e31c71.jpg)



(a) Synthetic time series change points and detection times


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/f21c8def8df6e4c5b8f5a213b5c7b307d89c90515678dad1d9d7e4b5366ef073.jpg)



(b) Kernel density estimation plots of synthetic distributions, forming five clusters


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/b8d92b46a3c3227f0447a218a8771555a0758a3a7b656c8bcbf49d639bf9efe4.jpg)



(c) Five determined volatility regimes



Figure 2. Synthetic data experiment for normal distributions (five regimes)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/490c4f5aa125c223a27036c30d9cd8ad76b0fb1ef4b630014341d6f39fcd6907.jpg)



(a) Distribution of FMI scores over 96 correctly matched instances


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/04e97f82317f6d4c60a2ab2d0c9dc4c9e6e54500daaba09a8b32c7bb3df3d99d.jpg)



(b) Distribution of FMI scores over 89 correctly matched instances



Figure 3. Smoothed histograms for FMI scores from the eigengap and gradient descent method for (a) synthetic Gaussian and (b) synthetic Laplace experiments


Our algorithm finds five volatility clusters over the 13-year period studied, seen in Figure 4. The cyan cluster is associated with extreme market behaviours such as the worst part of the global financial crisis (GFC). The next most volatile cluster is displayed in yellow, and the majority of the GFC, the August 2011 crash and the entirety of 2020 are grouped in this period. The red cluster contains the start of the GFC, the 2010 flash crash and the 2015-16 sell off and the US/China Trade War. Finally the blue and green clusters display more stable economic periods. Our trading strategy is predicated upon the idea that we associate the most recent window with a period in the past that is most similar. Details are described in Section 4. We observe that a change point is detected between the 17th and 18th segment of Figure 4(c), and yet there was no regime change in volatility at this time. This can occur when the distributions are different, but not different enough to warrant an entire regime change. 

## 3.3. Empirical results on various asset classes

In this section, we apply our methodology to a collection of asset classes including equity indices, individual stocks, ETFs and currency pairs. For each asset, we report the learned number of segments and clusters. This information is used to design the trading strategy. We display these results in Tables 1, 2, 3 and 4, respectively. 

Table 1 displays results for the S&P 500, Dow Jones, Nikkei, FTSE, ASX 200 indices. Table 2 displays results for large firms including: MSFT (Microsoft), APPL (Apple), AMZN (Amazon), GOOG (Alphabet) and BRK-A (Berkshire Hathaway Class A). Table 3 displays results for the collection of ETFs studied, including: RYT (Invesco S&P 500 Equal Weight Technology ETF), GLD (SPDR Gold Shares), XLF (Financial Select Sector SPDR Fund), IJS (iShares SP Small-Cap 600), MSCI (World Index Fund). Finally, Table 4 displays results for several currency pairs. In Appendix B, we provide figures displaying various time series' partitioned volatility regimes, and the respective clusters in which they are associated. 


Table 1. Volatility cluster structure for major indices


<table><tr><td>Ticker</td><td>No. Segments</td><td>No. Clusters</td><td>Cluster Sizes</td></tr><tr><td>S&amp;P 500</td><td>19</td><td>5</td><td>3, 8, 4, 3, 1</td></tr><tr><td>Dow Jones</td><td>19</td><td>5</td><td>5, 6, 3, 3, 2</td></tr><tr><td>Nikkei 225</td><td>17</td><td>2</td><td>6, 11</td></tr><tr><td>MSCI</td><td>15</td><td>4</td><td>4, 5, 4, 2</td></tr></table>


Table 2. Volatility cluster structure for large firms


<table><tr><td>Ticker</td><td>No. Segments</td><td>No. Clusters</td><td>Cluster Sizes</td></tr><tr><td>MSFT</td><td>15</td><td>4</td><td>6, 7, 1, 1</td></tr><tr><td>AAPL</td><td>15</td><td>3</td><td>4, 5, 6</td></tr><tr><td>AMZN</td><td>14</td><td>2</td><td>7, 7</td></tr><tr><td>GOOG</td><td>13</td><td>2</td><td>6, 7</td></tr><tr><td>BRK-A</td><td>15</td><td>4</td><td>5, 4, 3, 3</td></tr></table>

## 3.4. Sensitivity of methodology

All results in Section 3 are obtained over a 13-year period from 2008 to 2020. In this brief section, we discuss the effects of extending or subdividing our window of analysis, which will be particularly relevant for Section 4. When the period is extended or subdivided, results may change in numerous ways. First, the change point algorithm used (described in Appendix A) feeds in one data point in at a time. It may continually detect new change points, subject to a lower bound of 30 days before a new change point is permitted. Thus, extending the period of analysis will necessarily only increase the number of determined segments. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/035b1d3f2b769579c063cbe5a5a1fc0a8dfae3fd6c3e990aaa869f33ff768c94.jpg)



(a) SPY change points and detection times


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/d8ba2266cecc2944c4041839544473605c2f87281a8acb44e8cf574248c843c2.jpg)



(b) Kernel density estimation plots of SPY distributions, forming five clusters


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/2f178d0299ac8138a071efb83c50f9a9cfa3e1a9771fd5326c6f9e81cb98a034.jpg)



(c) Five determined volatility regimes for SPY



Figure 4. Volatility clustering results for SPY



Table 3. Volatility cluster structure for popular ETFs


<table><tr><td>Ticker</td><td>No. Segments</td><td>No. Clusters</td><td>Cluster Size</td></tr><tr><td>GLD</td><td>15</td><td>3</td><td>4, 6, 5</td></tr><tr><td>XLF</td><td>21</td><td>3</td><td>9, 8, 4</td></tr><tr><td>IJS</td><td>18</td><td>6</td><td>3, 4, 4, 3, 2, 2</td></tr><tr><td>RYT</td><td>17</td><td>4</td><td>4, 4, 5, 4</td></tr></table>


Table 4. Volatility cluster structure for currency pairs


<table><tr><td>Ticker</td><td>No. Segments</td><td>No. Clusters</td><td>Cluster Size</td></tr><tr><td>AUD/USD</td><td>14</td><td>3</td><td>5, 5, 4</td></tr><tr><td>EUR/USD</td><td>14</td><td>2</td><td>8, 6</td></tr><tr><td>GBP/USD</td><td>13</td><td>4</td><td>3, 5, 3, 2</td></tr><tr><td>NZD/USD</td><td>12</td><td>3</td><td>4, 5, 3</td></tr></table>

On the other hand, suppose the total period is subdivided into smaller intervals. For example, suppose a longer period $a \leq t \leq b$ is partitioned into $a \leq t \leq c$ and $c \leq t \leq b$ . Broadly speaking, the number of change points in $[a, b]$ will usually be the sum of the change points in $[a, c]$ and $[c, b]$ . This does not necessarily hold all the time, for example if a change point in $[a, b]$ is detected close to t = c; then it would feature in $[a, b]$ but potentially in neither $[a, c]$ nor $[c, b]$ . Translating this into the number of segments, this means that $n_{[a,b]}$ is usually $n_{[a,c]} + n_{[c,b]} - 1$ , where $n_{I}$ is the number of segments observed within an interval I. However, this may not hold exactly. 

The total number of clusters, however, is more difficult to predict regarding how it will behave with different period lengths. For example, as the analysis window is extended, more segments may be observed, yielding a larger space of segments to be clustered. However, as a space (in this case, consisting of probability densities) grows in cardinality, its determined number of clusters may change unpredictably. Usually, a larger cardinality of points in a high-dimensional space should produce a greater number of determined clusters. Yet this is not always the case. Consider a hypothetical example of a space neatly divided into three clusters (indexed 1, 2, and 3). If enough data points are added to “bridge” clusters 2 and 3, then clustering may subsequently detect only two clusters (1 and $2 \cup 3$ ). In our implementation, we broadly notice that smaller periods of analysis produce fewer clusters. This will be relevant in the precise trading methodology of Section 4. 

Finally, we remark that even two adjacent segments may lie in the same determined cluster. Indeed, this was demonstrated in Section 3.2. Fortunately, this works well in the context of our methodology. Synthetic experiments demonstrate that the change point detection algorithm identifies more change points than the ground truth time series in some circumstances. Our clustering overcomes this potential issue of oversensitivity. In the event that an erroneous change point partitions two adjacent segments with similar statistical properties, the clustering method would determine that these segments exist in the same cluster, ameliorating the false positive of the change point algorithm. 

## 3.5. Discussion of findings

Our empirical results demonstrate substantial heterogeneity in the volatility structure among the asset classes studied. The number of clusters ranges between 2 and 6, with the most common number of volatility regimes determined to be 3 or 4. The number of segments belonging to each clustering regime is relatively consistent among all asset classes. The one exception is at the height of the GFC, where a single segment is associated with one cluster. When a shorter time period is used, two regimes are consistently identified, which is consistent with prior findings (Guidolin 2011). 

Despite heterogeneity in the number of segments and clusters, we can still identify similarities between asset classes. For instance, the S&P 500 and the Dow Jones both exhibit periods of high volatility in March 2010, April 2011, and late 2018. All but several asset classes exhibit a volatile clustering regime at the height of the GFC. Typically, this period belongs to the same cluster as the COVID-19 market crash of 2020 - highlighting similar market structure during periods of severe crisis (James 2021). In addition, all five of the listed firms experienced a similarly volatile window associated with the US/China trade war in late 2018. By contrast, the ETFs under analysis did not exhibit similarly volatile periods - most likely due to their varied asset class mix. 

In their typical formulation, regime-switching models require assumptions regarding the number of regimes and candidate distributions a priori. They are often criticized for this highly parameterized structure (Guidolin 2011; Ang and Timmermann 2012). One meaningful advantage of our method, is the ability to account for extreme economic events and market crises with the flexibility of our method. No assumptions regarding the number of regimes or data generating process are required - indeed, we have validated our method both on Gaussian and non-Gaussian distributions. For those wishing to implement parametric regime-switching models, the methodology proposed in this work could be used as an accompaniment to algorithmically determine an appropriate number of regimes for various asset classes. As a further application, in the proceeding section we demonstrate how these results can be used to inform asset allocation decisions in the context of a dynamic trading strategy. 

## 4. Application of results: trading strategy

In recent times, passive investing has gathered more asset inflows than active investment management. In particular, index funds and ETFs that track major indices such as the S&P 500 are a popular way of attaining broad market exposure for investors. We apply our analysis to the S&P 500 index to determine a dynamic trading strategy that can simultaneously benefit from the index's appreciation while minimising risk. In Section 3.2, we determined that the S&P 500 has two distinct volatility regimes, captured in two distinct clusters of volatility periods. Our contrived trading strategy is to buy and hold SPY, a tracker of the S&P 500, in low volatility periods, and then flee to the safe haven of GLD, a gold bullion tracker, in high volatility periods. We improve on the previous work of Nystrup et al. (2016), who uses a live implementation of the rank test to move away from the S&P 500. This method has two drawbacks: first, as noted in Section 3, a change point does not necessarily indicate a change in regime; secondly, their method has an unpredictable delay in registering the change point, as discussed in Appendix A. 

Instead, we implement a dynamic procedure with a 4-year sliding window. Model parameters are learned within the prior window, and then applied to the proceeding four years of data. Suppose our algorithm begins with years 0:4. First, we analyze the SPY over the prior 4-year period, years -4:0. We determine the cluster structure of the distribution segments of the SPY over this prior period. To make investment decisions in the current period of 0:4 years, we try to match the present distribution with the most similar distribution in the prior window. We combine metric geometry and unsupervised learning for this purpose, minimizing a metric to one of an existing set of candidate segments. Specifically, we examine the present local distribution of the last n days, where n is a learned parameter, and determine the minimal distance between the present local distribution and the set of segment distributions of the prior 4-year period. We call n the look back length of the procedure. If this closest prior segment lies in (one of) the most volatile class of past distributions (characterized by greatest variance), we determine that the local distribution is volatile, and allocate all capital toward gold. This method works even if more than 2 volatility clusters are found during the previous window. For example, if r volatility regimes are determined to exist, the algorithm could avoid SPY if the current period is matched to the most volatile $\lceil r/2\rceil$ such regimes, ordered by variance. Fortunately, in our implementation, the number of volatility regimes in the 4-year windows is always 2. 

The parameter n is optimized relative to the -4 : 0 year window. Specifically, having determined the cluster structure, n is chosen to optimize the Sharpe ratio, a well-established measure of risk-adjusted returns, when testing over that window. We optimize n over a range $20 \leq n \leq 30$ , that is, 4 to 6 trading weeks. Thus, n is learned in this prior window and then used in the algorithm in the subsequent window. The window is then successively slid forward four years, and the process repeats. That is, model parameters estimated on years 0 : 4 are used to forecast in years 4 : 8, and so on. 

There are several reasons for our choice of 4-year windows. First, it is a suitable compromise in the length of the training data. If the training period were too short, we could erroneously capture transient behaviours as persistent market dynamics. By contrast, if the training period were too lengthy, our strategy may fail to prioritize more recent and relevant dynamics. Specifically regarding our use of clustering, our trading strategy always makes decisions (in the present) having been trained on data that was sufficiently recent to learn up-to-date behaviour of volatility regimes, but sufficiently long to observe non-trivial clustering results. In addition, this adaptive sliding window technique allows us to convincingly validate the long-run performance of our trading strategy. Second, as discussed in Section 3.4, the number of volatility regimes generally increases with the window length. Across 4-year training periods, our algorithm always determines r = 2 to be the optimal number of clusters when we examine SPY in its partitioned chunks, as shown in Table 5. This makes the decision process of avoiding volatile regimes simpler and less ambiguous - if the current distribution is matched (by minimal distance) to the more volatile (of two) class of distributions, we allocate all capital towards gold. Third, a 4-year period is suitable as equity markets follow 4-year cycles, associated with the cyclicality of Kitchin cycles (Korotayev and Tsirel 2010) and the US presidential election (Gärtner and Wellershoff 1995). 

We analyze the strategy's performance in a period from immediately prior to the global financial crisis (GFC), up to the present day. Accordingly, our initial backtest period of $-4:0$ is 2004-2008, while our first period of trading, years $0:4$ , is 2008-2012. We compare the performance of our dynamic trading strategy with three other strategies: holding SPY, holding GLD, and a baseline strategy holding an equal split between the two. We use six common validation metrics to evaluate and compare our trading strategy. 

(1) Annualized return (AR): the total return a strategy yields relative to the time the strategy has been in place. 

(2) The overall standard deviation (SD) of the portfolio. 

(3) Sharpe ratio (SR): a common measure of risk-adjusted return. Unfortunately, this penalizes both upside and downside volatility. Some strategies with strong annualized returns may have lower Sharpe ratios due to erratic, yet positive return profiles. 

(4) Maximum drawdown (MD): an alternative penalty function capturing the maximum peak to trough trading loss. 

(5) Sortino ratio (SoR): an alternative measure of risk-adjusted return that only penalizes downside deviation in the denominator. 

(6) Calmar ratio (CR): a measure of risk-adjusted returns that penalizes the maximum realized drawdown over some candidate investment period. 

If our trading strategy were applied among a collection of specific assets rather than an index (for example, a group of components of the index as featured in Table 2), the trading strategy could attain greater expected returns and higher risk-adjusted return ratios. In particular, while an index has a mean-reverting effect with respect to a large collection of stock returns, a smaller collection of carefully selected stocks may provide higher expected returns. Should the group of stocks be greater than $\sim$ 30, unsystematic (stock-specific) risk is diversified away for both the index and the portfolio of stocks. Both portfolios would have a risk component mostly comprised of systematic risk. This could lead to higher risk-adjusted returns for the trading strategy. 


Table 5. Volatility cluster structure of SPY in four year increments


<table><tr><td>Ticker</td><td>No. Segments</td><td>No. Clusters</td><td>Cluster Sizes</td></tr><tr><td>2000-2004</td><td>7</td><td>2</td><td>4,3</td></tr><tr><td>2004-2008</td><td>4</td><td>2</td><td>3,1</td></tr><tr><td>2008-2012</td><td>9</td><td>2</td><td>4,5</td></tr><tr><td>2012-2016</td><td>4</td><td>2</td><td>3,1</td></tr><tr><td>2016-2020</td><td>8</td><td>2</td><td>5,3</td></tr><tr><td>2020-2021</td><td>3</td><td>2</td><td>2,1</td></tr></table>

## 4.1. Model performance: 2008-2020

Implementing our trading strategy between January 2008 and December 2020 would have been highly successful for both risk-averse and risk-on investors. Seen in Table 6 and Figure 5, the strategy consistently outperformed the S&P 500 index and overall generated annualized returns of $9.4\%$ . The S&P 500 returned $7.5\%$ while the static baseline strategy returned $7.6\%$ . The strategy clearly generates alpha by its dynamic nature, automatically detecting market regimes and allocating capital successfully. This entire period can broadly be characterized as a bull market, and yet features several severe market shocks; the strategy's consistent performance demonstrates its robustness to varied market dynamics. Figure 6 shows the positions held by the strategy. 

Of the four strategies compared, our dynamic trading strategy has the best annualized returns, Sharpe ratio, Sortino ratio and Calmar ratio, and lowest drawdown. It has the second-lowest standard deviation of 0.10, close to the baseline static strategy's 0.0087. The most significant component of the Sharpe ratio's performance comes from strong annualized returns; the increased upside volatility is the main contributor to the standard deviation. Indeed, our strategy's Sortino ratio is about $33\%$ greater than that of the S&P 500; this confirms that a significant degree of the penalty in the standard deviation and Sharpe ratio is generated from upside returns. That is, the strong annualized returns of our trading strategy are generated in a relatively volatile manner. This is unsurprising, given that the strategy generates performance due to market timing. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/02c1e573a5fc4e8f11e4fd7a421a95105d7158a084946b5f48ac54b003ed01fc.jpg)



Figure 5. Cumulative returns: January 2008 - December 2020


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/44aac2cb62ccdf7ea2b66551a09b9de5cd3dd2c47cc2fcc27cee58fc96a5f643.jpg)



Figure 6. Positions held by dynamic strategy: January 2008 - December 2020



Table 6. Validation metrics: January 2008 - December 2020


<table><tr><td>Strategy</td><td>AR</td><td>SD</td><td>SR</td><td>MD</td><td>SoR</td><td>CR</td></tr><tr><td>Hold SPY</td><td>0.075</td><td>0.013</td><td>0.45</td><td>0.53</td><td>0.63</td><td>0.14</td></tr><tr><td>Hold GLD</td><td>0.057</td><td>0.011</td><td>0.40</td><td>0.46</td><td>0.56</td><td>0.13</td></tr><tr><td>Baseline</td><td>0.076</td><td>0.0087</td><td>0.33</td><td>0.60</td><td>0.85</td><td>0.23</td></tr><tr><td>Dynamic</td><td>0.094</td><td>0.010</td><td>0.63</td><td>0.27</td><td>0.89</td><td>0.35</td></tr></table>

## 4.2. Detailed analysis of performance over time

In this section, we describe the performance in detail over various time periods, particularly during market crises. While we have reported our findings over one period 2008-2020, in fact, four separate learning and evaluation procedures have been performed. All four periods were successful for our strategy, visible in Figure 5. 

First, the strategy performs well during the GFC. Our strategy generates the second-best returns during the GFC, surpassed only by gold. During the GFC, gold provided extraordinary returns for investors who invested prior to or during the crisis. After incurring a sharp drawdown, our strategy reallocates capital from S&P 500 into gold and consequently outperforms equity markets until late 2011. 

Next, the market experienced significant drawdown in December 2018. Given the brevity of this drawdown, our trading strategy is unable to reallocate capital away from the S&P 500 into gold fast enough to meaningfully reduce the strategy's drawdown. After all, our strategy is predicated on identifying regimes, and allocating capital when new data are identified as similar to past phenomena. It reflects the delicate balance in the look back length $n$ . If it were too long, trading decisions would be made too slowly; if it were too short, trading decisions would be made too frivolously. 

The final significant market crisis during our window of analysis is the market turbulence associated with COVID-19. Our strategy performs well during this period. While it does experience a dip around March of 2020, it does allocate funds away from SPY quite early and avoids a much larger crash. The algorithm then benefits from the V-shaped recovery and reaches its previous peak before switching to GLD for the remainder of the year, leading to a small drawdown. The algorithm reverts back to SPY at the very last trade. We expect the algorithm to benefit from the market rally around the 2021 economic stimulus plan and the possibility of COVID herd immunity through vaccines. 

During the four 4-year windows that make up the 2008-2020 experiment, the optimal look back length n changes as follows. For the four windows, the optimized n is 20, 25, 20, 27 and 20 for 2004-2008, 2008-2012, 2012-2016, 2016-2020, Jan 2020- Dec 2020 respectively. This suggests that continually updating the look back length is important, due to the dynamic nature of markets. The longest look back length is during 2012-2016, a bull market period with the greatest consistency and least volatility in the return profile. This suggests that regimes were more persistent and possibly easier to identify during the 2012-2016 period. 

## 5. Conclusion

This paper demonstrates an original means of clustering volatility regimes, highlighting it as a useful tool for descriptive analysis of financial time series and designing trading strategies. Results on both synthetic and real data are promising, with good validation scores across a range of synthetic data and significant simplification of real time series. The findings support previous work by Hamilton (1989), and Lamoureux and Lastrapes (1990) and many others who contributed to the idea of discrete changes in volatility regimes. Moreover, while previous models generally select the number of regimes in advance, our model applies self-tuning unsupervised learning to determine the number of clusters in its implementation. In real data, we showed that this is usually between 2 and 4, while remaining flexible enough to detect more during crises. Our method integrates well with others in the literature (Guidolin 2011; Ang and Timmermann 2012; Campani, Garcia, and Lewin 2021), as the determined number of volatility regimes can then be used in an alternative regime-switching model that requires this quantity to be set a priori. 

Additionally, the dynamic trading strategy performs well at avoiding periods of significant volatility and drawdown, and performs substantially better than the SPY in various market conditions. Our method continually updates its distributions and parameters, reflecting the need for ongoing learning of market conditions and volatility structure. The method is flexible and also integrates with other statistical and machine learning methods. For instance, one could replace the static safe haven of gold with a learned allocation of low beta assets as a dynamic safe haven. 

The precise methodology and applications described in this paper are not an exhaustive representation of the utility of this method. As long as there is consistency between the regime characteristic of interest, the change point algorithm, and the distance metric between distributions, the method could easily be reworked for classification of regimes of alternative characteristics, and in other domains of study. Future work could make several substitutions or improvements to the methodology. One could apply change point tests that detect changes in the mean to identify clusters with positive or negative returns. Different clustering algorithms could be used to uncover novel structures: for example, DBSCAN (Ester et al. 1996) could be used to find outlier distributions, or fuzzy clustering could be used to find distributions that might belong to multiple clusters. These additional insights could be useful for making new inferences about the time series. Currently, we recompute the distributions every four years; instead, we could make additional use of online change point detection to optimize this period in a more sophisticated manner. We could integrate more sophisticated procedures from metric geometry than simply minimizing the distance to the prior distributions. Finally, we could conceivably combine other machine learning methods of predicting volatility and online decision-making with our distance-based detection of high volatility historical regimes. 

## Acknowledgements

Many thanks to Kerry Chen and Alex Judge for helpful edits and discussion. 

## Disclosure statement

No potential conflict of interest was reported by the authors. 

## Funding

No funding was received for this research. 

## Appendix A. Details of change point detection algorithm

In this section, we describe the change point detection framework. Developed by Hawkins (1977); Hawkins and Zamba (2005), change point algorithms seek to determine breaks in a time series at which the stochastic properties of the underlying random variables change, and have become instrumental in time series analysis. 

## A.1. General change point detection framework

First, we outline the change point detection framework in general. A sequence of observations $x_{1}, x_{2}, ..., x_{n}$ are drawn from random variables $X_{1}, X_{2}, ..., X_{n}$ . We wish to determine points $\tau_{1}, ..., \tau_{m}$ at which the distributions change. One assumes that the underlying random variables are independent and identically distributed between change points. One can summarize this with the following notation, following Ross (2015): 

$$
X _ {i} \sim \left\{ \begin{array}{l} F _ {0} \text {if} i \leq \tau_ {1} \\ F _ {1} \text {if} \tau_ {1} <   i \leq \tau_ {2} \\ F _ {2} \text {if} \tau_ {2} <   i \leq \tau_ {3}, \\ \dots \end{array} \right.
$$

That is, one assumes $X_{i}$ is a random sampling of a different distribution over each time period $[\tau_{i},\tau_{i+1}]$ . In order to meet the apparently restrictive assumption of independence of the data, one must usually perform an appropriate transformation of the data. The log quotient transformation, which yields the log returns from the closing price data, is one such transformation (Gustafsson 2001). 

## A.2. Rank of observations and Mood Test

Ross (2013) points out the fact that log returns often exhibit heavy-tailed behaviour. As a result, a nonparametric test is needed to detect change points that do not a priori assume the distribution of the data. The rank test is one such test. Suppose there are two samples of observations from unknown distributions $A = \{r_{1,1}, r_{1,2}, ..., r_{1,m}\}$ and $B = \{r_{2,1}, r_{2,2}, ..., r_{2,n}\}$ . Define the rank of an observation $r \in A \cup B$ as follows: 

$$
\operatorname{rank} (r) = \sum_ {j} ^ {m} \mathbb {1} _ {\left(r \geq r _ {1, j}\right)} + \sum_ {j} ^ {n} \mathbb {1} _ {\left(r \geq r _ {2, j}\right)} = \# \{s \in A \cup B: r \geq s \}
$$

A larger rank indicates a higher positioning in the ordering of the elements of A and B. If both sets of samples have the same distribution, the median rank among $\{\operatorname{rank}(r): r \in A \cup B\}$ is $\frac{1}{2}(n + m + 1)$ . In this case, one would assume that both sets have a near equal split of the ranks. 

The Mood test determines the extent that each observation's rank differs from the median rank, thereby detecting differences in the distributions' variance. If the samples have different variances, then one set of samples would have more extreme values than the other, which means the ranks would not be even between the two sets. Specifically, the test statistic is as follows: 

$$
M _ {m, n} ^ {\prime} = \sum_ {i = 1} ^ {m} (\mathrm{rank} (r _ {1, i}) - (m + n + 1) / 2) ^ {2}
$$

This is appropriately normalized: 

$$
\begin{array}{c} {N = m + n} \\ {\mu_ {M ^ {\prime}} = \frac {1}{1 2} m (N ^ {2} - 1)} \\ {\sigma_ {M ^ {\prime}} = \frac {1}{1 8 0} m n (N + 1) (N ^ {2} - 4)} \\ {M _ {m n} = \frac {1}{\sigma_ {M ^ {\prime}}} (| M ^ {\prime} - \mu_ {M ^ {\prime}} |)} \end{array}
$$

If $M_{mn}$ is greater than some threshold h, we reject the null hypothesis that the distributions have the same variance, and conclude they have different variances. As depicted in B, the log return time series are tail heavy but strongly mean and median centred. Thus, the Mood test reliably detects changes in the variance without being affected by changes in the median. Compare Sections 4 and 5 of Mood (1954) for this distinction. 

## A.3. CPM algorithm

Ross' CPM algorithm (Ross 2015) works by feeding in one data point at a time. When a change point $\tau$ is detected, the algorithm restarts and proceeds from that point, so it suffices to describe how the algorithm determines its very first change point. 

Suppose $x_{1},...,x_{N}$ is a sequence for which no change point has been detected. For each $m=1,2,...,N$ define n=N-m, mirroring the notation of A.2, and compute the Mood test statistic $M_{m,n}$ . If the maximum among these, $M_{N}=\max_{m+n=N}M_{m,n}$ , exceeds a threshold parameter $h_{N}$ , we declare a change point in the variance has occurred at $\hat{\tau}=\arg\max_{m}M_{m,n}$ . If the maximum such test statistic does not exceed the threshold parameter, feed in the next data point $x_{N+1}$ and continue. If a change point $\hat{\tau}=m$ is detected at time N, there has been a delay of n units in its detection. This delay is necessary for the algorithm to examine data points on each side of the change point. The algorithm then restarts from the change point $\hat{\tau}$ . 

In our implementation of the algorithm, we read in at least 30 values before looking for another change point, so that all stationary periods have length at least 30. We choose our parameters h in order to manage the number of false positives (Type I errors). Given an acceptability threshold $\alpha$ , the following equations specify that this error should remain constant over time: 

$$
\begin{array}{r} P (M _ {1} > h _ {1}) = \alpha \\ P (M _ {t} > h _ {t} | M _ {t - 1} \leq h _ {t - 1}, \dots ., M _ {1} \leq h _ {1}) = \alpha \end{array}
$$

In the event that no change point exists, a false positive will nonetheless be detected at time $1/\alpha$ on average. This quantity is the average run length parameter $ARL_{0}$ that is passed to CPM, which in term calculates the appropriate choice of $h_{t}$ . In this case, $ARL_{0}$ is set to 10,000. 

## Appendix B. Plots

We include detailed results for the experiments in Section 3. Figures B1, B2 and B3 depict the clustered distributions and volatility regimes of the Dow Jones, Nikkei and MSCI indices, respectively. Figures B4, B5 and B6 do so for individual firms Apple, Amazon and Berkshire Hathaway, respectively. Figures B7, B8 and B9 depict popular ETFs XLF, IJS and RYT, respectively. Figures B10, B11 and B12 depict the distributions and regimes for the AUD/USD, GBP/USD and NZD/AUD, respectively. 

We remark that all distribution plots are strongly centred in mean and median about zero. This is an important technical point for the Mood test to work correctly to detect changes in variance, as described in Appendix A. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/f201ff1021bfcef7037a762b6262cca33341ea5e6562407b4b3eae8f2d635232.jpg)



(a) Kernel density estimation plots of Dow Jones distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/27a924a1795d12e0b75c66dcd5af138aa203065941e1f079ab0a5cd6ca359e11.jpg)



(b) Five determined volatility regimes for Dow Jones



Figure B1. Volatility clustering results for Dow Jones


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/a2710d33ecf5a84a2ea066fa2392743c27b0d0504b32c3c453da22caaabc6f2b.jpg)



(a) Kernel density estimation plots of Nikkei distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/5e1a19c54543a1bb0966623d378c62986bad69deaa963300b1b7b32a389612e7.jpg)



(b) Two determined volatility regimes for Nikkei



Figure B2. Volatility clustering results for Nikkei


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/dd7c67fc21d960d4cdb1893c80427884878593595c490337004b3c2a3e2787e1.jpg)



(a) Kernel density estimation plots of MSCI distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/c6c725c2a8617150fa276c2edbdc1db97a023207c1f90d30c91eb1734808263c.jpg)



(b) Four determined volatility regimes for MSCI



Figure B3. Volatility clustering results for MSCI


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/785030a8bc1da5e29aabe3d7d8e5454b9c918b038b7a1e72609c9b3ebc836404.jpg)



(a) Kernel density estimation plots of AAPL distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/e77c80143c2b896d350bca9fcfc6653b680e00db0b2d49bb399066912d3a57e7.jpg)



(b) Three determined volatility regimes for AAPL



Figure B4. Volatility clustering results for AAPL


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/9655604628720cae20fe29c3b41e773c014fd9233942aae524305695005670ed.jpg)



(a) Kernel density estimation plots of AMZN distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/d38375b8ba201d75c1f369c3237bcfea3ab63f45d974dfc466fd7cf403c2e1c3.jpg)



(b) Two determined volatility regimes for AMZN



Figure B5. Volatility clustering results for AMZN


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/81e959be775891f296089044ebeb516d527cfc2b5b42b86c2b965a6d4c00f537.jpg)



(a) Kernel density estimation plots of BRK-A distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/873b5dec745bdc09702560df5c2bca2a6657ad12ab3be6b03358f4f9759413fb.jpg)



(b) Four determined volatility regimes for BRK-A



Figure B6. Volatility clustering results for BRK-A


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/15a9d8dee2a15ef74a32c27f62d025e878847c28b94b26b7fabecc27cc360238.jpg)



(a) Kernel density estimation plots of XLF distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/a1309e44c5ae33c06669b6e9f052643b9148e310d6aa6c26a7b04121bd8e49c8.jpg)



(b) Three determined volatility regimes for XLF



Figure B7. Volatility clustering results for XLF


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/3a6f178008ccdb80c985768363f3d31fa146036ee67fabc2bca2e1050a7b2835.jpg)



(a) Kernel density estimation plots of IJS distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/d3218ffad916e57dce5f2bc36aa846e64952adf463fd1f1b0230af61f54418f7.jpg)



(b) Six determined volatility regimes for IJS



Figure B8. Volatility clustering results for IJS


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/8bb008957ffc201d4031679f0038c6b9a93ee05853dea776f934727c80234ac4.jpg)



(a) Kernel density estimation plots of RYT distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/1e9a2144e0150e6dd89b81d3c33e91c60e69a1ae0ce798823c3976454a068eba.jpg)



(b) Four determined volatility regimes for RYT



Figure B9. Volatility clustering results for RYT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/61fd14a25788fd83aa68ee668aa3d11c2f19e99acc024bb6d719d3b92b367224.jpg)



(a) Kernel density estimation plots of AUD/USD distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/965435280728bb694981d8ba8ecbb18b43d908835a3e7d41a020526a1208c2ab.jpg)



(b) Three determined volatility regimes for AUD/USD



Figure B10. Volatility clustering results for AUD/USD


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/923d5fcd00295ed8b57edd2bea81adb150b2269afc9188f661fa2ae0c3c591e1.jpg)



(a) Kernel density estimation plots of GBP/USD distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/6a84f4414fe76057764b084292872ce78dcc0c39cc52e9cfc2803146639b33e8.jpg)



(b) Four determined volatility regimes for GBP/USD



Figure B11. Volatility clustering results for GBP/USD


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/a205dc6202e9728c5c2b8c4e13a0bb0ceca7d89baf675e868c14c2584554dcf6.jpg)



(a) Kernel density estimation plots of NZD/USD distributions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/533a4c06-2188-492a-b6c4-bc03a225b35a/f9cc747e4eaeeea5a48cc69b349ac5cc8c30a3280c7e9bde26844db7144ef9a5.jpg)



(b) Three determined volatility regimes for NZD/USD



Figure B12. Volatility clustering results for NZD/USD




Ang, Andrew, and Allan Timmermann. 2012. “Regime Changes and Financial Markets.” Annual Review of Financial Economics 4 (1): 313–337. 





Arouri, Mohamed, Christophe Estay, Christophe Rault, and David Roubaud. 2016. “Economic policy uncertainty and stock markets: Long-run evidence from the US.” Finance Research Letters 18: 136–141. 





Baba, N., and Y. Sakurai. 2011. “Predicting regime switches in the VIX index with macroeconomic variables.” Applied Economics Letters 18 (15): 1415–1419. 





Baiardi, Lorenzo Cerboni, Massimo Costabile, Domenico De Giovanni, Fabio Lamantia, Arturo Leccadito, Ivar Massabó, Massimiliano Menzietti, Marco Pirra, Emilio Russo, and Alessandro Staino. 2020. “The Dynamics of the S&P 500 under a Crisis Context: Insights from a Three-Regime Switching Model.” Risks 8 (3): 71. 





Baillie, Richard T., and Claudio Morana. 2009. “Modelling long memory and structural breaks in conditional variances: An adaptive FIGARCH approach.” Journal of Economic Dynamics and Control 33 (8): 1577–1592. 





Balcombe, Kelvin, and Iain Fraser. 2017. “Do bubbles have an explosive signature in Markov switching models?” Economic Modelling 66: 81–100. 





Bollerslev, Tim. 1986. “Generalized autoregressive conditional heteroskedasticity.” Journal of Econometrics 31 (3): 307–327. 





Campani, Carlos Heitor, René Garcia, and Marcelo Lewin. 2021. “Optimal portfolio strategies in the presence of regimes in asset returns.” Journal of Banking & Finance 123: 106030. 





Carstensen, Kai, Markus Heinrich, Magnus Reif, and Maik H. Wolters. 2020. “Predicting ordinary and severe recessions with a three-state Markov-switching dynamic factor model.” International Journal of Forecasting 36 (3): 829–850. 





Cont, Rama, and José da Fonseca. 2002. “Dynamics of implied volatility surfaces.” Quantitative Finance 2 (1): 45–60. 





Dahlhaus, R. 1997. “Fitting time series models to nonstationary processes.” The Annals of Statistics 25 (1): 1–37. 





de Zeeuw, Aart, and Amos Zemel. 2012. “Regime shifts and uncertainty in pollution control.” Journal of Economic Dynamics and Control 36 (7): 939–950. 





del Barrio, Eustasio, Evarist Giné, and Carlos Matrán. 1999. “Central Limit Theorems for the Wasserstein Distance Between the Empirical and the True Distributions.” The Annals of Probability 27 (2): 1009–1071. 





Engle, Robert F. 1982. “Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.” Econometrica 50 (4): 987–1007. 





Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.” In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, KDD’96, 226–231. 





Fowlkes, E. B., and C. L. Mallows. 1983. “A Method for Comparing Two Hierarchical Clusterings.” Journal of the American Statistical Association 78 (383): 553–569. 





Gärtner, Manfred, and Klaus W. Wellershoff. 1995. “Is there an election cycle in American stock returns?” International Review of Economics & Finance 4 (4): 387–410. 





Gonçalves, Sílvia, and Massimo Guidolin. 2006. “Predictable Dynamics in the S&P 500 Index Options Implied Volatility Surface.” The Journal of Business 79 (3): 1591–1635. 





Goutte, Stéphane, Amine Ismail, and Huyên Pham. 2017. “Regime-switching stochastic volatility model: estimation and calibration to VIX options.” Applied Mathematical Finance 24 (1): 38–75. 





Guidolin, Massimo. 2011. “Markov Switching Models in Empirical Finance.” In Missing Data Methods: Time-Series Methods and Applications, 1–86. Emerald Group Publishing Limited. 





Guidolin, Massimo, and Allan Timmermann. 2007. “Asset allocation under multivariate regime switching.” Journal of Economic Dynamics and Control 31 (11): 3503–3544. 





Guillaume, Dominique M., Michel M. Dacorogna, Rakhal R. Davé, Ulrich A. Müller, Richard B. 





Olsen, and Olivier V. Pictet. 1997. “From the bird’s eye to the microscope: A survey of new stylized facts of the intra-daily foreign exchange markets.” Finance and Stochastics 1 (2): 95–129. 





Gustafsson, Fredrik. 2001. Adaptive Filtering and Change Detection. John Wiley & Sons, Ltd. Halkidi, Maria, Yannis Batistakis, and Michalis Vazirgiannis. 2001. “On Clustering Validation Techniques.” Journal of Intelligent Information Systems 17 (2/3): 107–145. 





Hamilton, James D. 1989. “A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.” Econometrica 57 (2): 357–384. 





Hassanat, Ahmad Basheer, Mohammad Ali Abbadi, Ghada Awad Altarawneh, and Ahmad Ali Alhasanat. 2014. “Solving the problem of the K parameter in the KNN classifier using an ensemble learning approach.” arXiv preprint arXiv:1409.0919. 





Hawkins, Douglas M. 1977. “Testing a Sequence of Observations for a Shift in Location.” Journal of the American Statistical Association 72 (357): 180–186. 





Hawkins, Douglas M., and K. D. Zamba. 2005. “A Change-Point Model for a Shift in Variance.” Journal of Quality Technology 37 (1): 21–31. 





James, Nick. 2021. “Dynamics, behaviours, and anomaly persistence in cryptocurrencies and equities surrounding COVID-19.” Physica A: Statistical Mechanics and its Applications 570:125831. 





Kirchler, Michael, and Jürgen Huber. 2007. “Fat tails and volatility clustering in experimental asset markets.” Journal of Economic Dynamics and Control 31 (6): 1844–1874. 





Klaassen, Franc. 2002. “Improving GARCH volatility forecasts with regime-switching GARCH.” Empirical Economics 27 (2): 363–394. 





Kolouri, Soheil, Se Rim Park, Matthew Thorpe, Dejan Slepcev, and Gustavo K. Rohde. 2017. "Optimal Mass Transport: Signal processing and machine-learning applications." IEEE Signal Processing Magazine 34 (4): 43–59. 





Korotayev, A., and Sergey Tsirel. 2010. “A Spectral Analysis of World GDP Dynamics: Kondratieff Waves, Kuznets Swings, Juglar and Kitchin Cycles in Global Economic Development, and the 2008–2009 Economic Crisis.” Structure and Dynamics : e-Journal of Anthropological and Related Sciences 4. 





Lamoureux, Christopher G., and William D. Lastrapes. 1990. “Persistence in Variance, Structural Change, and the GARCH Model.” Journal of Business & Economic Statistics 8 (2): 225–234. 





Lavielle, Marc, and Gilles Teyssière. 2007. “Adaptive Detection of Multiple Change-Points in Asset Price Volatility.” In Long Memory in Economics, 129–156. Springer Berlin Heidelberg. 





Mood, A. M. 1954. “On the Asymptotic Efficiency of Certain Nonparametric Two-Sample Tests.” The Annals of Mathematical Statistics 25 (3): 514–522. 





Nystrup, Peter, Bo William Hansen, Henrik Madsen, and Erik Lindström. 2016. “Detecting change points in VIX and S&P 500: A new approach to dynamic asset allocation.” Journal of Asset Management 17 (5): 361–374. 





Ozaki, T, and Howell Tong. 1975. “On fitting of non-stationary autoregressive models in time series analysis.” In Proceedings of the 8th Hawaii International Conference on System Science, 1055–1062. ACM Press. 





Papanicolaou, Andrew, and Ronnie Sircar. 2013. “A regime-switching Heston model for VIX and S&P 500 implied volatilities.” Quantitative Finance 14 (10): 1811–1827. 





Priestley, M. B. 1965. “Evolutionary Spectra and Non-Stationary Processes.” Journal of the Royal Statistical Society. Series B (Methodological) 27 (2): 204–237. 





Priestley, M. B., and H. Tong. 1973. “On the Analysis of Bivariate Non-Stationary Processes.” Journal of the Royal Statistical Society: Series B (Methodological) 35 (2): 153–166. 





Ross, Gordon J. 2013. “Modelling financial volatility in the presence of abrupt changes.” Physica A: Statistical Mechanics and its Applications 392 (2): 350–360. 





Ross, Gordon J. 2015. “Parametric and Nonparametric Sequential Change Detection in R: The cpm Package.” Journal of Statistical Software, Articles 66 (3): 1–20. 





Shah, Dev, Haruna Isah, and Farhana Zulkernine. 2019. “Stock Market Analysis: A Review and Taxonomy of Prediction Techniques.” International Journal of Financial Studies 7 (2): 26. 





Song, Wonho, Doojin Ryu, and Robert I. Webb. 2016. “Overseas market shocks and VKOSPI dynamics: A Markov-switching approach.” Finance Research Letters 16: 275–282. 





Song, Wonho, Doojin Ryu, and Robert I. Webb. 2018. “Volatility dynamics under an endogenous Markov-switching framework: a cross-market approach.” Quantitative Finance 18 (9): 1559–1571. 





Taylor, James W. 1999. “Evaluating volatility and interval forecasts.” Journal of Forecasting 18 (2): 111–128. 





Taylor, James W. 2005. “Generating Volatility Forecasts from Value at Risk Estimates.” Management Science 51 (5): 712–725. 





Tompkins, Robert G. 2001. “Implied volatility surfaces: uncovering regularities for options on financial futures.” The European Journal of Finance 7 (3): 198–230. 





Tong, H., and K. S. Lim. 1980. “Threshold Autoregression, Limit Cycles and Cyclical Data.” Journal of the Royal Statistical Society: Series B (Methodological) 42 (3): 245–268. 





von Luxburg, Ulrike. 2007. “A tutorial on spectral clustering.” Statistics and Computing 17(4): 395–416. 





Yang, Qing-Qing, Wai-Ki Ching, Jia-Wen Gu, and Tak-Kuen Siu. 2018. “Market-making strategy with asymmetric information and regime-switching.” Journal of Economic Dynamics and Control 90: 408–433. 





Zakoian, Jean-Michel. 1994. “Threshold heteroskedastic models.” Journal of Economic Dynamics and Control 18 (5): 931–955. 





Zelnik-Manor, Lihi, and Pietro Perona. 2004. “Self-Tuning Spectral Clustering.” In Proceedings of the 17th International Conference on Neural Information Processing Systems, NIPS’04, Cambridge, MA, USA, 1601–1608. MIT Press. 

