# A Volume-Price-Adjusted MACD Trading Strategy with Sensitivity Calibration for U.S. Equity Indices

Luyun Lin 

Independent Researcher 

Dallas, TX, USA 

Lixing Lin 

Yale University 

New Haven, CT, USA 

Zhen Zhang 

Independent Researcher 

Jersey City, NJ, USA 

Moxuan Zheng 

Independent Researcher 

Jersey City, NJ, USA 

Yiqing Wang 

Independent Researcher 

Dallas, TX, USA 

## Abstract

Traditional moving average convergence divergence (MACD) trading rules are often constrained by signal lag and susceptibility to false signals. To address these limitations, this study develops a volume-price-adjusted MACD (VP-MACD) framework that incorporates volume, volatility, and intraday price structure into the conventional indicator, and introduces a sensitivity parameter, λ, to allow earlier trade entry and improve responsiveness to market movements. Using the S&P 500, Nasdaq-100, and Dow Jones Industrial Average as representative U.S. equity indices, the model is calibrated over historical records from 2018 to 2022 and evaluated out of sample over 2023 to February 2026. The results indicate that the proposed framework generally delivers better economic performance than the baseline MACD strategy in terms of profitability, risk-adjusted return, and downside-risk control, while generating fewer but more selective trading signals. These findings suggest that incorporating additional market information into technical trading rules may enhance signal quality in U.S. equity index markets. 

Keywords: MACD; volume-price-adjusted MACD; technical analysis; backtesting; sensitivity parameter; U.S. equity market. 

## 1 Introduction

Technical analysis has long been used by both market practitioners and academic researchers as a framework for generating trading signals directly from market data, without relying on firm-level fundamental information (Edwards et al., 2018). Among the indicators developed within this tradition, the Moving Average Convergence Divergence (MACD) remains one of the most widely used because of its simplicity, interpretability, and applicability across asset classes and time horizons (Murphy, 1999). Despite these advantages, the conventional MACD sufers from two well-recognized limitations. First, because it is constructed solely from smoothed exponential moving averages of closing prices, its signals respond to market movements with an inherent lag, often delaying entry and exit decisions relative to actual trend reversals. Second, the indicator does not incorporate trading volume or intraday price information, making it more susceptible to false crossover signals when observed price changes are not supported by suficient market participation or directional strength. 

Prior studies have attempted to address these limitations either by optimizing MACD parameters for specific market (Kang, 2021; Agudelo Aguirre et al., 2020) or by augmenting the signal with additional technical indicators (Wang and Kim, 2018; Chio, 2022). However, parameter optimization is highly sensitive to the choice of calibration window and may lead a risk of in-sample overfitting. Indicator augmentation can improve signal quality, but it often results in a fragmented analytical structure rather than a coherent and unified framework 

This study addresses these gaps by proposing a volume-price-adjusted MACD (VP-MACD) framework embeding volume, volatility, and price-structure information directly into the indicator construction remains comparatively underexplored. The core modification replaces the conventional closing-price input with an adjusted price series that jointly weights volume intensity, intraday volatility, and candlestick body ratio, thereby enriching the informational basis of the indicator without altering itsfundamental architecture. In addition, a sensitivity parameter, λ, is introduced to relax the crossover condition, enabling earlier entry signals and improving responsiveness to nascent trend changes. The framework is calibrated on S&P 500, Nasdaq-100, and Dow Jones Industrial Average data from 2018 to 2022 and evaluated out-of-sample over 2023 to February 2026. 

The main contributions of this study are threefold. First, a unified structural enhancement of MACD is developed that integrates volume, volatility, and intraday price structure into a single adjusted price series, avoiding the fragmentation inherent in multi-indicator combinations. Second, a tunable sensitivity parameter λ is introduced to systematically govern the trade-of between signal selectivity and entry timing. Third, the framework is evaluated across three major U.S. equity indices using a strict out-of-sample design, providing evidence on both performance improvements and their statistical significance. 

The remainder of this paper is organized as follows. Section 2 reviews the relevant literature on MACD and technical trading rules. Section 3 develops the VP-MACD framework and describes the back-testing design. Section 4 presents and discusses the empirical results. Section 5 concludes and discusses the future direction. 

## 2 Literature Review

## 2.1 Market Eficiency and Technical Analysis

The empirical viability of technical trading rules is closely connected to the debate on market eficiency. Under the weak-form Eficient Market Hypothesis (EMH), past price information is already reflected in current prices, leaving no systematic role for technical analysis (Fama, 1970). Behavioral finance ofers a partial counter-argument, attributing recurring price patterns to investor sentiment and feedback dynamics (Shiller, 2003). The Adaptive Markets Hypothesis (AMH) reconciles these views by positing that the profitability of technical rules evolves over time and across market environments (Lo, 2004). This time varying perspective implies that indicators such as MACD may retain predictive value in specific markets or regimes even when eficiency broadly holds. 

## 2.2 Empirical Evidence on MACD Performance

Brock et al. (1992) provide an early benchmark demonstrating that simple technical trading rules can generate returns inconsistent with weak-form eficiency. A broader survey by Park and Irwin (2004) finds that technical rules yield economically significant profits in certain markets, particularly foreign exchange and futures, though results are sensitive to asset class, time period, and methodology. Evidence specific to MACD reveals comparable heterogeneity. Using a cross-country sample of 1,336 companies across 75 countries, Anghel (2015) show that informational eficiency varies substantially, with the United States ranking only 59th, suggesting that even large and liquid equity markets may not fully impound historical price information. This finding is directly relevant to the present study. 

Evidence from individual markets reinforces the conclusion that MACD performance is market-dependent rather than universally stable. Chong and Ng (2008) find that MACDbased strategies outperform passive benchmarks in the London market, whereas Chong et al. (2014) document considerable variation across OECD markets, with no consistent outperformance in the United States or Japan. Emerging-market results are similarly mixed: Chen (2021) reports positive outcomes for the Chinese equity market, while Hejase et al. (2017) find no significant improvement in Lebanon. Gunasekarage and Power (2001) provide additional evidence that moving-average rules can be profitable in South Asian markets. Collectively, these findings suggest that MACD efectiveness is shaped by market-specific characteristics—including liquidity, investor composition, and regime conditions—rather than by a simple developed-versus-emerging distinction. 

Beyond equities, MACD-based strategies have demonstrated adaptability across nontraditional asset classes. Cohen and Qadan (2022) documents robust predictive performance of optimized technical indicators in high-volatility cryptocurrency markets, and Kang (2026) confirm the efectiveness of MACD-based rules in capturing trend reversals in gold markets, though performance remains sensitive to market structure and implementation design. 

## 2.3 Limitations of Traditional MACD and Enhancement

Despite its broad application, the traditional MACD is constrained by two structural weaknesses. First, because the indicator is constructed entirely from exponential moving averages of historical closing prices, it reacts with an inherent delay relative to underlying market dynamics (Murphy, 1999). Wang and Kim (2018) attribute this to structural latency in the smoothing process, which limits the indicator’s responsiveness to rapidly changing conditions. Second, the conventional specification ignores trading volume and intraday price structure. Kang (2021) documents that the standard (12, 26, 9) parameterization frequently generates false signals in choppy markets, and Chen (2021) finds that MACD efectiveness deteriorates during consolidating or trend-reversal periods. These weaknesses help explain the heterogeneity in empirical findings documented above. 

Prior research has pursued two main remediation strategies. The first relies on parameter optimization: Kang (2021) shows that market-specific parameter selection improves performance in the Nikkei 225 futures market, and Agudelo Aguirre et al. (2020) apply machine-learning methods to automate this process. However, market-specific calibration carries a well-recognized risk of in-sample overfitting and data snooping (Anghel, 2015), and optimized parameters tend to have limited transferability across regimes. The second strategy augments MACD with supplementary market information. Wang and Kim (2018) propose a volatility-adjusted MACD, and Chio (2022) show that combining MACD with volume- and volatility-derived signals improves win rate and risk-adjusted performance in U.S. equities. Classical technical analysis has long recognized trading volume as a key confirmation variable for trend validity (Edwards et al., 2018), a view supported by empirical evidence on the price-volume relationship in equity markets (Sanvicente, 2015). Nevertheless, these augmentation approaches typically combine separate indicators rather than integrating additional information into the indicator’s core construction, resulting in fragmented rather than unified frameworks. 

The present study addresses these limitations by embedding volume, intraday volatility, and candlestick price structure directly into the MACD price input, and by introducing a sensitivity parameter λ to govern entry timing. The structural approach, volume-price adjusted MACD (VP-MACD), improve signal robustness and practical transferability without relying primarily on market-specific parameter optimization. 

## 3 Methodology

This section outlines the methodological framework. We first introduce the MACD indicator and its underlying components, followed by the specification of trading rules based on signal line crossovers. We then examine the limitations of the traditional MACD, including signal lag and susceptibility to false signals. Motivated by these limitations, we propose a volume-price-adjusted MACD (VP-MACD) model that integrates volume, price volatility, and structural price information into the conventional framework. Finally, the backtesting design and evaluation metrics are presented to assess the efectiveness of the proposed approach. 

## 3.1 MACD Technical Indicator Definition

The Moving Average (MA) is one of the most fundamental tools in technical analysis. The most basic variation is the Simple Moving Average (SMA), which is calculated by taking the arithmetic mean of an asset’s closing prices over a specific number of periods. It is expressed as: 

$$
\mathrm{SMA} _ {t} = \frac {1}{n} \sum_ {i = 0} ^ {n - 1} P _ {t - i}\tag{1}
$$

where $P _ { t - i }$ denotes the closing price at time $t - i ,$ and n represents the lookback period, defined as the number of trading days in this study. 

Moving averages are essentially trend-following devices designed to identify the beginning or reversal of a market trend by filtering out short-term price noise (Murphy, 1999). In practice, investors frequently employ diferent benchmarks to gauge market sentiment: shortterm trends are often captured by the 5-day MA (MA5), while medium and long-term sentiments are represented by the 20-day (MA20) and 60-day (MA60) averages, respectively. An upward-sloping MA, with price action sustained above the line, typically signifies a dominant buyer’s market. However, the SMA’s reliance on equal weighting for all data points introduces a significant drawback: it reacts slowly to sudden market shifts, a phenomenon commonly referred to as lag. This lag can result in delayed signals, which may be particularly problematic in volatile markets. 

To partially mitigate this, the Exponential Moving Average (EMA) assigns greater weight to more recent price observations. This weighting scheme improves responsiveness to recent market movements while preserving the smoothing function of a moving average. Therefore, the EMA provides a useful foundation for constructing momentum indicators that compare price dynamics across diferent time scales, such as the MACD. The EMA is defined recursively as 

$$
\mathrm{EMA} _ {t} = \alpha P _ {t} + (1 - \alpha) \mathrm{EMA} _ {t - 1}\tag{2}
$$

where α is the smoothing factor, typically defined as $\alpha = 2 / ( n + 1 )$ and n is the lookback period of the exponential moving average, defined as the number of trading days in this study. 

Although EMA improves the responsiveness of traditional moving averages, moving averages themselves primarily describe the direction of price trends and do not inherently measure trend strength, momentum acceleration, or potential exhaustion. Building on the EMA framework, the Moving Average Convergence Divergence (MACD) was originally developed by Gerald Appel in 1979 as a trend-following momentum indicator (Appel, 1979). Since then, MACD has become a cornerstone of technical analysis. The indicator is constructed from the interaction between short-term and long-term exponential moving averages to capture price dynamics across diferent time scales. The calculation of MACD involves three adjustable time-period parameters, typically denoted as $( n _ { f a s t } , n _ { s l o w } , n _ { s i g n a l } )$ . The standard configuration, commonly presented in technical analysis literature, is the (12, 26, 9) setting (Murphy, 1999). This configuration uses a 12-day EMA for short-term momentum, a 26-day EMA for longerterm trend tracking, and a 9-day EMA of the MACD line itself as a smoothing signal. Since MACD relies solely on price time series, it can be applied across a wide range of asset classes, including equities, commodities, options, and cryptocurrencies, as well as across diferent time frequencies such as daily and intraday intervals. 

The MACD is composed of three main components. 

• The MACD line: defined as the diference between short-term and long-term EMAs, oscillates around zero. Its magnitude reflects the degree of separation between the two trends: values near zero suggest weakening momentum, whereas larger deviations from 

zero indicate stronger momentum 

$$
\mathrm{MACD} _ {t} = \mathrm{EMA} _ {1 2} (P _ {t}) - \mathrm{EMA} _ {2 6} (P _ {t})\tag{3}
$$

• the signal line: defined as the exponential moving average of the MACD line, typically computed using a 9-period EMA 

$$
\mathrm{Signal} _ {t} = \mathrm{EMA} _ {9} (\mathrm{MACD} _ {t})\tag{4}
$$

• Histogram: defined as the diference between the MACD line and the signal line, providing a measure of momentum strength and direction. When the histogram is above the zero line but moves toward it, this indicates weakening bullish momentum and a potential trend reversal may occur 

$$
\mathrm{Histogram} _ {t} = \mathrm{MACD} _ {t} - \mathrm{Signal} _ {t}\tag{5}
$$

## 3.2 MACD Trading Rules

Based on the MACD indicator defined in Section 3.1, this section develops the corresponding trading rules. The rules are constructed from the interaction between the MACD line and the signal line, as well as their relationship with the zero line and price movements. 

## MACD signal line crossover strategy

A common trading rule is based on the crossover between the MACD line and the signal line. A buy signal is generated when the MACD line crosses above the signal line, indicating increasing upward momentum. Conversely, a sell signal is triggered when the MACD line crosses below the signal line, suggesting weakening momentum. Trading signals are defined as: 

$$
\text {Signal} = \left\{ \begin{array}{l l} \text {Buy}, & \text {if \mathrm{MACD} _{t - 1} \leq \mathrm{Signal} _{t - 1} and \mathrm{MACD} _{t} > \mathrm{Signal} _{t}} \\ \text {Sell}, & \text {if \mathrm{MACD} _{t - 1} \geq \mathrm{Signal} _{t - 1} and \mathrm{MACD} _{t} <   \mathrm{Signal} _{t}} \end{array} \right.\tag{6}
$$

## MACD zero-line crossover strategy

Another approach considers the position of the MACD relative to the zero line. A buy signal occurs when the MACD crosses above zero, indicating that short-term momentum exceeds long-term momentum. A sell signal occurs when the MACD crosses below zero. 

Trading signals are defined as: 

$$
\text {Signal} = \left\{ \begin{array}{l l} \text {Buy}, & \text {if \mathrm{MACD} _{t - 1} \leq 0 and \mathrm{MACD} _{t} > 0} \\ \text {Sell}, & \text {if \mathrm{MACD} _{t - 1} \geq 0 and \mathrm{MACD} _{t} <   0} \end{array} \right.\tag{7}
$$

## MACD divergence

MACD divergence refers to a mismatch between the direction of price movement and the direction of the MACD line, as defined in Eq. (3). This mismatch is commonly interpreted as a warning that the current price trend may be losing momentum and that a potential reversal may occur. A bullish divergence occurs when prices form lower lows while the MACD line forms higher lows, suggesting that downward momentum is weakening and that a potential buy signal may emerge. Conversely, a bearish divergence occurs when prices form higher highs while the MACD line forms lower highs, indicating that upward momentum may be weakening and that a potential sell signal may emerge. 

In this study, we focus on the signal-line crossover strategy as the foundational trading mechanism. Given its widespread adoption in technical analysis, this approach provides a standardized benchmark for evaluating the empirical performance of the proposed VP-MACD model. To maintain the internal validity of the comparative analysis, other trading rules, including zero-line crossovers and MACD divergence, are excluded from the initial scope. By isolating this specific rule, we can more efectively identify the behavioral characteristics of the indicators before addressing their inherent structural limitations in subsequent sections. 

## 3.3 Limitations of the Traditional MACD

As discussed in Section 2.3, prior studies have identified several limitations of the traditional MACD framework, including signal lag, sensitivity to noisy price movements, and the exclusion of supplementary market information such as volume and intraday price structure. From a methodological perspective, these limitations directly motivate the design of the proposed VP-MACD model. Because both the MACD line and the signal line are derived from smoothed historical price series, the indicator may react to price movements with delay, causing trading signals to appear only after a trend has already begun and thereby reducing entry-timing accuracy. In addition, the traditional MACD is constructed solely from price-based moving averages and does not explicitly incorporate trading volume, even though volume provides useful information about the strength of market participation and can help distinguish stronger price movements from weaker or noisier fluctuations. Furthermore, the traditional 

MACD primarily relies on closing prices and does not capture intraday price structure, such as the relationship between open, high, low, and close prices, which may contain information about intraday momentum, price range, and market pressure that is not reflected in closing prices alone. 

These limitations motivate the development of the proposed VP-MACD model, which embeds volume, volatility, and intraday price structure into the MACD price input. In addition, the λ-adjusted entry rule is introduced to mitigate signal lag and improve entry timing. 

## 3.4 Volume-price-adjusted MACD

Building on these limitations, this study proposes a volume-price-adjusted MACD (VP-MACD), which incorporates volume, price volatility, and structural price information into the conventional MACD framework. Specifically, we define an adjusted price $P _ { t } ^ { * }$ as a volumeweighted adjusted price that integrates both volatility and intraday price movement: 

$$
P _ {t} ^ {*} = \frac {\sum_ {i = t - N} ^ {t - 1} P _ {i} \cdot \mathrm{Volume} _ {i} \cdot \sigma_ {i} \cdot r _ {i}}{\sum_ {i = t - N} ^ {t - 1} \mathrm{Volume} _ {i}}\tag{8}
$$

where t denotes the current trading day, i indexes historical observations within the lookback window, and N represents the number of past trading days included in the rolling calculation. In addition, $P _ { i }$ denotes the closing price on day i, Volume<sub>i</sub> denotes the trading volume on day i, $\sigma _ { i }$ captures the relative volatility component, and $r _ { i }$ measures intraday price structure. 

The volume term serves as the primary weighting mechanism, while the volatility component $\sigma _ { i }$ captures the magnitude of price variation within each period. The relative price measure $r _ { i }$ further provides information on the directional strength of intraday price movement. Together, these components allow the adjusted price $P _ { t } ^ { * }$ to incorporate trading intensity, price variability, and directional price structure into the conventional MACD framework. 

The volatility term is defined as: 

$$
\sigma_ {i} = \frac {\mathrm{STD} \left(P _ {i} ^ {\mathrm{High}} - P _ {i} ^ {\mathrm{Low}}\right)}{P _ {i} ^ {\mathrm{Close}}}\tag{9}
$$

where $P _ { i } ^ { \mathrm { H i g h } } , P _ { i } ^ { \mathrm { L o w } }$ , and $P _ { i } ^ { \mathrm { C l o s e } }$ denote the high, low, and closing prices on day i, respectively. The numerator measures the variation in the daily high–low price range, while dividing by $P _ { i } ^ { \mathrm { C l o s e } }$ normalizes the volatility measure and makes it scale-adjusted across diferent price 

levels. 

and the relative price measure is defined as: 

$$
r _ {i} = \frac {\left| P _ {i} ^ {\mathrm{Close}} - P _ {i} ^ {\mathrm{Open}} \right|}{P _ {i} ^ {\mathrm{High}} - P _ {i} ^ {\mathrm{Low}}}\tag{10}
$$

where $P _ { i } ^ { \mathrm { { O p e n } } }$ denotes the opening price on day $i .$ 

The relative price measure $r _ { i }$ is designed to capture the directional strength of intraday price movement by normalizing the diference between closing and opening prices relative to the total price range. While volatility $\sigma _ { i }$ reflects the magnitude of price fluctuations, it does not distinguish between diferent candlestick structures that may carry distinct market implications. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/0d7073ce9a9921038f6be1f54afde69dd904dffce4a51bbdf416ac6f95e0cc6a.jpg)



Figure 1: Illustration of two candlesticks with the same volatility measure $\left( \sigma _ { i } \right)$ but diferent directional strength $( r _ { i } )$


As illustrated in Fig. 1, two price movements may exhibit identical volatility but represent fundamentally diferent market conditions. In the left candlestick, the large body indicates a strong directional move from open to close, suggesting sustained buying pressure. In contrast, the right candlestick, despite having the same high-low range, exhibits a smaller body with longer wicks, indicating greater intraday uncertainty and weaker directional conviction. 

By incorporating $r _ { i }$ into the adjusted price, the proposed method diferentiates between strong trends and noisy fluctuations, leading to more informative signal generation. This allows the model to jointly account for both the magnitude of price variation (captured by $\sigma _ { i } )$ and the directional intensity of price movement (captured by $r _ { i } )$ 

Using the adjusted price $P _ { t } ^ { * }$ , the VP-MACD indicator is constructed in a manner analogous to the traditional MACD: 

$$
\mathrm{VP-MACD} _ {t} = \mathrm{EMA} _ {1 2} (P _ {t} ^ {*}) - \mathrm{EMA} _ {2 6} (P _ {t} ^ {*})\tag{11}
$$

and the signal line is defined as: 

$$
\mathrm{Signal} _ {t} = \mathrm{EMA} _ {9} (\mathrm{VP-MACD} _ {t})\tag{12}
$$

To further address the lagging issue in signal generation, we introduce a sensitivity parameter λ to modify the entry rule. Instead of strictly relying on the standard crossover condition, the buy signal is adjusted to allow earlier market participation. In particular, the buy signal is triggered when the VP-MACD line exceeds the signal line scaled by a factor λ, where $\lambda \in ( 0 . 8 , 1 )$ 

The interval (0.8, 1) is predefined as a practical calibration range rather than derived from a theoretical restriction. Since $\lambda < 1$ lowers the efective signal-line threshold, it allows the strategy to enter the market before a conventional crossover occurs. At the same time, restricting λ to be greater than 0.8 prevents the entry condition from being excessively relaxed, thereby reducing the risk of premature signals caused by market noise. The resulting λ-adjusted trading rule is defined as: 

$$
\operatorname{Signal} _ {t} ^ {\lambda} = \left\{ \begin{array}{l l} \text {Buy,} & \text {if VP - MACD} _ {t - 1} \leq \lambda \operatorname{Signal} _ {t - 1} \text {and VP - MACD} _ {t} > \lambda \operatorname{Signal} _ {t}, \\ \text {Sell,} & \text {if VP - MACD} _ {t - 1} \geq \operatorname{Signal} _ {t - 1} \text {and VP - MACD} _ {t} <   \operatorname{Signal} _ {t}. \end{array} \right.\tag{13}
$$

## 3.5 Backtesting Framework and Evaluation Metrics

This section outlines the backtesting framework and evaluation methodology used to assess the performance of the proposed VP-MACD strategy in comparison with benchmark strategies. Specifically, we describe the data sources, trading rules, portfolio simulation setup, and evaluation metrics. For consistency throughout the empirical analysis, this study evaluates three strategies: Strategy A, the traditional MACD crossover strategy; Strategy B, the MACD+λ strategy, which applies a λ-adjusted entry rule to the traditional MACD signal; and Strategy C, the proposed VP-MACD strategy, which applies its own separately optimized λ-adjusted entry rule. Multiple performance metrics are then employed to compare these strategies across diferent market conditions. 

## 3.5.1 Data Description

This study uses daily market data from three major U.S. equity indices: the S&P 500 (SPY), the NASDAQ 100 (QQQ), and the Dow Jones Industrial Average (DIA). These indices represent diferent segments of the U.S. equity market, ranging from diversified large-cap stocks, growth-oriented technology firms, and blue-chip companies. 

Structurally, the S&P 500 and NASDAQ 100 employ a market-capitalization-weighted methodology, whereas the Dow Jones Industrial Average is price-weighted, resulting in diferences in index construction and price dynamics. While the NASDAQ 100 is characterized by its high concentration in the technology sector, the S&P 500 provides broader diversification across its 11 sectors. All three indices exhibit high liquidity with substantial trading volume, which ensures reliable pricing and reduces market noise, leading to more stable and realistic backtesting results. Despite some overlap in constituents, the divergent sectoral compositions and volatility characteristics of these indices facilitate a comprehensive evaluation of the proposed strategy under varying market conditions. 

The data are obtained from Yahoo Finance and include five raw market features: Open, High, Low, Close, and trading volume. The sample period spans from 2018 to February 2026. Because the data are reported at the daily trading frequency, weekends and non-trading days are excluded from the sample. To mitigate overfitting, the dataset is divided into a training period from 2018 to 2022 and a testing period from 2023 to February 2026. The training sample contains 1,259 trading-day observations, while the testing sample contains 791 trading-day observations for each instrument. Model parameters are calibrated on the training set and evaluated on the testing set. This split ensures that the evaluation reflects out-of-sample performance. 

## 3.5.2 Backtesting Setup

The backtesting simulation is initialized with a starting capital of $100,000. To maintain simplicity in the baseline evaluation, no leverage is applied, and positions are fully invested upon signal execution with a minimum trading unit of one share. We adopt a long-only trading strategy: a position is opened when a buy signal is generated and closed when a sell signal occurs. After exiting a position, the strategy waits for the next valid buy signal before re-entering the market. Since the MACD indicator is computed based on daily closing prices, trades cannot be executed at the signal generation time. To avoid look-ahead bias, all trades are executed at the opening price of the next trading day following a signal. This ensures that the simulation reflects a realistic execution environment where signals are processed after market close. Transaction costs are incorporated using a one-way cost assumption of 4 bps per trade, consisting of 3 bps in trading cost and 1 bp in slippage. The cost is applied at both entry and exit, resulting in a total round-trip cost of 8 bps per completed trade. 

To evaluate the impact of the parameter λ, we conduct a grid search over the range $\lambda \in [ 0 . 8 , 1 . 0 ]$ at an increment of 0.02. The optimal λ is determined separately for each strategy (MACD+λ and VP-MACD) and for each index using the training dataset. The selected strategy-specific and index-specific λ values are then fixed and applied to the testing dataset, allowing for diferent optimal parameter values across markets and diferent strategies. 

## 3.5.3 Evaluation Metrics

To comprehensively evaluate the performance of the proposed VP-MACD strategy relative to the benchmark models, a set of performance metrics is employed to assess profitability, risk exposure, trading eficiency, and statistical significance from multiple perspectives. These metrics are selected to provide both economic interpretation and practical trading insights. 

First, trading activity is measured by the total number of trades, defined as: 

$$
\mathrm{TotalTrades} = N\tag{14}
$$

where each completed buy–sell pair is counted as one trade. This metric reflects the frequency of trading signals and provides insight into the strategy’s level of market participation. A higher number of trades may indicate more opportunities but can also imply increased exposure to noise and transaction costs (if considered in practice). 

Second, the win ratio evaluates the consistency of the strategy in generating profitable trades: 

$$
\mathrm{WinRatio} = \frac {N _ {\mathrm{win}}}{N}\tag{15}
$$

where $N _ { \mathrm { w i n } }$ denotes the number of profitable trades and N is the total number of trades. A higher win ratio indicates that the strategy produces more consistent trading outcomes, although it does not account for the magnitude of gains and losses. 

Third, the profit and loss (PnL) ratio captures the relationship between average gains and average losses: 

$$
\mathrm{PnLRatio} = \frac {\mathrm{Gains} / N _ {\mathrm{win}}}{\mathrm{Losses} / N _ {\mathrm{Loss}}}\tag{16}
$$

where $N _ { \mathrm { L o s s } }$ represents the number of losing trades. This metric evaluates the risk-reward profile of the strategy, indicating whether gains from profitable trades are suficient to compensate for losses. In particular, a PnL ratio greater than 1 implies that the average gain exceeds the average loss, meaning that profits are suficient to cover losses on a per-trade basis. Even with a moderate or low win ratio, a strategy can remain profitable if the PnL ratio is suficiently high. Importantly, a strategy with a win ratio below 50% can still achieve positive overall profitability, as long as the magnitude of gains from winning trades significantly exceeds the losses from losing trades. 

Fourth, the expectancy metric is used to measure the expected profit per trade by combining both the win ratio and the PnL ratio: 

$$
\mathrm{Expectancy} = \mathrm{WinRatio} \times \overline {{G}} - (1 - \mathrm{WinRatio}) \times \overline {{L}}\tag{17}
$$

where $\overline { G }$ and $\overline { { L } }$ denote average gain and average loss per trade, respectively. This metric reflects the average return per trade and provides a direct measure of the strategy’s profitability. A positive expectancy indicates that the strategy is profitable on average, while a negative value suggests that losses outweigh gains over time. Importantly, expectancy integrates the frequency of winning trades and the magnitude of gains and losses, ofering a more comprehensive assessment than win ratio or PnL ratio alone. 

In addition, the total profit and loss (PnL) is calculated as: 

$$
\mathrm{TotalPnL} = V _ {T} - V _ {0}\tag{18}
$$

where $V _ { T }$ is the final portfolio value and $V _ { 0 }$ is the initial capital. This metric captures the absolute profitability of the strategy over the backtesting period and reflects its overall economic value. 

To evaluate risk-adjusted performance, the Sharpe ratio is employed: 

$$
\mathrm{SharpeRatio} = \frac {\mathbb {E} [ R _ {t} ]}{\sigma (R _ {t})} \times \sqrt {2 5 2}\tag{19}
$$

where $R _ { t }$ denotes daily returns and $\sigma ( R _ { t } )$ is their standard deviation. The Sharpe ratio evaluates how eficiently the strategy generates returns relative to the level of risk undertaken, allowing for comparison across strategies with diferent volatility profiles. 

Furthermore, the maximum drawdown (MDD) is used to quantify downside risk: 

$$
\mathrm{MDD} = \max _ {t} \left(\frac {\mathrm{Peak} _ {t} - \mathrm{Value} _ {t}}{\mathrm{Peak} _ {t}}\right)\tag{20}
$$

where $\mathrm { P e a k } _ { t } = \operatorname* { m a x } _ { s \leq t }$ Value<sub>s</sub> denotes the maximum portfolio value observed up to time t. This metric measures the largest loss from peak to a subsequent trough in portfolio value, reflecting the worst-case loss scenario faced by investors. It is particularly important for assessing the robustness and risk tolerance of the strategy in adverse market conditions. 

Finally, to assess whether the performance improvement of the proposed strategy is statistically significant, a one-sided t-test is conducted on the mean returns across diferent strategy pairs, including comparisons between the traditional MACD, the MACD+λ, and the proposed VP-MACD model. This stepwise comparison enables us to isolate the incremental contribution of the λ-adjustment and the VP enhancement. The hypotheses are defined as: 

$$
H _ {0}: \mu_ {\mathrm{new}} \leq \mu_ {\mathrm{old}} H _ {A}: \mu_ {\mathrm{new}} > \mu_ {\mathrm{old}}\tag{21}
$$

Specifically, three pairwise comparisons are performed: (i) traditional MACD vs. MACD+λ, (ii) MACD+λ vs. VP-MACD, and (iii) traditional MACD vs. VP-MACD. A one-sided test is adopted since the objective is to verify performance enhancement rather than merely detect diferences. Rejection of the null hypothesis indicates that the proposed method delivers statistically superior mean return performance. 

Given that financial return diferences may exhibit heteroskedasticity and serial dependence, HAC t-test with Newey-West adjusted standard errors are further employed. In addition, a blocked circular bootstrap is conducted as a non-parametric robustness check. 

Collectively, these metrics jointly provide a comprehensive evaluation framework that captures trading efectiveness, profitability, risk exposure, and statistical robustness, enabling a thorough comparison between the proposed model and benchmark strategies. 

## 4 Results

## 4.1 Trading Characteristics

To compare the three strategies, the sensitivity parameter λ is optimized separately for the MACD+λ and VP-MACD strategies for each index using the training sample. A grid search is conducted over the predefined range $\lambda \in [ 0 . 8 , 1 . 0 ]$ with an increment of 0.02, yielding eleven candidate values for each strategy-index combination. The optimal λ is selected through a multi-criteria evaluation of six trading performance metrics: win ratio, total profit and loss (PnL), PnL ratio, Sharpe ratio, maximum drawdown, and expected value. Rather than maximizing a single metric, the selection balances return generation, risk-adjusted performance, downside risk, and per-trade profitability. Specifically, preference is given to λ values that improve Sharpe ratio, PnL ratio, and expected value without materially increasing maximum drawdown or relying on unstable changes in total PnL. The full metric profiles across λ values for the MACD+λ and VP-MACD strategies are presented in Figures 2–4 and Figures. 5–7, respectively. 

For the MACD+λ strategy, the performance metrics exhibit notably diferent patterns as λ varies, ultimately leading to distinct optimal λ selections that reflect each index’s underlying market characteristics. For SPY, while a higher λ generally yields greater absolute returns along with higher PnL ratio and expected value, these gains are accompanied by increased volatility and drawdown. Given that the Sharpe ratio reaches a favorable balance around $\lambda { = } ~ 0 . 9 0$ without materially worsening downside risk, this value is selected as optimal. For QQQ, performance metrics exhibit greater instability across the candidate λ values, reflecting the index’s comparatively higher volatility and sensitivity to parameter choice. Although the Sharpe ratio peaks near $\lambda = 0 . 8 4$ , metrics such as expected value and PnL ratio do not improve consistently beyond this point, suggesting that a lower λ enhances responsiveness to short-term price movements without overfitting to noise. Accordingly, $\lambda = 0 . 8 4$ is selected. For DIA, the metric profiles display a clearer and more monotonic improvement as λ increases. The PnL ratio and expected value rise steadily, the Sharpe ratio improves markedly near $\lambda = 0 . 9 2$ , and maximum drawdown remains relatively contained throughout. This pattern indicates that a higher λ strengthens signal quality without disproportionately elevating risk, leading to the selection of $\lambda = 0 . 9 2$ 

For the VP-MACD strategy, a diferent pattern emerges across the three indices compared to the MACD+λ MACD. The incorporation of volume, volatility, and price structure information shifts the optimal λ values, reflecting the changed informational basis of the indicator. For SPY and DIA, the optimal λ values are slightly lower at 0.88 and 0.86, respectively, suggesting that the additional market information allows the indicator to accommodate stronger sensitivity adjustment without introducing excessive noise, thereby generating more responsive signals. In contrast, for QQQ, the optimal λ increases substantially to 0.98, indicating that in a highly volatile market environment, a lower λ risks amplifying noise and destabilizing signals. A higher λ corresponding to a more conservative crossover condition and smoother signal generation, is therefore preferred to prevent the strategy from overreacting to short-term price fluctuations. 

Overall, these results suggest that the optimal λ is jointly determined by market volatility and the informational structure of the indicator. In relatively stable markets, the incorporation of additional information such as volume, volatility, and price structure allows for stronger sensitivity adjustment, whereas more volatile markets require a more conservative, higher λ setting to maintain signal robustness. Table 1 summarizes the optimal λ selections for both the λ−adjusted MACD and VP-MACD strategies across all three indices. 


Table 1: Optimal λ selections for the MACD+λ and VP-MACD strategies across indices.


<table><tr><td>Index</td><td>MACD+λ</td><td>VP-MACD</td><td>Primary Selection Criterion</td></tr><tr><td>SPY</td><td>0.90</td><td>0.88</td><td>Sharpe ratio</td></tr><tr><td>QQQ</td><td>0.84</td><td>0.98</td><td>Sharpe ratio</td></tr><tr><td>DIA</td><td>0.92</td><td>0.86</td><td>Sharpe ratio + Expected value</td></tr></table>

## 4.2 Performance Evaluation

This section evaluates the performance of the three trading strategies using the testing data from January 1, 2023 to February 28, 2026. The training data are used only for parameter calibration, including the selection of the sensitivity parameter λ, whereas the testing data are used for performance evaluation. 

As defined in the backtesting framework in Section 3.5, the three strategies are Strategy A, the baseline MACD crossover strategy; Strategy B, the MACD+λ strategy, which applies an optimized λ-adjusted entry rule to the traditional MACD signal; and Strategy C, the proposed VP-MACD strategy, which applies its own separately optimized λ-adjusted entry rule. For both Strategy B and Strategy C, the optimal λ values are calibrated separately on the training data for each index before being applied to the testing period. 

## 4.2.1 Efect of λ: Strategy B vs. Strategy A

Across all three indices, Strategy B improves the win ratio relative to Strategy A, indicating that the introduction of λ enhances signal quality by filtering out weaker and noisier trading signals. This suggests that λ acts as a smoothing or filtering mechanism that prioritizes stronger trends. 

For SPY and QQQ, Strategy B also improves risk-adjusted performance, as reflected in higher Sharpe ratios and PnL ratios. This indicates that removing low-quality signals leads to more consistent returns. However, this improvement is not uniform across all markets. For DIA, although the win ratio increases, the total PnL decreases and becomes negative. This implies that some profitable trades are filtered out along with noisy signals. In relatively less trending or slower-moving markets like DIA, aggressive filtering may eliminate not only false signals but also smaller yet meaningful opportunities. 

Taken together, λ improves trade quality but introduces a trade-of by reducing trading frequency and potentially missing profitable signals. 

## 4.2.2 Impact of Additional Information: Strategy C vs. Strategy B

Across all three indices, Strategy C improves key risk-control and trade-quality metrics relative to Strategy B. In particular, the expected value increases for all three indices, while the maximum drawdown is reduced in each case. The improvement in expected value is especially notable: it increases by more than 200% for SPY, by approximately 57% for QQQ, and turns from negative to positive for DIA. 

For QQQ, Strategy C further improves both profitability and risk-adjusted performance. The higher Sharpe ratio, improved expected value, and reduced maximum drawdown suggest that incorporating volume, volatility, and price-structure information helps confirm trend strength and filter out false breakouts. In a highly volatile market such as QQQ, this additional information appears to enhance signal reliability and improve overall performance. 

For SPY, Strategy C improves some trade-quality metrics, including the PnL ratio and expected value, but its total PnL does not exceed that of Strategy B. The number of trades decreases sharply from 36 to 6. Although the PnL ratio is substantially higher, the reduced trading frequency limits total profitability. This may be explained by the highly liquid and relatively stable nature of SPY, where price information already captures much of the market trend. In such an environment, adding additional volume, volatility, and price-structure information may remove valid trading opportunities, so the gains in trade quality do not fully translate into higher total profitability. 

For DIA, Strategy C substantially outperforms Strategy B. It not only reverses the negative total PnL observed under Strategy B, but also improves most other performance metrics. This suggests that incorporating volume, volatility, and price-structure information helps recover useful signals that are not captured by the λ adjustment alone. In markets with weaker or less persistent trends, additional contextual information becomes important for distinguishing between noise and genuine trading opportunities. 

In this comparison, the benefit of Strategy C is strongest for QQQ and DIA, while the SPY result is more mixed because the number of VP-MACD trades is very small. These results suggest that VP-MACD improves signal quality and risk control, but its efect on total profitability depends on market conditions, particularly volatility, trend strength, and trading frequency. 


Table 2: Comparison of performance metrics across trading strategies for SPY.


<table><tr><td>Strategy</td><td>Trades</td><td>Win Rate</td><td>Total PnL</td><td>PnL Ratio</td><td>Sharpe</td><td>Max DD</td><td>Expectancy</td></tr><tr><td>Baseline MACD</td><td>40</td><td>50.00%</td><td>$13,612.34</td><td>1.39</td><td>0.47</td><td>-13.78%</td><td>$19.39</td></tr><tr><td>MACD (<eq>\lambda = 0.90</eq>)</td><td>36</td><td>55.56%</td><td>$52,688.94</td><td>2.98</td><td>1.35</td><td>-8.17%</td><td>$121.04</td></tr><tr><td>VP-MACD (<eq>\lambda = 0.88</eq>)</td><td>6</td><td>50.00%</td><td>$27,543.78</td><td>8.61</td><td>0.96</td><td>-6.33%</td><td>$365.72</td></tr></table>


Table 3: Comparison of performance metrics across trading strategies for QQQ.


<table><tr><td>Strategy</td><td>Trades</td><td>Win Rate</td><td>Total PnL</td><td>PnL Ratio</td><td>Sharpe</td><td>Max DD</td><td>Expectancy</td></tr><tr><td>Baseline MACD</td><td>35</td><td>45.71%</td><td>$16,823.12</td><td>1.56</td><td>0.45</td><td>-22.84%</td><td>$17.14</td></tr><tr><td>MACD (<eq>\lambda = 0.84</eq>)</td><td>31</td><td>61.29%</td><td>$63,321.06</td><td>1.89</td><td>1.13</td><td>-11.94%</td><td>$76.83</td></tr><tr><td>VP-MACD (<eq>\lambda = 0.98</eq>)</td><td>21</td><td>80.95%</td><td>$88,324.48</td><td>1.72</td><td>1.51</td><td>-7.41%</td><td>$120.26</td></tr></table>


Table 4: Comparison of performance metrics across trading strategies for DIA.


<table><tr><td>Strategy</td><td>Trades</td><td>Win Rate</td><td>Total PnL</td><td>PnL Ratio</td><td>Sharpe</td><td>Max DD</td><td>Expectancy</td></tr><tr><td>Baseline MACD</td><td>40</td><td>40.00%</td><td>$1,236.96</td><td>1.55</td><td>0.09</td><td>-13.95%</td><td>$2.09</td></tr><tr><td>MACD (<eq>\lambda = 0.92</eq>)</td><td>34</td><td>50.00%</td><td>-$6,862.00</td><td>0.75</td><td>-0.20</td><td>-14.19%</td><td>-$12.55</td></tr><tr><td>VP-MACD (<eq>\lambda = 0.86</eq>)</td><td>13</td><td>69.23%</td><td>$29,592.17</td><td>3.72</td><td>1.08</td><td>-6.20%</td><td>$226.87</td></tr></table>

## 4.2.3 Overall Comparison and Trade-of

Compared to the baseline Strategy A, Strategy C improves all key performance metrics, including win ratio, total PnL, PnL ratio, Sharpe ratio, maximum drawdown, and expected value. From an economic perspective, these improvements are meaningful because they reflect stronger profitability, better risk control, and improved trade-level performance. This demonstrates that integrating the λ adjustment with volume, volatility, and pricerelated information enhances both profitability and risk-adjusted performance relative to the traditional MACD strategy. 

Both Strategy B and Strategy C result in fewer trades compared to Strategy A, reflecting the filtering efect introduced by the λ adjustment and the additional information incorporated in VP-MACD. While this improves signal quality and reduces false positives, it also introduces the risk of missing profitable opportunities. From a market perspective, this highlights a fundamental trade-of between signal precision and market responsiveness. In more volatile markets, stronger filtering appears more efective because additional information helps distinguish meaningful price movements from short-term noise. In relatively stable markets, excessive filtering may remove valid trading opportunities and reduce trading frequency, thereby limiting the overall benefit of the enhanced strategy. 

Collectively, Strategy B improves signal quality by adjusting the responsiveness of MACD through λ, while Strategy C further extends the framework by incorporating additional market information. These findings suggest that the proposed VP-MACD model is most efective when the added information improves signal selection without excessively reducing the strategy’s ability to respond to valid trading opportunities. 

## 4.3 Statistical Significance of Performance Improvements

The performance results in Section 4.2 suggest that the proposed VP-MACD strategy provides economically meaningful improvements in profitability, risk control, and trade-level performance. To further examine whether these improvements are statistically significant, we conduct three complementary tests on the diferences in daily returns between strategies over the testing period from January 1, 2023 to February 28, 2026. Consistent with the backtesting design, Strategy B and Strategy C each use their own optimized λ values, calibrated separately on the training data before being applied to the testing period. 

First, we use one-sided t-tests to examine whether the mean return diference between two strategies is significantly greater than zero. Second, because financial return series may exhibit heteroskedasticity and autocorrelation, we apply Newey–West adjusted t-tests as a robustness check. Third, we conduct circular block bootstrap tests to provide a non-parametric validation of the results without relying heavily on normality assumptions. Specifically, the bootstrap procedure uses 1,000 resamples with a block length of five consecutive daily returns. 

Following the pairwise comparison framework defined in Section 3.5.3, this section evaluates the same three pairwise comparisons: Pair 1 compares Strategy B with Strategy A; Pair 2 compares Strategy C with Strategy A; and Pair 3 compares Strategy C with Strategy B. The hypotheses are defined as in Eq. (21), where $\mu$ represents the mean daily return of each strategy. A positive t-statistic value indicates that the first strategy in each pair outperforms the corresponding benchmark strategy. 


Table 5: One-sided t-test results for pairwise strategy comparisons.


<table><tr><td>Comparison</td><td>SPY (t)</td><td>SPY (p)</td><td>QQQ (t)</td><td>QQQ (p)</td><td>DIA (t)</td><td>DIA (p)</td></tr><tr><td>Pair 1: MACD+λ vs. MACD</td><td>1.7171</td><td>0.0432*</td><td>1.6653</td><td>0.0481*</td><td>-0.6313</td><td>0.0736</td></tr><tr><td>Pair 2: VP-MACD vs. MACD</td><td>0.4971</td><td>0.3096</td><td>1.6586</td><td>0.0488*</td><td>1.2804</td><td>0.1004</td></tr><tr><td>Pair 3: VP-MACD vs. MACD+λ</td><td>-0.8857</td><td>0.8120</td><td>0.3107</td><td>0.3781</td><td>1.7065</td><td>0.0442*</td></tr></table>


Note. <sup>∗</sup> denotes statistical significance at the 5% level $\left( p < 0 . 0 5 \right)$ . 


The standard one-sided t-test results are reported in Table 5. For SPY, Pair 1 is statistically significant $( p \ : = \ : 0 . 0 4 3 2 )$ , indicating that Strategy B, the MACD+λ strategy, improves performance relative to Strategy A, the baseline MACD strategy. For QQQ, both Pair 1 $( p = 0 . 0 4 8 1 )$ and Pair 2 $( p = 0 . 0 4 8 8 )$ are significant, suggesting that both the λ adjustment and the proposed VP-MACD framework improve performance relative to the baseline MACD strategy. For DIA, Pair 3 is significant $( p ~ = ~ 0 . 0 4 4 2 )$ , indicating that Strategy C, the VP-MACD strategy, significantly outperforms Strategy B, the MACD+λ strategy. 


Table 6: Newey–West adjusted t-test results for pairwise strategy comparisons.


<table><tr><td>Comparison</td><td>SPY (t)</td><td>SPY (p)</td><td>QQQ (t)</td><td>QQQ (p)</td><td>DIA (t)</td><td>DIA (p)</td></tr><tr><td>Pair 1: MACD+λ vs. MACD</td><td>1.8983</td><td>0.0288*</td><td>1.6931</td><td>0.0452*</td><td>-0.6593</td><td>0.7452</td></tr><tr><td>Pair 2: VP-MACD vs. MACD</td><td>0.5307</td><td>0.2978</td><td>1.7229</td><td>0.0425*</td><td>1.2964</td><td>0.0974</td></tr><tr><td>Pair 3: VP-MACD vs. MACD+λ</td><td>-0.9922</td><td>0.8394</td><td>0.3487</td><td>0.3637</td><td>1.7894</td><td>0.0368*</td></tr></table>


Note. <sup>∗</sup> denotes statistical significance at the 5% level $\overline { { ( p < 0 . 0 5 ) } }$ 


Table 6 reports the Newey–West adjusted t-test results. The significance pattern remains largely consistent with the standard t-tests. Pair 1 remains significant for SPY $\left( { p = 0 . 0 2 8 8 } \right)$ and QQQ $( p = 0 . 0 4 5 2 )$ , Pair 2 remains significant for QQQ $( p = 0 . 0 4 2 5 )$ , and Pair 3 remains significant for DIA $( p = 0 . 0 3 6 8 )$ . This consistency suggests that the main findings are robust after adjusting for potential heteroskedasticity and autocorrelation in the return series. 

To further validate the results, we also conduct circular block bootstrap tests. Figures 8–10 confirm the key significant findings: Pair 1 is significant for SPY $( p = 0 . 0 2 4 )$ , Pair 2 is significant for QQQ $( p = 0 . 0 3 9 )$ , and Pair 3 is significant for DIA $( p = 0 . 0 2 8 )$ . The bootstrap distributions are shifted to the right of zero in these cases, indicating that the mean return diferences are positive. 

The hypothesis testing results therefore support a two-step interpretation of the strategy improvements. First, the λ adjustment improves the traditional MACD strategy in selected markets, as shown by the significance of Pair 1 for SPY and QQQ. Second, the additional volume, volatility, and price-structure information in VP-MACD provides incremental value beyond the baseline MACD for QQQ and beyond MACD+λ for DIA, as shown by the significance of Pair 2 for QQQ and Pair 3 for DIA. These results suggest that the proposed enhancements improve performance across the three indices in diferent ways, although their efectiveness remains market-dependent. 

## 5 Conclusions

This study proposes a volume-price-adjusted MACD (VP-MACD) framework, together with a sensitivity parameter, λ, to address the lagging behavior and false signals of the traditional MACD model. 

A key component of the proposed framework is the determination of optimal λ values for both the $\mathrm { M A C D } { + } \lambda$ strategy and the VP-MACD strategy. Using the training period from 

2018 to 2022, the optimal λ values are selected separately for each strategy and each index, and are then applied to the testing period from 2023 to February 2026. This strategy and index specific design is motivated by diferences in market structure. SPY is characterized by the highest liquidity and moderate volatility, QQQ by high liquidity and the highest volatility, and DIA by relatively lower liquidity and lower volatility. The selected λ values suggest that markets with diferent volatility and trend characteristics may require diferent levels of signal responsiveness. 

The multi-metric evaluation framework shows that the VP-MACD model generally outperforms the baseline MACD model in terms of profitability, risk-adjusted return, and downside-risk control across the three indices. In particular, expectancy, which measures the average profit or loss per trade and captures the combined efect of win ratio and PnL ratio, emerges as an especially important indicator of long-run trading viability. The results show that expectancy improves substantially under VP-MACD relative to the baseline MACD, increasing by approximately 7 to 109 times across the three indices. These improvements indicate that the proposed framework delivers economically meaningful gains in trade-level performance over the traditional MACD approach. In addition, the comparison between VP-MACD and MACD+λ shows that the added volume, volatility, and price-structure information can provide value beyond the λ adjustment alone, especially for DIA, where VP-MACD reverses the negative performance of MACD+λ and produces stronger overall results. 

The statistical tests further support the economic findings. The one-sided t-tests, Newey–West adjusted t-tests, and bootstrap results provide consistent evidence that the proposed enhancements generate statistically significant improvements in selected markets. Specifically, MACD+λ shows significant improvement for SPY, VP-MACD significantly improves performance for QQQ relative to the baseline MACD, and VP-MACD significantly outperforms MACD+λ for DIA. These findings should therefore be interpreted as evidence that VP-MACD improves signal quality most efectively when the additional information helps identify meaningful opportunities without excessively reducing valid trades. More broadly, this market-dependent result is consistent with recent evidence that volatility dynamics and hedging efectiveness vary across market regimes, suggesting that trading indicators should be calibrated with attention to changing market conditions (Cheng et al., 2026). 

An important implication of the analysis is the trade-of between signal selectivity and trading frequency. The VP-MACD framework generally produces fewer trades than the baseline MACD, suggesting that the added volume, volatility, and price-structure information helps filter weaker or noisier signals. This selectivity can improve trade quality, risk control, and expected value, but it may also reduce the number of realized trades and limit statistical power when the sample of executed trades is small. These findings should therefore be interpreted as evidence that VP-MACD improves signal quality most efectively when the additional information helps identify meaningful opportunities without excessively reducing valid trades. Although the testing period contains approximately 800 daily observations, the number of executed trades can be limited under stricter signal-filtering rules, especially for VP-MACD. Future research may extend the testing horizon, use higher-frequency data, or evaluate the VP-MACD framework across a broader set of markets to obtain more trading observations and further assess the statistical significance and robustness of the proposed strategy. In addition, although this study focuses on the U.S. equity market, the VP-MACD framework could be extended to other equity markets, such as China and Japan, as well as to other asset classes, including foreign currencies, commodities, and cryptocurrencies. 

## Statements and Declarations

Funding. This research received no external funding. 

Competing interests. The authors declare no competing interests. 

AI usage. The authors used Claude Sonnet 4.6 (Anthropic) for assistance with manuscript polishing only. 

## References



Alberto Antonio Agudelo Aguirre, Ricardo Alfredo Rojas Medina, and N´estor Dar´ıo Duque M´endez. Machine learning applied in the stock market through the moving average convergence divergence (macd) indicator. Investment Management and Financial Innovations, 17(4):44–60, 2020. 





Gabriel Dan I Anghel. Stock market eficiency and the macd. evidence from countries around the world. Procedia economics and finance, 32:1414–1431, 2015. 





Gerald Appel. The Moving Average Convergence-Divergence Trading Method. Scientific Investment Systems, 1979. 





William Brock, Josef Lakonishok, and Blake LeBaron. Simple technical trading rules and the stochastic properties of stock returns. The Journal of finance, 47(5):1731–1764, 1992. 





Buyun Chen. The efectiveness test of macd in chinese stock market. In 2021 2nd Asia-Pacific Conference on Image Processing, Electronics and Computers, pages 297–300, 2021. 





Kai Cheng, Xiaoxi Qi, Zhiyuan Cheng, Longying Lai, and Xuan Liu. Regime-dependent volatility dynamics: Evidence from time-series analysis. SSRN Electronic Journal, 2026. 





Pat Tong Chio. A comparative study of the macd-base trading strategies: evidence from the us stock market. arXiv preprint arXiv:2206.12282, 2022. 





Terence Tai-Leung Chong and Wing-Kam Ng. Technical analysis and the london stock exchange: testing the macd and rsi rules using the ft30. Applied Economics Letters, 15 (14):1111–1114, 2008. 





Terence Tai-Leung Chong, Wing-Kam Ng, and Venus Khim-Sen Liew. Revisiting the performance of macd and rsi oscillators. Journal of risk and financial management, 7(1): 1–12, 2014. 





Gil Cohen and Mahmoud Qadan. The complexity of cryptocurrencies algorithmic trading. Mathematics, 10(12):2037, 2022. 





Robert D Edwards, John Magee, and WH Charles Bassetti. Technical analysis of stock trends. CRC press, 2018. 





Eugene F Fama. Eficient capital markets: A review of theory and empirical work. The journal of Finance, 25(2):383–417, 1970. 





Abeyratna Gunasekarage and David M Power. The profitability of moving average trading rules in south asian stock markets. Emerging markets review, 2(1):17–33, 2001. 





Ale J Hejase, Ruba M Srour, Hussin J Hejase, and Joumana Younis. Technical analysis: exploring macd in the lebanese stock market. Journal of Research in Business, Economics and Management, 8(4):1493–1502, 2017. 





Byung-Kook Kang. Improving macd technical analysis by optimizing parameters and modifying trading rules: evidence from the japanese nikkei 225 futures market. Journal of Risk and Financial Management, 14(1):37, 2021. 





Byung-Kook Kang. Optimal and non-optimal macd parameter ranges with stop-loss and takeprofit rules: Evidence from the gold market. Journal of Risk and Financial Management, 19(3):192, 2026. 





Andrew W Lo. The adaptive markets hypothesis: Market eficiency from an evolutionary perspective. Journal of Portfolio Management, Forthcoming, 2004. 





John J Murphy. Technical analysis of the financial markets: A comprehensive guide to trading methods and applications. Penguin, 1999. 





Cheol-Ho Park and Scott H Irwin. The profitability of technical analysis: A review. 2004. 





Antonio Zoratto Sanvicente. The price-trading volume relationship in the brazilian stock market, the impact of stock lending and a role for technical analysis. Brazilian Review of Finance, 13(4):631–649, 2015. 





Robert J Shiller. From eficient markets theory to behavioral finance. Journal of economic perspectives, 17(1):83–104, 2003. 





Jian Wang and Junseok Kim. Predicting stock price trend using macd optimized by historical volatility. Mathematical Problems in Engineering, 2018(1):9280590, 2018. 



## A Additional Figures

## A.1 Lambda Calibration Results

Performance Metrics across Lambda Thresholds 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/8c8afa604e12d411760b0de12f318af1fb90dc378c76a261985fce82b5d01430.jpg)



Figure 2: Performance metrics across diferent λ values for MACD strategy for SPY.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/128cb2b5217ae6dc722fbbdebd1e8cf6294ba64f06c29f27bbcf4e241e916098.jpg)



Figure 3: Performance metrics across diferent λ values for MACD strategy for QQQ.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/7f62a849c01a4c6b40e3a7c5b90f61ac2176b8cdd4a24a9eb74d200f02a819fc.jpg)



Figure 4: Performance metrics across diferent λ values for MACD strategy for DIA.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/7a8e463eaf6940daf4e74b10ca1a3744893a98b56e39884ccdc2af99099aed2b.jpg)



Figure 5: Performance metrics across diferent λ values for the VP-MACD strategy for SPY.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/83ee05d4a8b3931b0a2e37723944f73e7af5c579e430181383603823dcaf6708.jpg)



Figure 6: Performance metrics across diferent λ values for the VP-MACD strategy for QQQ.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/578798a6deedbd1b420c8ee93263329dbf4e11e3572ea3ca5e055b484696e049.jpg)



Figure 7: Performance metrics across diferent λ values for the VP-MACD stragegy for DIA.


## A.2 Bootstrap Distributions


Bootstrap Distribution of Mean Return Differences (MACD+λ - MACD) for SPY


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/3e6e444fb03b82483386cd1319b7c02c86df9689463eca3780a62999df7c7932.jpg)



Figure 8: Bootstrap distribution of mean return diferences between MACD+λ and MACD for SPY.



Bootstrap Distribution of Mean Return Differences (VP-MACD - MACD) for QQQ


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/e70473cf63faf047cf23eaeb9c1c888d5dd374b515ac8d27527b85dbd3721b23.jpg)



Figure 9: Bootstrap distribution of mean return diferences between VP-MACD and MACD for QQQ.


Bootstrap Distribution of Mean Return Differences (VP-MACD vs MACD+λ) for DIA 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-24/9ba79c8a-00c5-4764-969f-7187a00a3841/30f3a3fa5beeec058c9189a4a7bf61df14bd66e079c06b1352597e959cdf5a81.jpg)



Figure 10: Bootstrap distribution of mean return diferences between VP-MACD and MACD+λ for DIA.
