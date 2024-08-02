# helper_da.py

def get_technique_info(technique_name):
    technique_info = {
        "AMPR Analysis": {
            "context": """
            Adaptive Multi-dimensional Pattern Recognition (AMPR) is a technique that combines dimensionality reduction, clustering, and anomaly detection to identify patterns and outliers in multi-dimensional data.
            - n_components: Number of principal components needed to explain 95% of the variance. More components indicate higher data complexity.
            - best_eps: Optimal epsilon value for DBSCAN clustering. Represents the maximum distance between two samples to be considered as in the same neighborhood.
            - best_silhouette_score: Measure of how similar an object is to its own cluster compared to other clusters. Ranges from -1 to 1, with higher values indicating better-defined clusters.
            - n_clusters: Number of distinct groups identified in the data.
            - n_anomalies: Number of data points that significantly deviate from the norm.
            """,
            "guidelines": """
            When interpreting AMPR results:
            1. Assess data complexity: Higher n_components suggest more complex, multi-dimensional data.
            2. Evaluate cluster quality: A higher best_silhouette_score (closer to 1) indicates well-defined, distinct clusters.
            3. Analyze cluster structure: Consider the number of clusters in context with the data domain. Does it align with expected groupings?
            4. Examine anomalies: The proportion and nature of anomalies can reveal insights or potential data quality issues.
            5. Visualize results: Use the provided plots to visually confirm patterns and distributions identified by the algorithm.
            """
        },
        "ETSF Analysis": {
            "context": """
            Enhanced Time Series Forecasting (ETSF) analyzes time-dependent data to identify trends, seasonality, and make predictions.
            - adf_result: Results of the Augmented Dickey-Fuller test, including test statistic, p-value, and critical values. Used to determine if a time series is stationary.
            - mse: Mean Squared Error of the forecast. Lower values indicate better prediction accuracy.
            - rmse: Root Mean Squared Error of the forecast. Provides a measure of the typical size of the forecast errors.
            """,
            "guidelines": """
            When interpreting ETSF results:
            1. Stationarity: Check the ADF test results. A p-value < 0.05 typically indicates a stationary series, which is often easier to forecast.
            2. Forecast accuracy: Lower MSE and RMSE values suggest more accurate predictions. Compare these to the scale of your data.
            3. Trend analysis: Examine the trend plot to understand long-term movements in the data.
            4. Seasonality: The seasonal plot reveals repeating patterns. Consider how these align with known cyclical factors in your domain.
            5. Residuals: The residual plot should ideally show random scatter. Patterns here might indicate missed factors in the model.
            6. Forecast vs Actual: Compare the forecast plot with actual values to assess the model's predictive power and identify any systematic over- or under-predictions.
            """
        },
        "Trend Analysis": {
            "context": """
            Trend Analysis examines data over time to identify consistent upward, downward, or stable patterns. It helps in understanding long-term movements in the data and can be used for forecasting.
            - Start and end values: The first and last data points in the time series.
            - Change: The absolute difference between the end and start values.
            - Percent change: The relative change expressed as a percentage.
            """,
            "guidelines": """
            When interpreting Trend Analysis results:
            1. Direction: Identify whether the overall trend is increasing, decreasing, or stable.
            2. Magnitude: Assess the size of the change over time. Is it significant in the context of the data?
            3. Rate of change: Look for acceleration or deceleration in the trend.
            4. Consistency: Note any periods where the trend deviates from the overall pattern.
            5. Seasonality: Be aware of any regular fluctuations that might obscure the underlying trend.
            6. External factors: Consider known events or changes that might explain trend patterns.
            """
        },
        "Variance Analysis": {
            "context": """
            Variance Analysis examines the spread of data points around the mean. It helps in understanding the variability and consistency of data across different categories or over time.
            - Variance: A measure of variability, calculated as the average squared deviation from the mean.
            - Standard Deviation: The square root of the variance, providing a measure of spread in the same units as the original data.
            """,
            "guidelines": """
            When interpreting Variance Analysis results:
            1. Spread: Higher variance indicates greater spread in the data. Consider if this is expected or problematic.
            2. Consistency: Low variance suggests more consistent or predictable data.
            3. Outliers: High variance might indicate the presence of outliers or extreme values.
            4. Comparison: Compare variances across different categories or time periods to identify areas of higher variability.
            5. Context: Interpret the variance in the context of the data. What's considered high or low variance depends on the specific domain.
            6. Implications: Consider how varying levels of variance might impact decision-making or risk assessment.
            """
        },
        "Regression Analysis": {
            "context": """
            Regression Analysis examines the relationship between variables, typically to predict one variable based on others. It provides insights into correlations and potential causal relationships.
            - R-squared: Indicates the proportion of variance in the dependent variable explained by the independent variable(s).
            - Coefficient: Represents the change in the dependent variable for a one-unit change in the independent variable.
            - Intercept: The predicted value of the dependent variable when all independent variables are zero.
            """,
            "guidelines": """
            When interpreting Regression Analysis results:
            1. Model fit: A higher R-squared (closer to 1) indicates a better fit, but be cautious of overfitting.
            2. Coefficient significance: Check if the p-value for each coefficient is less than 0.05 to determine statistical significance.
            3. Coefficient interpretation: Understand what a positive or negative coefficient means in the context of your data.
            4. Residuals: Examine residual plots to check for patterns that might indicate non-linear relationships or heteroscedasticity.
            5. Multicollinearity: Be aware of potential high correlations between independent variables, which can affect interpretation.
            6. Predictions: Use the model cautiously for predictions, especially for values outside the range of the original data.
            """
        },
        "Stratification Analysis": {
            "context": """
            Stratification Analysis involves dividing data into subgroups (strata) based on specific characteristics and analyzing patterns or differences between these groups.
            - Group means: The average value for each subgroup.
            - Group medians: The middle value for each subgroup when ordered.
            - Group standard deviations: A measure of spread within each subgroup.
            """,
            "guidelines": """
            When interpreting Stratification Analysis results:
            1. Group differences: Look for significant differences in means or medians between groups.
            2. Within-group variation: Compare standard deviations to understand if some groups are more variable than others.
            3. Outliers: Identify any groups with unusually high or low values.
            4. Patterns: Look for trends or patterns across the strata. Are differences consistent or do they vary?
            5. Context: Consider what the stratification variable represents and how it might explain the observed differences.
            6. Sample size: Be cautious when interpreting results from strata with very small sample sizes.
            """
        },
        "Gap Analysis": {
            "context": """
            Gap Analysis identifies the difference between current performance and desired or potential performance. It helps in identifying areas for improvement or optimization.
            - Current value: The present state or performance level.
            - Target value: The desired or benchmark state.
            - Gap: The absolute difference between current and target values.
            - Gap percentage: The relative difference expressed as a percentage.
            """,
            "guidelines": """
            When interpreting Gap Analysis results:
            1. Size of gap: Assess how significant the gap is in the context of your domain.
            2. Priority: Larger gaps might indicate areas needing immediate attention.
            3. Achievability: Consider how realistic it is to close each gap. Some gaps might be more easily addressed than others.
            4. Root causes: Try to understand the reasons behind the gaps. Are they due to resource constraints, process inefficiencies, or external factors?
            5. Trends: If possible, look at how gaps have changed over time. Are they widening or narrowing?
            6. Action planning: Use the results to inform specific strategies for closing the most critical gaps.
            """
        },
        "Duplicate Detection": {
            "context": """
            Duplicate Detection identifies identical or near-identical records in a dataset. It's crucial for data quality management and can reveal issues in data collection or processing.
            - Duplicate count: The number of records that are exact duplicates of other records.
            - Duplicate percentage: The proportion of the dataset that consists of duplicates.
            - Duplicate rows: The actual duplicate records identified.
            """,
            "guidelines": """
            When interpreting Duplicate Detection results:
            1. Prevalence: Assess how common duplicates are in your dataset. Even a small percentage can be significant for large datasets.
            2. Patterns: Look for any patterns in the duplicates. Are they concentrated in certain time periods or categories?
            3. Impact: Consider how duplicates might affect your analysis or business processes. Could they lead to overestimation or double-counting?
            4. Root causes: Try to understand why duplicates are occurring. Are they due to data entry errors, system issues, or intentional repetition?
            5. False positives: Be aware that some apparent duplicates might be legitimate repeated data points.
            6. Action plan: Develop a strategy for handling duplicates, whether through removal, merging, or flagging.
            """
        },
        "Process Mining": {
            "context": """
            Process Mining analyzes event logs to discover, monitor, and improve business processes. It provides insights into how processes are actually performed, which may differ from designed or assumed workflows.
            - Process sequences: The different paths or sequences of activities observed in the data.
            - Frequency of sequences: How often each distinct process path occurs.
            - Average activities per case: The typical number of steps in a process instance.
            """,
            "guidelines": """
            When interpreting Process Mining results:
            1. Common paths: Identify the most frequent process sequences. These represent the typical flow of activities.
            2. Variations: Note less common paths and consider why they occur. Are they exceptions, errors, or alternative valid processes?
            3. Bottlenecks: Look for activities or transitions that seem to slow down the process.
            4. Compliance: Compare the discovered processes with the intended or designed process. Where do they diverge?
            5. Efficiency: Consider the average number of activities per case. Could this be optimized?
            6. Outliers: Pay attention to very long or short process sequences. These might represent best practices or problematic cases.
            """
        },
        "Data Validation Techniques": {
            "context": """
            Data Validation Techniques assess the quality, accuracy, and consistency of data. They help identify issues like missing values, outliers, or data that doesn't meet predefined rules or expectations.
            - Missing values: The count or percentage of empty or null values in each column.
            - Negative values: The count of negative values in numeric columns where negatives might be unexpected.
            - Out-of-range values: The count of values that fall outside expected or typical ranges.
            """,
            "guidelines": """
            When interpreting Data Validation results:
            1. Missing data: Assess the extent of missing data. Is it random or concentrated in specific variables or time periods?
            2. Negative values: For columns where negatives are unexpected, investigate the cause. Are they data entry errors or valid special cases?
            3. Out-of-range values: Determine if these are true outliers, data errors, or indicators of special events or conditions.
            4. Patterns: Look for any patterns in the validation issues. Are they associated with particular data sources or time periods?
            5. Impact: Consider how these data quality issues might affect your analysis or business processes.
            6. Remediation: Develop strategies for addressing each type of issue, whether through data cleaning, imputation, or adjusting collection processes.
            """
        },
        "Risk Scoring Models": {
            "context": """
            Risk Scoring Models assess and quantify potential risks associated with different data points or entities. They typically combine multiple factors to produce a single risk score.
            - Average risk score: The mean risk level across all entities.
            - Median risk score: The middle value when all risk scores are ordered.
            - High-risk threshold: Often defined as a certain percentile (e.g., 90th) of the risk score distribution.
            - High-risk count: The number of entities exceeding the high-risk threshold.
            """,
            "guidelines": """
            When interpreting Risk Scoring Model results:
            1. Distribution: Examine the overall distribution of risk scores. Is it normal, skewed, or multi-modal?
            2. Thresholds: Consider the appropriateness of the high-risk threshold. Does it effectively separate high-risk from low-risk entities?
            3. High-risk proportion: Assess the percentage of entities classified as high-risk. Is this proportion manageable and expected?
            4. Factors: If possible, analyze which factors contribute most to high risk scores.
            5. Validation: Compare risk scores to actual outcomes (if available) to assess the model's predictive power.
            6. Action planning: Use the results to prioritize resources for risk mitigation or further investigation.
            """
        },
        "Fuzzy Matching": {
            "context": """
            Fuzzy Matching identifies similar but not identical text entries. It's useful for detecting potential duplicates, misspellings, or variations in text data.
            - Match pairs: Pairs of text entries that are similar above a certain threshold.
            - Similarity score: A measure of how closely two strings match, often expressed as a percentage.
            - Match counts: The number of fuzzy matches found for each column or category.
            """,
            "guidelines": """
            When interpreting Fuzzy Matching results:
            1. Threshold: Consider the appropriateness of the similarity threshold used. Lower thresholds will find more matches but increase false positives.
            2. Pattern types: Look for common types of variations (e.g., misspellings, abbreviations, word order changes).
            3. Frequency: Assess how common fuzzy matches are. High frequencies might indicate systemic data quality issues.
            4. False positives: Be aware that some matches might be coincidental rather than true variations of the same entity.
            5. Standardization: Use the results to inform data standardization efforts, creating rules for common variations.
            6. Root causes: Consider why variations are occurring. Are they due to data entry inconsistencies, multiple data sources, or legitimate variations?
            """
        },
        "Continuous Auditing Techniques": {
            "context": """
            Continuous Auditing Techniques involve ongoing, automated analyses to identify anomalies, trends, or issues in data. They help maintain data quality and detect potential problems in real-time or near-real-time.
            - Mean and standard deviation: Basic statistical measures for each numeric column.
            - Minimum and maximum values: The range of values observed for each column.
            - Outlier count: The number of data points identified as potential outliers, often based on standard deviation or percentile thresholds.
            """,
            "guidelines": """
            When interpreting Continuous Auditing results:
            1. Trends: Look for changes in basic statistics over time. Are means shifting or standard deviations increasing?
            2. Outliers: Assess the frequency and magnitude of outliers. Are they becoming more or less common?
            3. Range violations: Pay attention to minimum and maximum values that fall outside expected ranges.
            4. Patterns: Look for patterns in when or where anomalies occur. Are they associated with specific time periods, categories, or data sources?
            5. Thresholds: Regularly review and adjust the thresholds used to identify outliers or anomalies.
            6. Responsiveness: Consider how quickly issues are detected and addressed. Is the continuous auditing process enabling timely interventions?
            """
        },       
        "Sensitivity Analysis": {
            "context": """
            Sensitivity Analysis assesses how changes in input variables affect the output of a model or system. It helps identify which variables have the most significant impact on results.
            - Baseline: The initial set of input values and corresponding output.
            - Sensitivity measure: Often expressed as the percentage change in output for a given percentage change in input.
            - Impact ranking: A ranking of input variables based on their influence on the output.
            """,
            "guidelines": """
            When interpreting Sensitivity Analysis results:
            1. Key drivers: Identify which variables have the largest impact on the output. These are potential leverage points for influencing outcomes.
            2. Robustness: Assess how sensitive the model is to small changes. High sensitivity might indicate a less robust model.
            3. Nonlinear effects: Look for variables where small changes produce disproportionately large effects.
            4. Interactions: Consider if the sensitivity to one variable depends on the values of others.
            5. Boundaries: Pay attention to any thresholds where the sensitivity changes dramatically.
            6. Implications: Use the results to focus data collection efforts, prioritize risk management, or simplify models by excluding low-impact variables.
            """
        },
        "Scenario Analysis": {
            "context": """
            Scenario Analysis evaluates potential future outcomes by considering different possible scenarios. It's useful for strategic planning and risk assessment.
            - Baseline scenario: The expected or most likely future state.
            - Alternative scenarios: Often including optimistic and pessimistic projections.
            - Key variables: The main factors that differ between scenarios.
            - Outcome measures: The projected results for each scenario, often including financial or operational metrics.
            """,
            "guidelines": """
            When interpreting Scenario Analysis results:
            1. Range of outcomes: Assess the spread between best and worst-case scenarios. A wide range suggests higher uncertainty.
            2. Likelihood: Consider the relative probability of each scenario occurring.
            3. Key drivers: Identify which factors have the biggest impact on differentiating scenarios.
            4. Robustness: Look for strategies or decisions that perform well across multiple scenarios.
            5. Breakpoints: Identify any scenarios where outcomes change dramatically, indicating potential risks or opportunities.
            6. Preparedness: Use the analysis to develop contingency plans and improve organizational adaptability.
            """
        },
        "Monte Carlo Simulation": {
            "context": """
            Monte Carlo Simulation uses repeated random sampling to obtain numerical results and understand the impact of uncertainty in predictive models.
            - Number of simulations: The number of random scenarios generated.
            - Input distributions: The range and probability distribution for each input variable.
            - Output distribution: The resulting range and probabilities of different outcomes.
            - Confidence intervals: Often reported as percentiles (e.g., 5th and 95th) of the output distribution.
            """,
            "guidelines": """
            When interpreting Monte Carlo Simulation results:
            1. Central tendency: Look at the mean or median of the output distribution as a "most likely" outcome.
            2. Spread: Assess the range between low and high percentiles to understand the level of uncertainty.
            3. Skewness: Check if outcomes are symmetrically distributed or skewed towards high or low values.
            4. Extreme scenarios: Pay attention to the tails of the distribution for best and worst-case scenarios.
            5. Sensitivity: Analyze which input variables contribute most to the variation in outcomes.
            6. Probability of specific outcomes: Use the results to estimate the likelihood of meeting targets or exceeding thresholds.
            """
        },
        "KPI Analysis": {
            "context": """
            Key Performance Indicator (KPI) Analysis involves measuring and evaluating specific metrics that are crucial to organizational success.
            - KPI values: The current values of each tracked metric.
            - Targets: Predefined goals or benchmarks for each KPI.
            - Trends: How KPI values have changed over time.
            - Comparisons: How KPIs compare to industry standards or past performance.
            """,
            "guidelines": """
            When interpreting KPI Analysis results:
            1. Goal achievement: Assess which KPIs are meeting, exceeding, or falling short of targets.
            2. Trends: Look for consistent improvements or declines in KPI values over time.
            3. Correlations: Consider how different KPIs relate to each other. Are improvements in one area linked to changes in others?
            4. Benchmarking: Compare KPI values to industry standards or best practices where available.
            5. Leading vs. lagging: Distinguish between KPIs that predict future performance (leading) and those that measure past results (lagging).
            6. Action planning: Use the analysis to identify areas needing improvement and develop specific strategies to enhance performance.
            """
        },
        
        "Cook's Distance Analysis": {
            "context": """
            Cook's Distance is a measure of the influence of each observation on a linear regression model. It considers both the leverage and residual of each data point.
            - Influential points: Data points with a Cook's Distance greater than 4/n, where n is the number of observations.
            - Threshold: The cut-off value (4/n) used to identify influential points.
            """,
            "guidelines": """
            When interpreting Cook's Distance results:
            1. Identify influential points: Look for observations with Cook's Distance exceeding the threshold.
            2. Assess impact: Consider how the regression results might change if influential points were removed.
            3. Investigate causes: Examine the characteristics of influential points to understand why they have high influence.
            4. Context matters: Not all influential points are errors; some may represent important outliers or edge cases.
            5. Model robustness: If many influential points exist, consider using robust regression techniques.
            6. Iterative process: After addressing influential points, rerun the analysis to see if new points become influential.
            """
        },
        "STL Decomposition Analysis": {
            "context": """
            Seasonal and Trend decomposition using Loess (STL) breaks down a time series into three components: trend, seasonality, and residuals.
            - Trend: The long-term progression of the series.
            - Seasonality: The repeating, periodic component of the series.
            - Residuals: The remaining variation after accounting for trend and seasonality.
            - Trend strength: Measure of how much the trend component contributes to the overall variation.
            - Seasonal strength: Measure of how much the seasonal component contributes to the overall variation.
            """,
            "guidelines": """
            When interpreting STL Decomposition results:
            1. Trend analysis: Examine the trend component for long-term patterns or shifts in the data.
            2. Seasonal patterns: Look for consistent cyclical patterns in the seasonal component.
            3. Residual assessment: Check if residuals appear random; patterns might indicate missed components.
            4. Component strength: Use trend and seasonal strength to understand which components dominate the series.
            5. Anomaly detection: Large residuals may indicate unusual events or outliers.
            6. Forecasting implications: Consider how each component might evolve for future predictions.
            """
        },
        "Hampel Filter Analysis": {
            "context": """
            The Hampel Filter is used for outlier detection and removal in time series data. It uses a moving window to calculate the median and the median absolute deviation (MAD).
            - Outliers: Data points that fall outside a certain number of MADs from the median.
            - Outlier count: The number of data points identified as outliers.
            - Outlier percentage: The proportion of data points identified as outliers.
            """,
            "guidelines": """
            When interpreting Hampel Filter results:
            1. Outlier prevalence: Assess the proportion of data points identified as outliers.
            2. Temporal patterns: Look for clusters of outliers in specific time periods.
            3. Threshold sensitivity: Consider how changing the threshold (number of MADs) affects outlier detection.
            4. Context: Evaluate whether detected outliers represent errors, important events, or natural variability.
            5. Impact on analysis: Consider how including or excluding these outliers might affect further analyses.
            6. Iterative approach: After addressing outliers, consider rerunning the filter to detect any newly revealed outliers.
            """
        },
        "GESD Test Analysis": {
            "context": """
            The Generalized Extreme Studentized Deviate (GESD) test is used to detect one or more outliers in a univariate data set that follows an approximately normal distribution.
            - Outliers: Data points identified as statistically significant outliers.
            - Number of outliers: The count of detected outliers.
            - Critical values: The threshold values used to determine outliers.
            """,
            "guidelines": """
            When interpreting GESD Test results:
            1. Outlier count: Consider the number of outliers detected relative to the dataset size.
            2. Outlier values: Examine the specific values identified as outliers and their deviation from the mean.
            3. Distribution assumption: Remember that GESD assumes approximate normality; verify this assumption.
            4. Iterative nature: The test removes outliers one at a time; consider the order of detection.
            5. Context: Evaluate whether detected outliers represent errors, important events, or natural variability.
            6. Comparison: Consider comparing GESD results with other outlier detection methods for robustness.
            """
        },
        "Dixon's Q Test Analysis": {
            "context": """
            Dixon's Q Test is used to identify outliers in small-sized samples (3 to 30 observations). It's particularly useful when dealing with normally distributed data.
            - Q statistic: The calculated test statistic for each potential outlier.
            - Critical Q value: The threshold for determining if a value is an outlier.
            - Outliers: Data points identified as statistically significant outliers.
            """,
            "guidelines": """
            When interpreting Dixon's Q Test results:
            1. Sample size: Remember that this test is most appropriate for small samples (3 to 30 observations).
            2. Outlier identification: Focus on the values identified as outliers and their position in the dataset.
            3. One-sided test: Dixon's Q typically tests the minimum and maximum values only.
            4. Normality assumption: Keep in mind that the test assumes normally distributed data.
            5. Limitations: Be aware that the test may not detect outliers in the middle of the distribution.
            6. Multiple outliers: The test may be less effective if multiple outliers are present.
            """
        },
        "Peirce's Criterion Analysis": {
            "context": """
            Peirce's Criterion is a method for eliminating outliers from data sets. It's based on probabilistic reasoning and can handle multiple outliers.
            - Rejection threshold: The calculated threshold for determining outliers.
            - Outliers: Data points that exceed the rejection threshold.
            - Number of outliers: The count of detected outliers.
            """,
            "guidelines": """
            When interpreting Peirce's Criterion results:
            1. Multiple outliers: This method can handle multiple outliers, so consider all identified points.
            2. Iterative process: The criterion may be applied iteratively; examine the order of outlier detection.
            3. Comparison with mean: Look at how far the outliers deviate from the mean of the dataset.
            4. Dataset size: Consider the number of outliers in relation to the total number of observations.
            5. Context: Evaluate whether detected outliers represent errors, important events, or natural variability.
            6. Robustness: Compare results with other outlier detection methods to ensure consistency.
            """
        },
        "Thompson Tau Test Analysis": {
            "context": """
            The Thompson Tau test is used to determine whether to keep or discard suspected outliers in a dataset. It's based on the concept of the Student's t-distribution.
            - Tau value: The calculated threshold for determining outliers.
            - Outliers: Data points that exceed the tau value.
            - Number of outliers: The count of detected outliers.
            """,
            "guidelines": """
            When interpreting Thompson Tau Test results:
            1. Sample size consideration: The test's effectiveness can vary with sample size.
            2. Outlier magnitude: Examine how far the outliers deviate from the mean.
            3. Iterative application: The test may be applied repeatedly; consider the order of outlier detection.
            4. Normality assumption: Keep in mind that the test assumes normally distributed data.
            5. Comparison: Consider comparing results with other outlier detection methods.
            6. Context: Evaluate whether detected outliers represent errors, important events, or natural variability.
            """
        },
        "Control Charts Analysis": {
            "context": """
            Control Charts, including CUSUM (Cumulative Sum) and EWMA (Exponentially Weighted Moving Average), are used to monitor process stability and detect shifts in the process mean.
            - CUSUM: Tracks cumulative deviations from the target value.
            - EWMA: Gives more weight to recent observations while still accounting for all historical data.
            - Upper and Lower Control Limits: Thresholds for detecting out-of-control points.
            - Out-of-control points: Observations that exceed the control limits.
            """,
            "guidelines": """
            When interpreting Control Charts results:
            1. Trend identification: Look for persistent upward or downward trends in the charts.
            2. Shift detection: Identify any sudden shifts in the process mean.
            3. Out-of-control points: Examine points that exceed the control limits and investigate their causes.
            4. Pattern recognition: Look for cyclic patterns or other non-random behavior in the charts.
            5. Comparison: Compare CUSUM and EWMA results for a more comprehensive view of the process.
            6. Sensitivity: Consider adjusting control limits or parameters to balance between false alarms and missed detections.
            """
        },
        "KDE Anomaly Detection Analysis": {
            "context": """
            Kernel Density Estimation (KDE) Anomaly Detection uses density estimation to identify data points in low-density regions as potential anomalies.
            - Density estimate: The estimated probability density function of the data.
            - Anomaly threshold: The density value below which points are considered anomalies.
            - Anomalies: Data points with density estimates below the threshold.
            - Anomaly count and percentage: The number and proportion of data points identified as anomalies.
            """,
            "guidelines": """
            When interpreting KDE Anomaly Detection results:
            1. Threshold selection: Consider the impact of the chosen anomaly threshold on results.
            2. Distribution shape: Examine the overall shape of the density estimate for insights into data distribution.
            3. Anomaly clusters: Look for clusters of anomalies that might indicate systematic issues.
            4. Context: Evaluate whether detected anomalies represent errors, important events, or rare but valid cases.
            5. Multivariate consideration: For multivariate data, consider how anomalies relate across different dimensions.
            6. Comparison: Consider comparing results with other anomaly detection methods for robustness.
            """
        },
        "Hotelling's T-squared Analysis": {
            "context": """
            Hotelling's T-squared is a multivariate statistical technique used to detect outliers in multi-dimensional data.
            - T-squared statistic: A measure of the multivariate distance of each observation from the center of the data.
            - Critical value: The threshold for determining multivariate outliers.
            - Outliers: Observations with T-squared values exceeding the critical value.
            - Outlier count and percentage: The number and proportion of data points identified as outliers.
            """,
            "guidelines": """
            When interpreting Hotelling's T-squared results:
            1. Multivariate perspective: Remember that outliers are detected based on their combined behavior across all variables.
            2. Contribution analysis: For detected outliers, examine which variables contribute most to their high T-squared values.
            3. Correlation consideration: Be aware that the test accounts for correlations between variables.
            4. Sample size effect: Consider how the sample size affects the critical value and detection sensitivity.
            5. Visualization: Use paired with visual techniques like scatter plots for better understanding.
            6. Root cause analysis: For detected outliers, investigate potential causes in the context of all variables.
            """
        },
        "Breakdown Point Analysis": {
            "context": """
            Breakdown Point Analysis assesses the robustness of statistical estimators to the presence of outliers.
            - Breakdown point: The proportion of arbitrary outliers an estimator can handle before giving an arbitrarily large result.
            - Mean breakdown point: Always 1/n, where n is the sample size.
            - Median breakdown point: Always 0.5, regardless of sample size.
            - Trimmed mean: Mean calculated after removing a certain percentage of the highest and lowest values.
            """,
            "guidelines": """
            When interpreting Breakdown Point Analysis results:
            1. Estimator comparison: Compare the breakdown points of different estimators (e.g., mean vs. median).
            2. Robustness assessment: Higher breakdown points indicate more robust estimators.
            3. Sample size consideration: Consider how the sample size affects the breakdown point, especially for the mean.
            4. Trimmed mean analysis: Examine how different trimming levels affect the results.
            5. Outlier impact: Use this analysis to understand how sensitive your results might be to extreme values.
            6. Method selection: Use breakdown points to inform the choice of statistical methods for further analysis.
            """
        },
        "Chi-Square Test Analysis": {
            "context": """
            The Chi-Square Test is used to determine if there is a significant difference between the expected and observed frequencies in one or more categories.
            - Chi-square statistic: Measures the overall difference between observed and expected frequencies.
            - p-value: The probability of obtaining test results at least as extreme as the observed results.
            - Degrees of freedom: The number of independent categories minus one.
            - Expected frequencies: The theoretical frequency for each category if the null hypothesis is true.
            """,
            "guidelines": """
            When interpreting Chi-Square Test results:
            1. Significance level: Compare the p-value to the chosen significance level (often 0.05) to determine statistical significance.
            2. Effect size: Consider the magnitude of the chi-square statistic, not just its statistical significance.
            3. Category contributions: Examine which categories contribute most to the chi-square statistic.
            4. Expected vs. Observed: Compare expected and observed frequencies to understand the nature of the differences.
            5. Assumptions check: Ensure the test's assumptions (e.g., independent observations, sufficient expected frequencies) are met.
            6. Context: Interpret results in the context of your research question and practical significance.
            """
        },
        "Simple Thresholding Analysis": {
            "context": """
            Simple Thresholding is a basic method for detecting outliers by identifying data points that fall outside a specified range, often defined using the Interquartile Range (IQR).
            - Lower bound: Typically set at Q1 - 1.5 * IQR, where Q1 is the first quartile.
            - Upper bound: Typically set at Q3 + 1.5 * IQR, where Q3 is the third quartile.
            - Outliers: Data points falling below the lower bound or above the upper bound.
            - Outlier count and percentage: The number and proportion of data points identified as outliers.
            """,
            "guidelines": """
            When interpreting Simple Thresholding results:
            1. Threshold sensitivity: Consider how changing the multiplier (e.g., from 1.5 to 3) affects outlier detection.
            2. Distribution shape: Remember that this method assumes a roughly normal distribution.
            3. Contextual evaluation: Assess whether the identified outliers represent errors, important events, or expected variability.
            4. Comparison across variables: If applied to multiple variables, compare outlier prevalence across them.
            5. Visualization: Use box plots or scatter plots to visualize the outliers in context.
            6. Follow-up analysis: For detected outliers, investigate potential causes and consider their impact on further analyses.
            """
        },
        "Lilliefors Test Analysis": {
            "context": """
            The Lilliefors Test is a statistical test used to assess whether a sample comes from a normally distributed population. It's an adaptation of the Kolmogorov-Smirnov test that allows for the mean and variance of the normal distribution to be estimated from the sample.
            - Test statistic: The maximum difference between the empirical distribution function and the cumulative distribution function of the normal distribution.
            - p-value: The probability of observing a test statistic as extreme as the calculated value under the null hypothesis.
            - Critical values: Threshold values for the test statistic at different significance levels.
            - Null hypothesis: The sample comes from a normally distributed population.
            - Alternative hypothesis: The sample does not come from a normally distributed population.
            """,
            "guidelines": """
            When interpreting Lilliefors Test Analysis results:
            1. p-value interpretation: If the p-value is less than your chosen significance level (e.g., 0.05), reject the null hypothesis and conclude the data is not normally distributed.
            2. Test statistic: Compare the test statistic to the critical values. If it exceeds the critical value, this also suggests rejecting the null hypothesis.
            3. Sample size consideration: Be aware that for very large sample sizes, even small deviations from normality can lead to rejecting the null hypothesis.
            4. Practical significance: Consider whether the deviation from normality, if present, is practically significant for your specific analysis needs.
            5. Visual inspection: Use the test results in conjunction with visual tools like Q-Q plots or histograms for a more comprehensive assessment of normality.
            6. Robustness: If normality is rejected, consider whether your subsequent analyses are robust to departures from normality or if you need to use non-parametric methods.
            """
        },
        "Jarque-Bera Test Analysis": {
            "context": """
            The Jarque-Bera test is used to check if sample data have the skewness and kurtosis matching a normal distribution.
            - Test statistic: A measure combining skewness and kurtosis.
            - p-value: The probability of obtaining test results at least as extreme as the observed results.
            - Skewness: A measure of the asymmetry of the probability distribution.
            - Kurtosis: A measure of the "tailedness" of the probability distribution.
            """,
            "guidelines": """
            When interpreting Jarque-Bera Test results:
            1. Null hypothesis: The null hypothesis is that the data are normally distributed. Reject if p-value is less than significance level (often 0.05).
            2. Skewness: Values close to 0 indicate symmetry. Positive values indicate right skew, negative values indicate left skew.
            3. Kurtosis: For normal distribution, kurtosis is 3. Higher values indicate heavy tails, lower values indicate light tails.
            4. Visual inspection: Use Q-Q plots and histograms to visually confirm the test results.
            5. Sample size consideration: The test is more reliable for larger sample sizes.
            6. Implications: Non-normality might affect the choice of further statistical methods or indicate the need for data transformation.
            """
        },
        "Matrix Profile": {
            "context": """
            Matrix Profile is a technique for time series analysis that can be used for pattern discovery, anomaly detection, and forecasting.
            - Profile: A vector that stores the distance between each subsequence and its nearest neighbor.
            - Index: A vector that stores the index of the nearest neighbor for each subsequence.
            - Window size: The length of the subsequences considered.
            """,
            "guidelines": """
            When interpreting Matrix Profile results:
            1. Motif discovery: Look for low points in the profile, which indicate similar subsequences (motifs).
            2. Discord detection: Look for high points in the profile, which indicate unique subsequences (discords or anomalies).
            3. Seasonality: Repeating patterns in the profile may indicate seasonality in the data.
            4. Window size sensitivity: Consider how changing the window size affects the results.
            5. Multidimensional data: For multivariate time series, consider how patterns align across different dimensions.
            6. Time warping: Be aware that the technique is sensitive to phase shifts and might miss similar but time-warped patterns.
            """
        },
        "Ensemble Anomaly Detection": {
            "context": """
            Ensemble Anomaly Detection combines multiple anomaly detection algorithms to improve overall performance and robustness.
            - Individual models: Different anomaly detection algorithms used in the ensemble (e.g., Isolation Forest, Elliptic Envelope).
            - Anomaly scores: Scores assigned by each model to data points, indicating their likelihood of being anomalies.
            - Ensemble method: The way individual model outputs are combined (e.g., voting, averaging).
            """,
            "guidelines": """
            When interpreting Ensemble Anomaly Detection results:
            1. Consensus: Look for data points consistently flagged as anomalies across multiple models.
            2. Disagreement: Investigate cases where models disagree, as these might be borderline anomalies.
            3. False positives: Consider the trade-off between detecting all anomalies and minimizing false alarms.
            4. Model contribution: Assess which individual models contribute most to the ensemble's performance.
            5. Threshold sensitivity: Analyze how changing the anomaly threshold affects the results.
            6. Context: Always interpret anomalies in the context of the domain knowledge and business implications.
            """
        },
        "Gaussian Mixture Models (GMM)": {
            "context": """
            Gaussian Mixture Models are probabilistic models that assume data points are generated from a mixture of a finite number of Gaussian distributions.
            - Components: The number of Gaussian distributions in the mixture.
            - Means: The center of each Gaussian component.
            - Covariances: The spread and orientation of each Gaussian component.
            - Weights: The relative importance of each component in the mixture.
            """,
            "guidelines": """
            When interpreting Gaussian Mixture Models results:
            1. Number of components: Assess if the chosen number of components adequately represents the data structure.
            2. Component interpretation: Try to assign meaning to each component based on its parameters and the data it represents.
            3. Outliers: Points with low probability under all components may be considered outliers.
            4. Clustering: GMM can be used for soft clustering; examine the probability of each point belonging to each component.
            5. Model comparison: Consider comparing GMMs with different numbers of components using criteria like BIC or AIC.
            6. Visualization: For high-dimensional data, consider projecting the results onto lower dimensions for visualization.
            """
        },
        "Expectation-Maximization Algorithm": {
            "context": """
            The Expectation-Maximization (EM) algorithm is an iterative method to find maximum likelihood estimates of parameters in statistical models with latent variables.
            - Expectation step: Estimates the expected value of the log-likelihood function.
            - Maximization step: Maximizes the expected log-likelihood found in the E step.
            - Convergence: The point at which the parameter estimates stabilize.
            - Log-likelihood: A measure of how well the model fits the data.
            """,
            "guidelines": """
            When interpreting Expectation-Maximization results:
            1. Convergence: Check if the algorithm converged and how many iterations it took.
            2. Log-likelihood: Monitor how the log-likelihood changes across iterations to ensure improvement.
            3. Parameter estimates: Examine the final parameter estimates and their standard errors.
            4. Multiple runs: Consider running the algorithm multiple times with different starting points to ensure global optimum.
            5. Model comparison: Use information criteria (AIC, BIC) to compare different models if applicable.
            6. Sensitivity analysis: Assess how sensitive the results are to changes in initial conditions or model assumptions.
            """
        },
        "Statistical Process Control (SPC) Charts": {
            "context": """
            Statistical Process Control Charts are tools used to determine if a process is in a state of statistical control.
            - Control limits: Upper and lower bounds for acceptable process variation.
            - Center line: The average or target value of the process.
            - Out-of-control points: Data points that fall outside the control limits.
            - Runs: Sequences of points exhibiting non-random patterns.
            """,
            "guidelines": """
            When interpreting SPC Charts:
            1. Out-of-control points: Investigate any points falling outside the control limits for special causes.
            2. Trends: Look for sequences of points consistently increasing or decreasing.
            3. Shifts: Identify any abrupt changes in the process average.
            4. Cycles: Check for repeating patterns that might indicate periodic influences on the process.
            5. Clustering: Be aware of any unusual clustering of points, even if within control limits.
            6. Control limit calculation: Ensure control limits are based on appropriate historical data and recalculated when necessary.
            """
        },
        "Z-Score and Modified Z-Score": {
            "context": """
            Z-Score measures how many standard deviations away a data point is from the mean. Modified Z-Score is more robust to outliers as it uses median instead of mean.
            - Z-Score: (x - mean) / standard deviation
            - Modified Z-Score: 0.6745 * (x - median) / MAD, where MAD is the median absolute deviation
            - Threshold: Often set at ±3 for Z-Score and ±3.5 for Modified Z-Score
            """,
            "guidelines": """
            When interpreting Z-Score and Modified Z-Score results:
            1. Outlier detection: Points beyond the threshold are potential outliers.
            2. Distribution shape: Z-Scores can indicate skewness if many points are on one side of the mean.
            3. Comparison: Check if Modified Z-Score identifies different outliers than standard Z-Score.
            4. Scale interpretation: Remember that Z-Scores represent standard deviations from the mean.
            5. Normality assumption: Standard Z-Score assumes normally distributed data; Modified Z-Score is more robust.
            6. Context: Always interpret scores in the context of the data and domain knowledge.
            """
        },
        "Mahalanobis Distance": {
            "context": """
            Mahalanobis Distance is a multi-dimensional generalization of measuring how many standard deviations away a point is from the mean of a distribution.
            - Distance: A unitless measure of distance from the centroid of the data distribution.
            - Covariance matrix: Captures the variance and correlation structure of the data.
            - Chi-square distribution: Mahalanobis distances follow a chi-square distribution if the data is multivariate normal.
            """,
            "guidelines": """
            When interpreting Mahalanobis Distance results:
            1. Outlier detection: Points with large Mahalanobis distances are potential multivariate outliers.
            2. Threshold: Often set using the chi-square distribution with degrees of freedom equal to the number of variables.
            3. Correlation consideration: Mahalanobis distance accounts for correlations between variables, unlike Euclidean distance.
            4. Normality assumption: Be aware that the interpretation relies on an assumption of multivariate normality.
            5. Scalability: Consider computational limitations for very high-dimensional data.
            6. Visualization: Use scatter plots or pair plots with Mahalanobis distance as color to visualize multivariate outliers.
            """
        },
        "Box-Cox Transformation": {
            "context": """
            The Box-Cox transformation is a family of power transformations used to stabilize variance and make data more normal distribution-like.
            - Lambda (λ): The parameter that defines the specific transformation.
            - Log-likelihood: Used to determine the optimal λ value.
            - Transformed data: The result of applying the Box-Cox transformation to the original data.
            """,
            "guidelines": """
            When interpreting Box-Cox Transformation results:
            1. Optimal lambda: Check the selected λ value and its interpretation (e.g., λ=0 indicates log transformation).
            2. Normality improvement: Compare the distribution of original and transformed data for improved normality.
            3. Variance stabilization: Assess if the variance of the transformed data is more consistent across its range.
            4. Impact on relationships: Consider how the transformation affects relationships with other variables.
            5. Interpretability: Be aware that transformations can make results less interpretable in original units.
            6. Limitations: Remember that Box-Cox only works for positive data; consider alternatives for zero or negative values.
            """
        },
        "Grubbs' Test": {
            "context": """
            Grubbs' Test is used to detect outliers in a univariate dataset that follows an approximately normal distribution.
            - Test statistic: Measures the maximum deviation from the mean in standard deviation units.
            - Critical value: The threshold for the test statistic, based on sample size and significance level.
            - Outliers: Data points identified as statistically significant outliers.
            """,
            "guidelines": """
            When interpreting Grubbs' Test results:
            1. Outlier identification: Check if the test statistic exceeds the critical value, indicating a significant outlier.
            2. One-sided vs. two-sided: Consider whether you're testing for outliers on one or both extremes of the distribution.
            3. Iterative process: The test is typically applied iteratively, removing one outlier at a time.
            4. Normality assumption: Remember that the test assumes approximately normal distribution.
            5. Sample size consideration: Be aware that the test's power can be low for small sample sizes.
            6. Multiple testing: If performed iteratively, consider adjusting for multiple comparisons.
            """
        },
        "Chauvenet's Criterion": {
            "context": """
            Chauvenet's Criterion is a method for detecting outliers based on how far a data point is from the mean, in relation to the standard deviation.
            - Probability threshold: Typically set at 1/(2N), where N is the number of data points.
            - Z-score: The number of standard deviations a point is from the mean.
            - Rejection criterion: Points are rejected if their probability of occurrence is less than the threshold.
            """,
            "guidelines": """
            When interpreting Chauvenet's Criterion results:
            1. Outlier identification: Check which points, if any, are flagged as outliers based on the criterion.
            2. Sample size effect: Be aware that the method's sensitivity changes with sample size.
            3. Normality assumption: Remember that the method assumes normally distributed data.
            4. Iterative application: Consider whether to apply the criterion iteratively or only once.
            5. Comparison with other methods: Compare results with other outlier detection techniques for robustness.
            6. Domain knowledge: Always interpret results in the context of subject-matter expertise.
            """
        },
        "Benford's Law Analysis": {
            "context": """
            Benford's Law describes the frequency distribution of leading digits in many real-life sets of numerical data.
            - Expected frequencies: The probabilities of each leading digit according to Benford's Law.
            - Observed frequencies: The actual frequencies of leading digits in the dataset.
            - Chi-square statistic: A measure of how well the observed frequencies match the expected frequencies.
            """,
            "guidelines": """
            When interpreting Benford's Law Analysis results:
            1. Conformity assessment: Compare the observed frequency distribution to the expected Benford distribution.
            2. Digit analysis: Look for digits that deviate significantly from their expected frequencies.
            3. Chi-square test: Use the chi-square statistic to assess overall conformity to Benford's Law.
            4. Data quality indicator: Significant deviations might indicate data quality issues or manipulation.
            5. Applicability: Consider whether Benford's Law is expected to apply to your specific dataset.
            6. Further investigation: Use results as a screening tool to identify areas for more detailed auditing or analysis.
            """
        },
        "Forensic Accounting Techniques": {
            "context": """
            Forensic Accounting Techniques are methods used to detect financial anomalies, fraud, or irregularities in accounting data.
            - Ratio analysis: Comparison of financial ratios to industry benchmarks or historical data.
            - Trend analysis: Examination of changes in financial metrics over time.
            - Benford's Law: Analysis of the frequency distribution of leading digits in financial data.
            - Anomaly detection: Identification of unusual transactions or patterns.
            """,
            "guidelines": """
            When interpreting Forensic Accounting Techniques results:
            1. Red flags: Identify any significant deviations from expected patterns or benchmarks.
            2. Contextual analysis: Consider results in the context of the business environment and industry norms.
            3. Materiality: Focus on anomalies that are financially significant or indicative of systematic issues.
            4. Corroboration: Look for multiple indicators that support potential findings.
            5. False positives: Be aware that anomalies may have legitimate explanations.
            6. Further investigation: Use results to guide more detailed examination of specific areas or transactions.
            """
            },
        "Network Analysis for Fraud Detection": {
            "context": """
            Network Analysis for Fraud Detection uses graph theory to identify suspicious patterns or relationships in transactional or relational data.
            - Nodes: Entities in the network (e.g., accounts, individuals).
            - Edges: Connections between entities (e.g., transactions, relationships).
            - Centrality measures: Metrics that identify important or influential nodes in the network.
            - Community detection: Identification of closely connected groups within the network.
            """,
            "guidelines": """
            When interpreting Network Analysis for Fraud Detection results:
            1. Unusual connections: Look for unexpected relationships or transaction patterns between entities.
            2. High centrality: Identify nodes with unusually high centrality measures, which might indicate key players in fraudulent activities.
            3. Dense subgraphs: Pay attention to unusually dense connections within subgroups, which could indicate collusion.
            4. Temporal patterns: Consider how network structures or metrics change over time.
            5. Outlier nodes: Investigate nodes with unusual characteristics or connectivity patterns.
            6. Context: Always interpret network metrics and patterns in the context of the specific domain and known fraud schemes.
            """
        },
        "Sequence Alignment and Matching": {
            "context": """
            Sequence Alignment and Matching techniques are used to identify similarities between sequences of data, often applied in bioinformatics but also useful in fraud detection and pattern recognition.
            - Alignment score: A measure of similarity between sequences.
            - Gap penalties: Costs associated with inserting gaps in sequences to improve alignment.
            - Local vs. global alignment: Focusing on regions of high similarity vs. aligning entire sequences.
            - Similarity matrix: A visualization of pairwise similarities between multiple sequences.
            """,
            "guidelines": """
            When interpreting Sequence Alignment and Matching results:
            1. High similarity scores: Identify sequences or subsequences with unusually high alignment scores.
            2. Pattern recognition: Look for recurring patterns or motifs across multiple sequences.
            3. Anomaly detection: Pay attention to sequences that don't align well with any others, as these might be anomalies.
            4. Clustering: Consider how sequences cluster based on their alignment scores.
            5. Time-series application: In financial data, consider how this might reveal similar transaction patterns over time.
            6. Scalability: Be aware of computational limitations when dealing with large numbers of long sequences.
            """
        },
        "Conformal Anomaly Detection": {
            "context": """
            Conformal Anomaly Detection is a method that provides a statistically rigorous way to identify anomalies while controlling the false discovery rate.
            - Nonconformity measure: A score that quantifies how different an observation is from others.
            - P-values: Measures of how likely an observation is under the null hypothesis of it being normal.
            - Significance level: The threshold used to determine anomalies.
            - Prediction intervals: Ranges within which new observations are expected to fall.
            """,
            "guidelines": """
            When interpreting Conformal Anomaly Detection results:
            1. Anomaly identification: Focus on data points with p-values below the chosen significance level.
            2. False discovery rate: Consider the trade-off between detecting anomalies and false positives.
            3. Prediction intervals: Examine how often new data falls outside the predicted intervals.
            4. Model-agnostic nature: Appreciate that this method can be applied on top of various underlying models.
            5. Calibration: Check if the method is well-calibrated (i.e., if it detects the expected proportion of anomalies on normal data).
            6. Comparison: Consider comparing results with other anomaly detection methods for a more robust analysis.
            """
        },
            "Factor Analysis": {
            "context": """
            Factor Analysis is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors.
            - Factors: Unobserved variables that explain the common variance in a set of observed variables.
            - Factor loadings: The correlation between each variable and the factor.
            - Communalities: The proportion of each variable's variance that can be explained by the factors.
            - Eigenvalues: Represent the amount of variance explained by each factor.
            """,
            "guidelines": """
            When interpreting Factor Analysis results:
            1. Factor loadings: Look for high loadings (typically > 0.3 or 0.4) to identify which variables are most strongly associated with each factor.
            2. Communalities: Higher communalities indicate that a larger portion of a variable's variance is explained by the factors.
            3. Variance explained: Consider the cumulative variance explained by the factors to determine how well they represent the original data.
            4. Factor interpretation: Try to give meaningful names to factors based on the variables they strongly correlate with.
            5. Number of factors: Use methods like scree plots or parallel analysis to determine the appropriate number of factors to retain.
            6. Rotation: Consider using factor rotation (e.g., varimax, oblimin) to improve interpretability of the factor structure.
            """
        },
        "Multidimensional Scaling (MDS)": {
            "context": """
            Multidimensional Scaling (MDS) is a technique used to visualize the level of similarity between individual cases of a dataset.
            - Stress: A measure of how well the MDS solution represents the input distances.
            - Configuration: The resulting low-dimensional representation of the data.
            - Distance matrix: Pairwise distances between all points in the original high-dimensional space.
            - Dimensions: The axes of the low-dimensional space (usually 2D or 3D for visualization).
            """,
            "guidelines": """
            When interpreting Multidimensional Scaling results:
            1. Stress value: Lower stress indicates a better fit. Typically, stress < 0.1 is considered excellent, < 0.2 is good.
            2. Clustering: Look for clusters or groups of points that are close together in the MDS plot.
            3. Outliers: Identify points that are far from other points, as these may represent unique or anomalous cases.
            4. Dimensions: Try to interpret what each dimension might represent based on the arrangement of points.
            5. Comparison to original space: Consider how well the low-dimensional representation preserves relationships from the original high-dimensional space.
            6. Robustness: If possible, run MDS multiple times to ensure stability of the configuration.
            """
        },
        "t-Distributed Stochastic Neighbor Embedding (t-SNE)": {
            "context": """
            t-SNE is a machine learning algorithm for dimensionality reduction, particularly well-suited for visualizing high-dimensional datasets.
            - Perplexity: A hyperparameter that balances local and global aspects of the data.
            - Iterations: The number of optimization steps performed.
            - KL divergence: A measure of how well the low-dimensional representation preserves the high-dimensional structure.
            - Embeddings: The resulting low-dimensional representation of the data points.
            """,
            "guidelines": """
            When interpreting t-SNE results:
            1. Clusters: Look for distinct clusters in the t-SNE plot, which may represent different classes or groups in the data.
            2. Outliers: Points that are far from others may represent anomalies or unique cases.
            3. Local structure: t-SNE aims to preserve local structure, so focus on relationships between nearby points.
            4. Global structure: Be cautious about interpreting large-scale structures or distances in t-SNE plots.
            5. Perplexity sensitivity: Consider running t-SNE with different perplexity values to ensure robust results.
            6. Iteration effects: Examine how the plot evolves with more iterations; the final plot should be relatively stable.
            """
        },
        "Conditional Plots": {
            "context": """
            Conditional Plots visualize the relationship between two variables while accounting for the effect of one or more other variables.
            - Facets: Subplots representing different levels or categories of the conditioning variable(s).
            - Main variables: The primary x and y variables being plotted.
            - Conditioning variables: The variables used to split or color the data.
            - Trend lines: Optional lines showing the relationship between x and y within each facet.
            """,
            "guidelines": """
            When interpreting Conditional Plots:
            1. Relationships: Look for patterns in how the relationship between the main variables changes across facets.
            2. Interactions: Identify any interactions between the main variables and the conditioning variables.
            3. Consistency: Check if trends are consistent across facets or if there are notable differences.
            4. Outliers: Look for outliers or unusual patterns within specific facets.
            5. Sample size: Be aware of the sample size in each facet, as smaller samples may lead to less reliable patterns.
            6. Context: Interpret the plots in the context of domain knowledge and expected relationships.
            """
        },
        "Individual Conditional Expectation (ICE) Plots": {
            "context": """
            ICE Plots show how the prediction of a machine learning model changes as a feature varies for individual instances.
            - Feature: The variable being varied along the x-axis.
            - Prediction: The model's output, shown on the y-axis.
            - Individual lines: Each line represents how the prediction changes for a single instance as the feature varies.
            - Centered ICE: A variant where lines are centered to show relative changes.
            """,
            "guidelines": """
            When interpreting ICE Plots:
            1. Monotonicity: Check if the relationship between the feature and prediction is consistently increasing, decreasing, or non-monotonic.
            2. Heterogeneity: Look for instances where lines have different slopes or shapes, indicating heterogeneous effects.
            3. Interactions: Crossing lines may suggest interaction effects with other features.
            4. Range of effects: Observe the range of y-values to understand the magnitude of the feature's impact on predictions.
            5. Clusters: Look for clusters of lines with similar behavior, which may indicate subgroups in the data.
            6. Outliers: Identify any lines with unusual patterns, as these may represent interesting cases or potential errors.
            """
        },
        "Time Series Decomposition": {
            "context": """
            Time Series Decomposition breaks down a time series into its constituent components: trend, seasonality, and residuals.
            - Trend: The long-term progression of the series.
            - Seasonality: Repeating patterns or cycles over fixed periods.
            - Residuals: The remaining variation after accounting for trend and seasonality.
            - Additive vs. Multiplicative: Two common models for how components interact.
            """,
            "guidelines": """
            When interpreting Time Series Decomposition results:
            1. Trend: Analyze the overall direction and rate of change in the trend component.
            2. Seasonality: Identify the pattern and strength of seasonal effects, including their period.
            3. Residuals: Check if residuals appear random; patterns might indicate missed components or additional effects.
            4. Component magnitudes: Compare the relative sizes of trend, seasonal, and residual components.
            5. Model appropriateness: Assess whether an additive or multiplicative model better fits the data.
            6. Anomalies: Look for unusual periods where observed values deviate significantly from the trend and seasonal components.
            """
        },
        "Autocorrelation Plots": {
            "context": """
            Autocorrelation Plots show the correlation between a time series and lagged versions of itself.
            - Lag: The time shift applied to the series.
            - Autocorrelation: The correlation coefficient for each lag.
            - Confidence intervals: Typically shown to indicate statistical significance.
            - Partial Autocorrelation: The autocorrelation that's not explained by previous lags.
            """,
            "guidelines": """
            When interpreting Autocorrelation Plots:
            1. Seasonality: Look for repeating patterns of high autocorrelation at regular intervals.
            2. Trend: A slow decay in autocorrelation often indicates a trend in the data.
            3. Stationarity: Rapid decay to zero suggests stationarity.
            4. Significant lags: Identify which lags have autocorrelations outside the confidence intervals.
            5. Model selection: Use significant lags in partial autocorrelation plots to inform ARIMA model orders.
            6. Differencing: If autocorrelations decay slowly, consider differencing the series.
            """
        },
        "Bayesian Networks": {
            "context": """
            Bayesian Networks are probabilistic graphical models that represent a set of variables and their conditional dependencies.
            - Nodes: Represent variables in the network.
            - Edges: Represent conditional dependencies between variables.
            - Conditional Probability Tables (CPTs): Specify the probability of each variable given its parents.
            - Directed Acyclic Graph (DAG): The structure of the network, showing causal relationships.
            """,
            "guidelines": """
            When interpreting Bayesian Networks:
            1. Causal relationships: Examine the direction of edges to understand potential causal relationships.
            2. Conditional independence: Variables not directly connected are conditionally independent given their parents.
            3. Markov blanket: Identify the Markov blanket of key variables to understand their direct influences and dependents.
            4. Probability queries: Use the network to answer probabilistic queries about variable states.
            5. Network complexity: Consider the trade-off between network complexity and interpretability.
            6. Domain knowledge: Validate the learned structure against domain expertise and known relationships.
            """
        },
        "Isolation Forest": {
            "context": """
            Isolation Forest is an unsupervised learning algorithm for anomaly detection based on the principle that anomalies are rare and different.
            - Isolation trees: Binary trees that partition the data.
            - Path length: The number of splits required to isolate a sample.
            - Anomaly score: Derived from the average path length across multiple trees.
            - Contamination: The proportion of outliers in the dataset.
            """,
            "guidelines": """
            When interpreting Isolation Forest results:
            1. Anomaly scores: Lower scores indicate higher likelihood of being an anomaly.
            2. Threshold selection: Choose an appropriate threshold to classify points as anomalies.
            3. Feature importance: Analyze which features contribute most to anomaly detection.
            4. Visualization: Plot data points colored by anomaly score to identify patterns.
            5. Comparison: Compare Isolation Forest results with other anomaly detection methods.
            6. Context: Interpret detected anomalies in the context of domain knowledge.
            """
        },
        "One-Class SVM": {
            "context": """
            One-Class SVM is an unsupervised algorithm that learns a decision boundary to classify new data as similar or different to the training set.
            - Support vectors: The subset of points that define the decision boundary.
            - Kernel: The function used to transform the input space (e.g., RBF, linear).
            - Nu: A parameter controlling the upper bound on the fraction of training errors and the lower bound on the fraction of support vectors.
            - Decision function: The function learned to separate the data from the origin in feature space.
            """,
            "guidelines": """
            When interpreting One-Class SVM results:
            1. Decision boundary: Visualize the decision boundary in feature space if possible.
            2. Anomaly scores: Examine the distribution of decision function values.
            3. Support vectors: Analyze the characteristics of points selected as support vectors.
            4. Kernel choice: Consider how different kernels affect the decision boundary and results.
            5. Parameter sensitivity: Assess how changing nu and kernel parameters impacts the results.
            6. Comparison: Compare One-Class SVM results with other anomaly detection methods.
            """
        },
        "Local Outlier Factor (LOF)": {
            "context": """
            Local Outlier Factor is an unsupervised anomaly detection method that compares the local density of a point to the densities of its neighbors.
            - K-distance: The distance to the k-th nearest neighbor.
            - Reachability distance: A smoothed version of the distance between two points.
            - Local reachability density (LRD): The inverse of the average reachability distance of a point to its neighbors.
            - LOF score: The ratio of a point's LRD to that of its neighbors.
            """,
            "guidelines": """
            When interpreting Local Outlier Factor results:
            1. LOF scores: Higher scores indicate higher likelihood of being an outlier.
            2. Threshold selection: Choose an appropriate threshold to classify points as anomalies.
            3. Locality: Consider that LOF identifies local outliers relative to their neighborhood.
            4. Density variations: Be aware that LOF can handle datasets with varying densities.
            5. K sensitivity: Assess how changing the number of neighbors affects the results.
            6. Visualization: Plot data points colored by LOF score to identify patterns and potential outliers.
            """
        },
        "Robust Principal Component Analysis (RPCA)": {
            "context": """
            Robust PCA is a variant of PCA that works well in the presence of outliers by decomposing the data matrix into a low-rank component and a sparse component.
            - Low-rank component: Captures the main structure of the data.
            - Sparse component: Captures outliers and corruptions in the data.
            - Principal components: The directions of maximum variance in the low-rank component.
            - Explained variance ratio: The proportion of variance explained by each principal component.
            """,
            "guidelines": """
            When interpreting Robust PCA results:
            1. Low-rank structure: Analyze the patterns in the low-rank component to understand the main data structure.
            2. Outliers: Examine the sparse component to identify potential outliers or corruptions.
            3. Comparison with standard PCA: Compare the results to those of standard PCA to understand the impact of outliers.
            4. Dimensionality reduction: Use the top principal components for dimensionality reduction.
            5. Explained variance: Analyze the cumulative explained variance to determine how many components to retain.
            6. Reconstruction: Consider the quality of data reconstruction using the low-rank component.
            """
        },
        "Bayesian Change Point Detection": {
            "context": """
            Bayesian Change Point Detection is a method for identifying points in a time series where the underlying data generation process changes.
            - Change points: Time points where the statistical properties of the time series change.
            - Posterior probability: The probability of a change point at each time step.
            - Model evidence: A measure of how well the change point model fits the data.
            - Segmentation: The division of the time series into segments between change points.
            """,
            "guidelines": """
            When interpreting Bayesian Change Point Detection results:
            1. Change point locations: Identify the time points with high posterior probability of being change points.
            2. Uncertainty: Consider the uncertainty in change point locations, often represented by probability distributions.
            3. Segment characteristics: Analyze the statistical properties of each segment between change points.
            4. Multiple change points: Be aware that multiple change points may be detected in a single series.
            5. Model comparison: Compare different change point models using model evidence.
            6. Domain context: Interpret detected change points in the context of known events or domain knowledge.
            """
        },
        "Hidden Markov Models (HMMs)": {
            "context": """
            Hidden Markov Models are probabilistic models that assume the system being modeled is a Markov process with unobserved (hidden) states.
            - Hidden states: The unobserved states of the system.
            - Observations: The visible outputs of the system.
            - Transition matrix: Probabilities of transitioning between hidden states.
            - Emission matrix: Probabilities of observing each output given each hidden state.
            - Viterbi algorithm: Used to find the most likely sequence of hidden states.
            """,
            "guidelines": """
            When interpreting Hidden Markov Model results:
            1. State interpretation: Try to assign meaning to each hidden state based on its emission probabilities and transitions.
            2. State sequence: Analyze the predicted sequence of hidden states for patterns or regime changes.
            3. Transition probabilities: Examine which state transitions are most likely and what this implies about the system.
            4. Model fit: Consider how well the model explains the observed data, possibly using likelihood or other fit metrics.
            5. Prediction: Use the model to make predictions about future observations or hidden states.
            6. Comparison: If possible, compare HMM results with other time series or sequential data analysis methods.
            """
        },
        "Dynamic Time Warping": {
            "context": """
            Dynamic Time Warping (DTW) is a technique used to find an optimal alignment between two time-dependent sequences.
            - Warping path: The optimal alignment between two sequences.
            - DTW distance: A measure of similarity between two sequences, allowing for stretching and compression.
            - Warping window: A constraint on how far the warping path can deviate from the diagonal.
            - Similarity matrix: A matrix of distances between each pair of points in the two sequences.
            """,
            "guidelines": """
            When interpreting Dynamic Time Warping results:
            1. Distance interpretation: Lower DTW distances indicate more similar sequences.
            2. Alignment visualization: Examine the warping path to understand how the sequences are aligned.
            3. Warping analysis: Look for areas of significant stretching or compression in the alignment.
            4. Comparison: Use DTW to compare multiple sequences and identify the most similar pairs.
            5. Pattern recognition: Use DTW distances to cluster or classify time series data.
            6. Robustness: Consider how DTW results compare to simpler measures like Euclidean distance, especially for sequences of different lengths.
            """
        },
        "Parallel Coordinates Plot": {
            "context": """
            Parallel Coordinates Plot is a way of visualizing high-dimensional data, where each dimension corresponds to a vertical axis and each data point is represented as a line connecting its values on each axis.
            - Axes: Vertical lines representing different variables or dimensions.
            - Lines: Each line represents a single data point across all dimensions.
            - Axis order: The order of axes can significantly affect the patterns visible in the plot.
            - Scaling: Different scaling methods can be applied to each axis for better comparison.
            """,
            "guidelines": """
            When interpreting Parallel Coordinates Plots:
            1. Correlation: Look for parallel or anti-parallel line segments between adjacent axes, indicating correlation.
            2. Clusters: Identify bundles of lines that follow similar paths, suggesting groups of similar data points.
            3. Outliers: Look for lines that deviate significantly from the general pattern.
            4. Dimensional relationships: Examine how values in one dimension relate to values in others across the dataset.
            5. Range and distribution: Observe the range and density of values on each axis.
            6. Axis reordering: Consider how changing the order of axes might reveal different patterns or relationships.
            """
        },
        "Andrews Curves": {
            "context": """
            Andrews Curves represent multivariate data as a set of curves, where each data point is transformed into a finite Fourier series and plotted as a function.
            - Curves: Each curve represents a single multivariate data point.
            - X-axis: Typically represents t, varying from -π to π.
            - Y-axis: The value of the function for each data point at each t.
            - Coefficients: The values of the variables determine the coefficients in the Fourier series.
            """,
            "guidelines": """
            When interpreting Andrews Curves:
            1. Clustering: Look for groups of curves that follow similar paths, indicating clusters in the data.
            2. Outliers: Identify curves that deviate significantly from the main bundles.
            3. Symmetry: Observe any symmetry or lack thereof in the curves, which can indicate relationships between variables.
            4. Crossover points: Pay attention to where curves tend to intersect, as these can be informative.
            5. Amplitude: The overall amplitude of curves can indicate the magnitude of the data points' values.
            6. Variable importance: Consider how changing the order of variables in the function affects the plot, as earlier terms have more influence.
            """
        },
        "Radar Charts": {
            "context": """
            Radar Charts (also known as Spider Charts or Star Charts) display multivariate data on axes starting from the same point, with the data values determining the distance from this central point.
            - Axes: Radial lines representing different variables, usually starting from a central point.
            - Data points: Typically represented as a polygon formed by connecting the values on each axis.
            - Scale: The scale of each axis can be independent or normalized.
            - Area: The area of the polygon can be used as a summary statistic.
            """,
            "guidelines": """
            When interpreting Radar Charts:
            1. Shape analysis: Look at the overall shape of the polygon to quickly assess the profile of a data point.
            2. Balance: Consider how evenly distributed the values are across all axes.
            3. Comparison: When multiple data points are plotted, compare the shapes to identify similarities and differences.
            4. Outliers: Look for axes where a data point extends much further or falls much shorter than on other axes.
            5. Scaling: Be aware of how the scaling of axes affects the visual representation.
            6. Limitations: Remember that the area can be misleading, especially when the order of axes is changed.
            """
        },
        "Sankey Diagrams": {
            "context": """
            Sankey Diagrams are flow diagrams where the width of the arrows is proportional to the flow quantity.
            - Nodes: Represent stages or categories in a process or system.
            - Flows: Arrows connecting nodes, with width proportional to the quantity of flow.
            - Colors: Often used to distinguish different categories or types of flow.
            - Hierarchy: Can represent hierarchical data or multi-stage processes.
            """,
            "guidelines": """
            When interpreting Sankey Diagrams:
            1. Flow magnitude: Analyze the width of flows to understand the relative quantities moving between nodes.
            2. Pathways: Trace pathways through the diagram to understand process flows or hierarchical breakdowns.
            3. Bottlenecks: Identify nodes or flows that represent bottlenecks or key decision points.
            4. Efficiency: In systems representing processes, look for inefficiencies or unexpected flows.
            5. Comparisons: Compare different flows to understand relative proportions or importance.
            6. Color coding: Use color to track specific categories or types of flow through the system.
            """
        },
        "Bubble Charts": {
            "context": """
            Bubble Charts are an extension of scatter plots where a third dimension is added through the size of the markers (bubbles).
            - X and Y axes: Represent two variables, as in a standard scatter plot.
            - Bubble size: Represents a third variable, often quantity or magnitude.
            - Color: Can be used to represent a fourth variable or category.
            - Positioning: The position of each bubble's center is determined by its x and y values.
            """,
            "guidelines": """
            When interpreting Bubble Charts:
            1. Correlation: Look for relationships between the x and y variables, as you would in a scatter plot.
            2. Size patterns: Analyze how the size of bubbles relates to their position on the x and y axes.
            3. Clustering: Identify groups of bubbles with similar characteristics.
            4. Outliers: Look for bubbles that are unusually large or in unexpected positions.
            5. Proportions: Compare the sizes of bubbles to understand relative proportions of the third variable.
            6. Color coding: If color is used, look for patterns or relationships indicated by color distribution.
            """
        },
        "Geographical Plots": {
            "context": """
            Geographical Plots display data on a map, allowing for the visualization of spatial patterns and relationships.
            - Base map: The underlying geographical representation.
            - Data points: Often represented as markers or color-coded regions on the map.
            - Color scale: Used to represent data values in choropleth maps.
            - Layers: Different types of data can be displayed as overlapping layers.
            - Projections: The method used to represent the Earth's surface on a 2D plane.
            """,
            "guidelines": """
            When interpreting Geographical Plots:
            1. Spatial patterns: Look for geographical clustering or dispersion of data points or values.
            2. Regional variations: Analyze how data values or characteristics vary across different regions.
            3. Hotspots: Identify areas of high concentration or extreme values.
            4. Relationships: Consider how geographical features (e.g., coastlines, mountains) might relate to the data.
            5. Scale: Be aware of how the choice of scale affects the patterns visible in the data.
            6. Projections: Consider how the map projection might distort distances or areas, especially for global data.
            """
        },
        "Word Clouds": {
            "context": """
            Word Clouds are visual representations of text data where the size of each word indicates its frequency or importance.
            - Word size: Represents the frequency or importance of the word.
            - Color: Can be used to represent categories or can be purely aesthetic.
            - Layout: The arrangement of words, which can be random or follow a specific shape.
            - Stop words: Common words that are typically excluded from the visualization.
            - Stemming/Lemmatization: Techniques to combine different forms of the same word.
            """,
            "guidelines": """
            When interpreting Word Clouds:
            1. Dominant themes: Identify the largest words to understand the most common or important themes.
            2. Relative importance: Compare word sizes to understand relative frequencies or importance.
            3. Unexpected terms: Look for surprising words that appear larger than expected.
            4. Contextual relationships: Consider how the prominent words relate to each other and the overall topic.
            5. Absent words: Think about important words that you might expect but don't see.
            6. Limitations: Remember that word clouds don't show context, syntax, or sentiment, and can sometimes overemphasize long words.
            """
        },
        "Hierarchical Clustering Dendrogram": {
            "context": """
            Hierarchical Clustering Dendrograms visualize the results of hierarchical clustering, showing how data points are grouped at various levels of similarity.
            - Leaves: Represent individual data points or lowest-level clusters.
            - Branches: Show how clusters are merged.
            - Height: Represents the distance or dissimilarity between clusters.
            - Cuts: Horizontal lines that can be used to define a specific number of clusters.
            - Cophenetic distance: The height of the branch where two elements are first clustered together.
            """,
            "guidelines": """
            When interpreting Hierarchical Clustering Dendrograms:
            1. Cluster identification: Look for natural groupings by identifying long vertical lines.
            2. Similarity assessment: Compare heights at which clusters merge to understand relative similarities.
            3. Outliers: Identify data points that join the main clusters at a very high level.
            4. Cluster stability: Consider how stable clusters are by looking at the length of branches.
            5. Optimal cluster number: Use the dendrogram to inform decisions about the number of clusters to use.
            6. Validation: Cross-validate the clustering results with domain knowledge or other clustering methods.
            """
        },
        "ECDF Plots": {
            "context": """
            Empirical Cumulative Distribution Function (ECDF) Plots show the proportion of data points that are less than or equal to each value.
            - X-axis: Represents the range of the data.
            - Y-axis: Represents the cumulative probability (0 to 1).
            - Step function: The ECDF is a step function that increases at each data point.
            - Percentiles: Can be easily read from the y-axis.
            - Multiple ECDFs: Can be plotted together for comparison.
            """,
            "guidelines": """
            When interpreting ECDF Plots:
            1. Distribution shape: Analyze the overall shape to understand the distribution of the data.
            2. Median: Identify the 50th percentile (where y = 0.5).
            3. Spread: Look at the steepness of the curve to understand data spread.
            4. Outliers: Check for long flat areas at the beginning or end of the curve.
            5. Comparisons: When multiple ECDFs are plotted, compare their shapes and positions.
            6. Percentiles: Use the plot to estimate specific percentiles of the data.
            """
        },
        "Ridgeline Plots": {
            "context": """
            Ridgeline Plots (also known as Joy Plots) display the distribution of a numeric variable for several groups.
            - X-axis: Represents the range of the numeric variable.
            - Y-axis: Represents different categories or groups.
            - Density curves: Show the distribution of the numeric variable for each group.
            - Overlap: The amount of overlap between density curves can be adjusted.
            - Color: Often used to differentiate between groups or to show a gradient.
            """,
            "guidelines": """
            When interpreting Ridgeline Plots:
            1. Distribution comparison: Compare the shape and position of distributions across groups.
            2. Central tendency: Look for shifts in the peaks of distributions between groups.
            3. Spread: Analyze the width of distributions to understand variability within groups.
            4. Multimodality: Identify groups with multiple peaks, indicating subgroups or complex patterns.
            5. Outliers: Look for unusual shapes or long tails in specific groups.
            6. Trends: If groups are ordered (e.g., by time), look for trends in distribution changes.
            """
        },
        "Hexbin Plots": {
            "context": """
            Hexbin Plots are a form of heatmap for bivariate data, where data points are grouped into hexagonal bins.
            - X and Y axes: Represent two variables, as in a scatter plot.
            - Hexagons: Replace individual points, with color representing point density.
            - Color scale: Indicates the number of points falling within each hexagon.
            - Bin size: The size of hexagons, which affects the resolution of the plot.
            - Marginal distributions: Often shown on the sides of the main plot.
            """,
            "guidelines": """
            When interpreting Hexbin Plots:
            1. Density patterns: Identify areas of high and low point density.
            2. Correlation: Look for overall patterns that suggest relationships between variables.
            3. Outliers: Check for isolated hexagons far from the main concentration of data.
            4. Multimodality: Look for multiple distinct high-density regions.
            5. Edge effects: Be aware of how the choice of bin size affects the appearance of patterns, especially at the edges.
            6. Comparison with scatter plots: Consider how the hexbin representation changes your perception compared to individual points.
            """
        },
        "Mosaic Plots": {
            "context": """
            Mosaic Plots visualize the relationship between two or more categorical variables, with rectangle sizes proportional to cell frequencies.
            - Axes: Represent different categorical variables.
            - Rectangles: Each represents a combination of categories, with size proportional to frequency.
            - Colors: Often used to represent residuals or another categorical variable.
            - Standardization: Plots can be standardized to better show deviations from independence.
            - Spacing: Gaps between rectangles can indicate hierarchy or grouping.
            """,
            "guidelines": """
            When interpreting Mosaic Plots:
            1. Area comparison: Compare the sizes of rectangles to understand relative frequencies.
            2. Independence: In a standardized plot, look for rectangles that are larger or smaller than expected under independence.
            3. Color interpretation: If colors represent residuals, look for patterns of over- or under-representation.
            4. Conditional proportions: Examine how proportions change across different levels of a variable.
            5. Interactions: Look for patterns that suggest interactions between variables.
            6. Missing combinations: Be aware of combinations that don't appear in the data (empty cells).
            """
        },
        "Lag Plots": {
            "context": """
            Lag Plots are used to check for autocorrelation in time series data by plotting each observation against a lagged version of itself.
            - X-axis: Represents the original time series.
            - Y-axis: Represents the lagged time series.
            - Lag: The number of time steps between the original and lagged series.
            - Diagonal line: A reference line y=x is often included.
            - Scatter: The pattern of points indicates the type and strength of autocorrelation.
            """,
            "guidelines": """
            When interpreting Lag Plots:
            1. Randomness: A random scatter of points suggests no autocorrelation.
            2. Positive autocorrelation: Look for points clustering around the diagonal line.
            3. Negative autocorrelation: Points will tend to cluster in the opposite corners.
            4. Nonlinearity: Curved patterns suggest nonlinear autocorrelation.
            5. Outliers: Identify points that are far from the main cluster.
            6. Multiple lags: Compare lag plots with different lag values to understand the autocorrelation structure.
            """
        },
        "Shapley Value Analysis": {
            "context": """
            Shapley Value Analysis is a method from cooperative game theory used to determine feature importance in machine learning models.
            - Feature importance: Quantifies the contribution of each feature to the model's prediction.
            - Global importance: Averages feature importance across all predictions.
            - Local importance: Shows feature importance for individual predictions.
            - Interaction effects: Can capture complex interactions between features.
            - Model-agnostic: Can be applied to any machine learning model.
            """,
            "guidelines": """
            When interpreting Shapley Value Analysis:
            1. Feature ranking: Identify the most and least important features based on their Shapley values.
            2. Direction of impact: Understand whether features have a positive or negative impact on predictions.
            3. Consistency: Check if feature importance is consistent across different subsets of data.
            4. Interactions: Look for features whose importance varies significantly depending on other feature values.
            5. Outliers: Identify instances where feature importance differs dramatically from the average.
            6. Model understanding: Use Shapley values to gain insights into how the model makes predictions.
            """
        },
        "Partial Dependence Plots": {
            "context": """
            Partial Dependence Plots (PDPs) show the marginal effect of one or two features on the predicted outcome of a machine learning model.
            - X-axis: Represents the values of the feature of interest.
            - Y-axis: Represents the average predicted outcome.
            - Curve: Shows how the prediction changes as the feature value changes.
            - Interaction PDPs: Use two features to create a 3D or contour plot.
            - Ranges: Often show the distribution of the feature values in the dataset.
            """,
            "guidelines": """
            When interpreting Partial Dependence Plots:
            1. Trend analysis: Examine the overall trend of the PDP to understand the feature's effect.
            2. Nonlinearity: Look for nonlinear relationships between the feature and the prediction.
            3. Interactions: In 2D PDPs, analyze how the effect of one feature changes based on the other.
            4. Feature importance: Steeper curves generally indicate more important features.
            5. Ranges: Pay attention to regions where the plot is based on more data vs. extrapolated regions.
            6. Causal interpretation: Be cautious about causal interpretations, as PDPs assume feature independence.
            """
        },
            "Value Counts Analysis": {
            "context": """
            Value Counts Analysis provides a count of unique values in a categorical column.
            - Frequency: The number of occurrences of each unique value.
            - Proportion: The percentage of each unique value in the dataset.
            - Ranking: Values are typically sorted by frequency in descending order.
            """,
            "guidelines": """
            When interpreting Value Counts Analysis results:
            1. Dominant categories: Identify the most frequent categories and their proportions.
            2. Rare categories: Look for categories with very low frequencies, which might be errors or outliers.
            3. Distribution: Assess whether the distribution is balanced or skewed towards certain categories.
            4. Expected vs. Observed: Compare the results with domain knowledge or expectations.
            5. Missing values: Check if 'null' or 'NaN' appears in the value counts and its frequency.
            6. Implications: Consider how the distribution of categories might impact further analyses or business decisions.
            """
        },
        "Grouped Summary Statistics": {
            "context": """
            Grouped Summary Statistics provide descriptive statistics for numerical variables, grouped by one or more categorical variables.
            - Mean: Average value for each group.
            - Median: Middle value for each group when sorted.
            - Standard Deviation: Measure of spread for each group.
            - Min/Max: Minimum and maximum values for each group.
            - Count: Number of non-null observations in each group.
            """,
            "guidelines": """
            When interpreting Grouped Summary Statistics:
            1. Central tendency: Compare means and medians across groups to identify differences.
            2. Variability: Look at standard deviations to understand the spread within each group.
            3. Ranges: Examine min and max values to identify potential outliers or data quality issues.
            4. Sample sizes: Consider the count for each group, as smaller groups may have less reliable statistics.
            5. Patterns: Look for consistent patterns or trends across groups.
            6. Outlier groups: Identify any groups with significantly different statistics from the others.
            """
        },
        "Frequency Distribution Analysis": {
            "context": """
            Frequency Distribution Analysis visualizes the distribution of a variable using histograms or frequency plots.
            - Bins: Intervals into which the data is divided.
            - Frequency: The count or proportion of observations in each bin.
            - Shape: The overall form of the distribution (e.g., normal, skewed, bimodal).
            - Central tendency: Visible as peaks or centers of the distribution.
            - Spread: Represented by the width of the distribution.
            """,
            "guidelines": """
            When interpreting Frequency Distribution Analysis:
            1. Shape: Identify the overall shape of the distribution (e.g., normal, skewed, uniform).
            2. Central tendency: Look for the peak(s) of the distribution to understand the most common values.
            3. Spread: Assess the width of the distribution to understand variability.
            4. Outliers: Check for isolated bars or long tails that might indicate outliers.
            5. Multimodality: Look for multiple peaks which might suggest subgroups in the data.
            6. Gaps: Identify any gaps in the distribution which might indicate data collection issues or natural breakpoints.
            """
        },
        "KDE Plot Analysis": {
            "context": """
            Kernel Density Estimation (KDE) Plot Analysis provides a smoothed, continuous estimate of the probability density function of a variable.
            - Density estimate: A smooth curve representing the distribution of the data.
            - Bandwidth: Controls the smoothness of the KDE curve.
            - Multimodality: Multiple peaks in the KDE curve suggest multiple subpopulations.
            - Support: The range of values over which the KDE is defined.
            """,
            "guidelines": """
            When interpreting KDE Plot Analysis:
            1. Shape: Analyze the overall shape of the density curve for insights into the distribution.
            2. Peaks: Identify the location and number of peaks to understand central tendencies and potential subgroups.
            3. Spread: Assess the width of the curve to understand variability in the data.
            4. Tails: Examine the tails of the distribution for insights into extreme values or outliers.
            5. Comparison: If multiple KDE plots are presented, compare their shapes and locations.
            6. Smoothing: Be aware that the choice of bandwidth can affect the appearance of the KDE.
            """
        },
        "Violin Plot Analysis": {
            "context": """
            Violin Plot Analysis combines box plots with KDE plots to show the distribution of numerical data across categories.
            - Shape: The overall form of the violin represents the distribution.
            - Width: Indicates the frequency of data points at that value.
            - Box plot: Often included within the violin to show quartiles.
            - Comparison: Multiple violins allow for comparison across categories.
            """,
            "guidelines": """
            When interpreting Violin Plot Analysis:
            1. Shape comparison: Compare the shapes of violins across categories to understand differences in distributions.
            2. Central tendency: Look at the width of the violin at its center to understand the most common values.
            3. Spread: Assess the overall height of each violin to understand the range of values.
            4. Outliers: Check for thin extensions at the top or bottom of violins, which might indicate outliers.
            5. Multimodality: Look for multiple bulges in a violin, suggesting multiple subgroups within a category.
            6. Symmetry: Assess whether the distributions are symmetric or skewed by comparing the top and bottom halves of each violin.
            """
        },
        "Pair Plot Analysis": {
            "context": """
            Pair Plot Analysis creates a grid of scatter plots for every pair of numerical variables in a dataset, often including univariate distributions on the diagonal.
            - Scatter plots: Show relationships between pairs of variables.
            - Diagonal plots: Often show the distribution of individual variables.
            - Correlation: Visible as patterns in the scatter plots.
            - Outliers: Can be identified across multiple variables simultaneously.
            """,
            "guidelines": """
            When interpreting Pair Plot Analysis:
            1. Relationships: Look for clear patterns in the scatter plots that suggest relationships between variables.
            2. Correlation strength: Assess how tightly clustered points are around a potential relationship line.
            3. Non-linear relationships: Identify any curved patterns in the scatter plots.
            4. Outliers: Look for points that are far from the main cluster in multiple plots.
            5. Distributions: Examine the diagonal plots to understand the distribution of individual variables.
            6. Groupings: If color-coded, look for separation of groups across different variable combinations.
            """
        },
        "Box Plot Analysis": {
            "context": """
            Box Plot Analysis visualizes the distribution of numerical data through quartiles, often used to identify outliers and compare distributions.
            - Median: Represented by the line inside the box.
            - Interquartile Range (IQR): The box represents the middle 50% of the data.
            - Whiskers: Typically extend to 1.5 times the IQR.
            - Outliers: Often plotted as individual points beyond the whiskers.
            """,
            "guidelines": """
            When interpreting Box Plot Analysis:
            1. Central tendency: Compare median lines across boxes to understand differences in central tendencies.
            2. Spread: Assess the height of boxes and length of whiskers to understand variability.
            3. Skewness: Look at the position of the median line within the box and the relative lengths of the whiskers.
            4. Outliers: Identify any points plotted beyond the whiskers.
            5. Comparison: When multiple box plots are present, compare their shapes and positions.
            6. Symmetry: Assess whether the box and whiskers are roughly symmetrical around the median.
            """
        },
        "Scatter Plot Analysis": {
            "context": """
            Scatter Plot Analysis visualizes the relationship between two numerical variables by plotting points on a two-dimensional graph.
            - Correlation: Visible as patterns in the arrangement of points.
            - Strength of relationship: Indicated by how closely points follow a pattern.
            - Outliers: Points that deviate significantly from the overall pattern.
            - Clusters: Groups of points that are close together might indicate subgroups in the data.
            """,
            "guidelines": """
            When interpreting Scatter Plot Analysis:
            1. Direction: Determine if there's a positive, negative, or no apparent relationship between variables.
            2. Strength: Assess how tightly the points cluster around a potential trend line.
            3. Linearity: Determine if the relationship appears linear or if there's a curved pattern.
            4. Outliers: Identify any points that are far from the main cluster of data.
            5. Clusters: Look for distinct groups of points that might suggest subpopulations.
            6. Homoscedasticity: Check if the spread of points is consistent across the range of both variables.
            """
        },
        "Time Series Analysis": {
            "context": """
            Time Series Analysis examines data points collected over time to identify patterns, trends, and potentially make forecasts.
            - Trend: Long-term increase or decrease in the data.
            - Seasonality: Repeating patterns or cycles over fixed time periods.
            - Noise: Random variation in the data.
            - Change points: Moments where the behavior of the time series changes significantly.
            """,
            "guidelines": """
            When interpreting Time Series Analysis:
            1. Trend identification: Look for overall upward or downward movements in the data over time.
            2. Seasonal patterns: Identify any recurring patterns at fixed intervals (e.g., daily, monthly, yearly).
            3. Cyclical patterns: Look for longer-term cycles that aren't tied to a fixed time period.
            4. Anomalies: Identify any unusual spikes or dips that don't fit the overall pattern.
            5. Change points: Look for points where the behavior of the series changes significantly.
            6. Stationarity: Assess whether the statistical properties of the series (like mean and variance) are constant over time.
            """
        },
        "Outlier Detection": {
            "context": """
            Outlier Detection identifies data points that significantly differ from other observations.
            - IQR method: Often uses 1.5 times the Interquartile Range below Q1 or above Q3 to define outliers.
            - Z-score method: Identifies outliers based on how many standard deviations they are from the mean.
            - Local Outlier Factor: Considers the local density of points to identify outliers.
            - Isolation Forest: Uses the ease of isolating a point to determine if it's an outlier.
            """,
            "guidelines": """
            When interpreting Outlier Detection results:
            1. Frequency: Consider the number of outliers detected and whether this is reasonable for your dataset.
            2. Magnitude: Assess how far the outliers are from the main body of the data.
            3. Pattern: Look for any patterns in the outliers (e.g., do they occur at certain times or in certain categories?).
            4. Validity: Determine whether outliers are likely to be genuine unusual values or data errors.
            5. Impact: Consider how including or excluding outliers might affect further analyses.
            6. Context: Interpret outliers in the context of domain knowledge and data collection methods.
            """
        },
        "Feature Importance Analysis": {
            "context": """
            Feature Importance Analysis quantifies the contribution of each feature to the prediction of a target variable, often using machine learning models like Random Forests.
            - Importance scores: Numerical values indicating the relative importance of each feature.
            - Ranking: Features are typically ordered from most to least important.
            - Cumulative importance: The total importance accounted for by top N features.
            - Model-specific: Different models may produce different importance rankings.
            """,
            "guidelines": """
            When interpreting Feature Importance Analysis:
            1. Top features: Identify the most important features and consider their relevance to the problem domain.
            2. Relative importance: Compare importance scores to understand the relative contribution of different features.
            3. Cumulative importance: Determine how many features are needed to account for a large portion (e.g., 80%) of total importance.
            4. Irrelevant features: Identify features with very low importance scores, which might be candidates for removal.
            5. Correlation: Consider whether important features might be proxies for other, possibly unmeasured, variables.
            6. Model dependency: If possible, compare feature importance across different types of models for robustness.
            """
        },
        "PCA Analysis": {
            "context": """
            Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms the data into a new coordinate system.
            - Principal components: New variables that are linear combinations of the original features.
            - Explained variance ratio: The proportion of variance explained by each principal component.
            - Cumulative explained variance: The total variance explained by the first N components.
            - Loadings: The weights of original features in each principal component.
            """,
            "guidelines": """
            When interpreting PCA Analysis:
            1. Explained variance: Examine the explained variance ratio to understand how much information each component captures.
            2. Dimensionality reduction: Determine how many components are needed to explain a satisfactory amount of variance (e.g., 80%).
            3. Scree plot: Look for an 'elbow' in the scree plot to identify where additional components start to contribute less.
            4. Loadings: Analyze the loadings to understand which original features contribute most to each principal component.
            5. Biplot: If available, use biplots to visualize both the transformed data points and feature loadings.
            6. Outliers: In low-dimensional projections, look for any points that stand out as potential outliers.
            """
        },
        "Cluster Analysis": {
            "context": """
            Cluster Analysis groups similar data points together, often using algorithms like K-means.
            - Clusters: Groups of data points that are more similar to each other than to those in other clusters.
            - Centroids: The center points of clusters (in K-means).
            - Inertia: The sum of squared distances of samples to their closest cluster center.
            - Silhouette score: A measure of how similar an object is to its own cluster compared to other clusters.
            """,
            "guidelines": """
            When interpreting Cluster Analysis:
            1. Number of clusters: Assess whether the chosen number of clusters seems appropriate for the data.
            2. Cluster sizes: Look at the distribution of points across clusters. Are some much larger or smaller?
            3. Separation: Evaluate how well-separated the clusters are in feature space.
            4. Centroids: Examine the characteristics of cluster centers to understand what defines each group.
            5. Silhouette score: Use this to assess the quality of the clustering. Higher scores indicate better-defined clusters.
            6. Stability: If possible, assess whether the clustering is stable across different runs or subsets of the data.
            """
        },
        "Correlation Network Analysis": {
            "context": """
            Correlation Network Analysis visualizes relationships between variables as a network, where nodes represent variables and edges represent correlations.
            - Nodes: Represent variables in the dataset.
            - Edges: Connections between nodes, often with thickness or color representing correlation strength.
            - Layout: The arrangement of nodes, often using force-directed algorithms.
            - Clusters: Groups of highly interconnected nodes.
            """,
            "guidelines": """
            When interpreting Correlation Network Analysis:
            1. Strong correlations: Identify pairs or groups of variables with strong connections.
            2. Clusters: Look for clusters of interconnected variables, which might represent related concepts.
            3. Central nodes: Identify variables that are highly connected to many others, as these might be key factors.
            4. Isolated nodes: Notice any variables that have few or weak connections to others.
            5. Positive vs negative correlations: If indicated, distinguish between positive and negative relationships.
            6. Overall structure: Consider the general structure of the network (e.g., densely connected, sparse, hub-and-spoke).
            """
        },
        "Q-Q Plot Analysis": {
            "context": """
            Q-Q (Quantile-Quantile) Plot Analysis is a graphical technique for determining if two data sets come from populations with a common distribution.
            - Theoretical quantiles: Quantiles from a theoretical distribution (often normal).
            - Sample quantiles: Quantiles from the observed data.
            - Reference line: A 45-degree line representing perfect agreement between theoretical and sample quantiles.
            - Deviations: Points deviating from the reference line indicate departures from the theoretical distribution.
            """,
            "guidelines": """
            When interpreting Q-Q Plot Analysis:
            1. Linearity: Check if points generally follow the reference line, indicating agreement with the theoretical distribution.
            2. S-shape: An S-shaped curve suggests the data have heavier or lighter tails than the theoretical distribution.
            3. Outliers: Look for points that deviate substantially from the line at the ends of the plot.
            4. Skewness: If points curve above or below the line at one end, it suggests skewness in the data.
            5. Kurtosis: If points curve above the line at both ends and below in the middle (or vice versa), it suggests different kurtosis than the theoretical distribution.
            6. Transformations: Consider if a transformation of the data might result in a better fit to the theoretical distribution.
            """
        },
        "Data Summary": {
            "context": """
            Data Summary provides an overview of the dataset, including basic information about its size, structure, and content.
            - Number of rows: Total count of observations in the dataset.
            - Number of columns: Total count of variables or features.
            - Column names and data types: List of variables with their respective data types.
            - Preview: A glimpse of the first few rows of the dataset.
            """,
            "guidelines": """
            When interpreting Data Summary:
            1. Dataset size: Consider if the number of rows is sufficient for robust analysis and if it matches expectations.
            2. Feature space: Assess if the number of columns aligns with the expected complexity of the data.
            3. Data types: Check if the data types for each column are appropriate and as expected.
            4. Missing columns: Look for any expected columns that are not present in the dataset.
            5. Preview patterns: Examine the first few rows for any immediate patterns or potential data quality issues.
            6. Balance: For classification problems, consider if the preview suggests a balanced or imbalanced dataset.
            """
        },
        "Detailed Statistics Summary": {
            "context": """
            Detailed Statistics Summary provides comprehensive descriptive statistics for numerical variables and frequency counts for categorical variables.
            - Numeric stats: Include mean, median, standard deviation, min, max, and various percentiles.
            - Categorical stats: Include counts and proportions of each category.
            - Box plots: Visual representation of the distribution of numeric variables.
            """,
            "guidelines": """
            When interpreting Detailed Statistics Summary:
            1. Central tendency: Compare mean and median to understand the typical values and potential skewness.
            2. Spread: Use standard deviation and percentiles to gauge the dispersion of numeric variables.
            3. Ranges: Check min and max values to identify potential outliers or data entry errors.
            4. Categorical distributions: Examine the frequency of different categories to understand the composition of categorical variables.
            5. Skewness: Compare mean, median, and mode (most frequent value) to assess skewness in distributions.
            6. Outliers: Use box plots to visually identify potential outliers and their impact on summary statistics.
            """
        },
        "Null, Missing, and Unique Value Analysis": {
            "context": """
            This analysis focuses on identifying and quantifying missing data, null values, and the number of unique values in each column.
            - Null counts: Number of null entries in each column.
            - Null percentages: Proportion of null values in each column.
            - Unique value counts: Number of distinct values in each column.
            - Missing value heatmap: Visual representation of missing data patterns.
            """,
            "guidelines": """
            When interpreting Null, Missing, and Unique Value Analysis:
            1. Missing data patterns: Look for columns or rows with high proportions of missing data.
            2. Impact on analysis: Consider how missing data might affect subsequent analyses and if imputation is necessary.
            3. Unique values: For categorical variables, compare unique value counts to total rows to understand cardinality.
            4. Constant columns: Identify any columns with only one unique value, which may not be informative for analysis.
            5. Missing completely at random: Assess if missing data appears randomly distributed or follows a pattern.
            6. Data collection issues: High levels of missing data in specific columns may indicate data collection or processing issues.
            """
        },
        "Column Importance Analysis": {
            "context": """
            Column Importance Analysis assesses the potential significance of each column in the dataset based on various metrics.
            - Data type: The kind of data stored in the column (e.g., numeric, categorical).
            - Unique count: The number of distinct values in the column.
            - Null percentage: The proportion of missing values in the column.
            """,
            "guidelines": """
            When interpreting Column Importance Analysis:
            1. Data type relevance: Consider if the data type is appropriate for the intended analysis.
            2. Cardinality: For categorical variables, assess if the number of unique values is appropriate (e.g., not too high for effective categorical analysis).
            3. Missing data impact: Evaluate if the percentage of missing data might compromise the column's usefulness.
            4. Potential predictors: Identify columns with high unique counts and low null percentages as potentially important predictors.
            5. Redundancy: Look for columns with very low unique counts, which might be constant or near-constant.
            6. Data quality indicators: Use this analysis to prioritize columns for more detailed examination or cleaning.
            """
        },
        "Data Quality Report": {
            "context": """
            Data Quality Report provides a comprehensive overview of various data quality aspects across the dataset.
            - Missing values: Count and percentage of missing entries per column.
            - Duplicate rows: Number of duplicate entries in the dataset.
            - Data types: The data type of each column.
            - Unique values: Count of distinct values in each column.
            - Missing value visualization: Often a bar plot showing the percentage of missing values by column.
            """,
            "guidelines": """
            When interpreting Data Quality Report:
            1. Missing data severity: Assess the extent of missing data and its potential impact on analysis.
            2. Duplicates: Consider the implications of duplicate rows and whether they represent valid repeated measurements or data errors.
            3. Data type consistency: Check if data types align with the expected nature of each variable.
            4. Unique value assessment: For categorical variables, consider if the number of unique values aligns with expectations.
            5. Missing patterns: Examine the missing value visualization to identify any patterns or columns particularly affected by missing data.
            6. Overall quality: Use this report to gauge the overall cleanliness and reliability of the dataset, and to prioritize data cleaning efforts.
            """
        },
        "Hypothesis Testing Suggestions": {
            "context": """
            Hypothesis Testing Suggestions provide ideas for statistical tests that could be applied to the data based on its characteristics.
            - Test types: Suggestions may include t-tests, ANOVA, correlation tests, etc.
            - Variables involved: The specific variables in the dataset that could be used in each suggested test.
            - Test descriptions: Brief explanations of what each suggested test would examine.
            """,
            "guidelines": """
            When interpreting Hypothesis Testing Suggestions:
            1. Relevance: Consider how well each suggested test aligns with your research questions or business objectives.
            2. Assumptions: Remember that each statistical test has assumptions that need to be verified before application.
            3. Data types: Ensure that the suggested tests are appropriate for the data types of the variables involved.
            4. Multiple testing: Be aware of the potential need for multiple testing corrections if many tests are performed.
            5. Exploratory vs. Confirmatory: Distinguish between exploratory analyses and tests of pre-specified hypotheses.
            6. Further investigation: Use these suggestions as starting points for more detailed statistical analysis plans.
            """
        }
    }
    return technique_info.get(technique_name, {"context": "", "guidelines": ""})
