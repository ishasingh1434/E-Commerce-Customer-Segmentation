# Customer Segmentation and Cluster Analysis

This project performs customer segmentation on transactional online retail data using RFM (Recency, Frequency, Monetary) analysis and unsupervised machine learning models.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [RFM Feature Engineering](#rfm-feature-engineering)
- [Clustering Models](#clustering-models)
  - [K-Means](#k-means)
  - [DBSCAN](#dbscan)
- [Dimensionality Reduction & Visualization](#dimensionality-reduction--visualization)
  - [PCA](#pca)
  - [t-SNE](#t-sne)
- [Customer Personas and Marketing Strategies](#customer-personas-and-marketing-strategies)
- [Outputs](#outputs)

---

## Project Overview
The goal of this project is to categorize online customers into distinct groups based on their purchasing behavior. Businesses can then implement highly-targeted marketing strategies to optimize customer retention, maximize lifetime value, and win back dormant customers.

## Data Preprocessing
- Missing values (particularly `CustomerID`) are removed.
- Cancelled orders (invoices starting with 'C') are filtered out.
- Records with non-positive quantities (`Quantity <= 0`) or unit prices (`UnitPrice <= 0`) are excluded.
- A `TotalAmount` column (`Quantity * UnitPrice`) is created for transaction values.

## RFM Feature Engineering
We group transactions by `CustomerID` and compute three metrics:
1. **Recency**: Days since the customer's last purchase (relative to a reference date).
2. **Frequency**: Count of unique invoices (orders) placed by the customer.
3. **Monetary**: Total amount of money spent by the customer.

*Note: To handle skewness, metrics are log-transformed (`np.log1p`) and standardized using `StandardScaler` prior to clustering.*

## Clustering Models

### K-Means
- Evaluated using the **Elbow Method** (inertia check across cluster sizes 2–19).
- Final Model: **K-Means with 5 clusters**.
- Silhouette Score: **~0.25**

### DBSCAN
- Automatically identifies core density clusters and treats outlier customers as noise (labeled `-1`).
- Model parameters: `eps=0.5`, `min_samples=10`.
- Silhouette Score (excluding outliers): **~0.23**

## Dimensionality Reduction & Visualization
To visually validate the clustering performance in a 2D space, two techniques are applied:
- **PCA (Principal Component Analysis)**: Explains ~81.5% of total variance using 2 components.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Used with perplexity mapping on a randomized sample of 2,000 customers.

## Customer Personas and Marketing Strategies

Based on K-Means centroids compared to the average metrics of the entire cohort, customers are classified into:

| Cluster ID | Persona | Key Behavioral Pattern | Actionable Marketing Strategy |
| :--- | :--- | :--- | :--- |
| **0 & 1** | **Loyal Customers** | Low Recency (recent purchase) & High Frequency | VIP programs, early product releases, loyalty points |
| **2 & 3** | **At-Risk Customers** | High Recency (haven't returned in a while) | Targeted win-back emails, heavy recovery discounts |
| **4** | **Occasional Customers**| Low Frequency | Second-purchase incentives, bundle coupons, recommendations |

## Outputs
The pipeline saves two clean files to the root workspace:
1. `customer_segments.csv`: The complete customer dataset with their RFM metrics, PCA coordinates, and cluster labels.
2. `cluster_analysis.csv`: Grouped summary stats (mean Recency, Frequency, Monetary values) for each K-Means cluster.

