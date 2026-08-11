# Knowledge Graph Based Relationship Discovery for Loan Approval Prediction

## Overview

This project explores how statistical analysis, machine learning models, explainability techniques, and knowledge graphs can be combined to discover and represent meaningful relationships within a loan approval dataset.

Instead of treating machine learning as a black box, the goal is to identify **which applicant characteristics influence loan approval decisions**, validate those relationships using multiple analytical techniques, and transform the findings into an interpretable **Knowledge Graph (KG)**.

The final graph captures:

* Important applicant features
* Strength of evidence supporting each relationship
* Decision rules extracted from machine learning models
* Connections between applicant attributes and loan outcomes
* Explainable reasoning paths for individual applicants

---

## Problem Statement

Financial institutions often use large amounts of applicant information when evaluating loan applications.

Traditional machine learning models can predict approval outcomes, but they rarely explain:

* Why a prediction was made
* Which features are most influential
* How different factors interact
* What relationships exist between applicant attributes and approval decisions

This project aims to bridge that gap by discovering relationships through multiple analytical methods and representing them as a Knowledge Graph.

---

## Dataset

The project uses a Loan Approval Dataset containing applicant information such as:

* Gender
* Marital Status
* Dependents
* Education
* Self Employment Status
* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area
* Loan Approval Status

---

## Project Pipeline

```text
Loan Dataset
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Statistical Relationship Discovery
      │
      ▼
Machine Learning Relationship Discovery
      │
      ▼
Evidence Validation
      │
      ▼
Knowledge Graph Construction
      │
      ▼
Applicant-Level Explanations
```

---

## Data Preprocessing

### Missing Value Handling

Missing values were imputed using mode and median based strategies.

### Categorical Encoding

Categorical attributes were converted into numerical representations using Label Encoding.

### Feature Engineering

Several additional features were created:

#### Total Income

```python
TotalIncome = ApplicantIncome + CoapplicantIncome
```

#### Loan Income Ratio

```python
LoanAmount / TotalIncome
```

Measures the financial burden of the requested loan.

#### Income Per Dependent

```python
TotalIncome / (Dependents + 1)
```

Captures available income relative to family size.

#### Income Category

Applicants grouped into:

* Low
* Medium
* High
* Very High

#### Loan Category

Loans grouped into:

* Small
* Medium
* Large
* Very Large

---

## Relationship Discovery Methods

The project validates relationships using multiple independent approaches.

### 1. Statistical Analysis

#### Pearson Correlation

Used for numerical features to measure linear relationships with loan approval.

#### Chi-Square Test

Used for categorical variables to determine dependency with approval status.

#### Cramér's V

Measures the strength of association between categorical variables and loan outcomes.

#### Independent T-Test

Compares approved and rejected applicants across numerical features.

---

### 2. Logistic Regression

A Logistic Regression model was trained to determine:

* Positive contributors to approval
* Negative contributors to approval
* Odds ratios for each feature

This provides interpretable feature coefficients.

---

### 3. Mutual Information

Mutual Information was used to detect non-linear relationships between features and the target variable.

---

### 4. Random Forest Feature Importance

A Random Forest classifier was trained and feature importance scores were extracted to identify influential variables.

---

### 5. Permutation Importance

Feature importance was validated by measuring performance degradation when individual features were shuffled.

---

### 6. SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to:

* Explain model predictions
* Quantify feature contributions
* Generate global feature importance rankings

Visualizations include:

* SHAP Summary Plots
* Waterfall Explanations

---

### 7. Decision Tree Rule Extraction

A Decision Tree was trained to extract human-readable approval rules.

Example:

```text
Credit_History > 0.5
        └── Loan Approved

Credit_History <= 0.5
        └── Loan Rejected
```

These rules are later incorporated into the Knowledge Graph.

---

## Multi-Evidence Validation Framework

A relationship is not accepted based on a single method.

Each feature receives evidence from:

* Statistical significance
* Logistic Regression
* Random Forest Importance
* Permutation Importance
* SHAP Importance

An Evidence Score is computed to quantify confidence.

### Confidence Levels

| Evidence Score | Confidence |
| -------------- | ---------- |
| 5              | Very High  |
| 4              | High       |
| 3              | Moderate   |
| 2              | Low        |
| 1              | Very Low   |

Only sufficiently validated relationships are used in Knowledge Graph construction.

---

## Knowledge Graph Construction

The Knowledge Graph is built using NetworkX.

### Node Types

#### Outcome Nodes

* Loan Approved
* Loan Rejected

#### Feature Nodes

Examples:

* Credit_History
* LoanIncomeRatio
* TotalIncome
* Education

#### Decision Rule Nodes

Examples:

* Credit_History > 0.5
* LoanIncomeRatio <= Threshold

#### Applicant Nodes

Represent individual applicants.

#### Feature Value Nodes

Examples:

* Credit_History = 1
* Education = Graduate
* LoanIncomeRatio = 0.03

---

### Edge Types

#### Influences

```text
Feature → Outcome
```

Represents validated relationships.

#### Generates Rule

```text
Feature → Decision Rule
```

Connects attributes to extracted rules.

#### Supports

```text
Decision Rule → Outcome
```

Shows which outcome is supported by the rule.

#### Has Attribute

```text
Applicant → Feature Value
```

Stores applicant-specific information.

#### Indicates

```text
Feature Value → Feature
```

Links values to their corresponding feature.

---

## Explainable Applicant Analysis

The graph can generate explanations for individual applicants.

Example output:

```text
Applicant 25 is primarily influenced by:

1. Credit History
2. Loan Income Ratio
3. Total Income
```

This allows model decisions to be traced through graph relationships rather than treated as opaque predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* SciPy
* SHAP
* NetworkX
* Matplotlib

---

## Key Learning Outcomes

* Statistical relationship discovery
* Feature engineering
* Explainable AI (XAI)
* Knowledge Graph construction
* Evidence-based relationship validation
* Graph-based reasoning
* Financial risk analytics

---

## Future Work

* Neo4j-based graph storage
* GraphRAG integration
* Applicant similarity retrieval
* Loan approval recommendation engine
* What-if scenario analysis
* Counterfactual explanations
* Knowledge Graph embeddings
* Graph Neural Networks (GNNs)

---

## Author

**Aarushi Arora**

AI & Machine Learning Enthusiast | Knowledge Graphs | Explainable AI | Applied Machine Learning
