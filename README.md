# Startup Success Navigator🚀
Welcome to the "Startup Success Navigator" repository! This repository highlights the performance of our advanced machine learning models in predicting startup success. Additionally, I have deployed a Streamlit application that you can [Explore here](https://startup-success-navigator.streamlit.app/) to see in action.

## Table of Contents 
 - [Overview](#overview)
 - [Project Structure](#project-structure)
 - [Dataset and Description](#dataset-and-description)
 - [Business Key Performance Indicators](#business-key-performance-indicators)
 - [Features](#features)
 - [Tools and Technologies](#tools-and-technologies)
 - [Results](#results)
 - [Documentation](#documentation)
 - [Getting Started](#getting-started)
 - [Usage](#usage)
 - [Contributing](#contributing)
 - [License](#license)
   
## Overview 
In the rapidly changing startup landscape, having timely and precise predictions is essential. The Startup Success Navigator🚀 harnesses advanced machine learning techniques to anticipate startup success. This project provides real-time predictions and insightful visualizations to support investment decisions. Its goal is to analyze startup characteristics, pinpoint critical factors affecting success, and create a predictive model to forecast which startups are likely to succeed.

## Project Structure 
Here's an overview of the project's structure
```bash
│
├── Datasets/ 
│       ├── origional_startup_data.csv
│       └── startup_data.csv
│ 
├── Documentations/ 
│       ├── Documentation.pdf
│       └── Metrics.pdf
│ 
├── Notebooks/  
│       └── Startup Success Navigator.ipynb
│ 
└── Streamlit/ 
       ├── streamlit_app.py
       ├── startup_data.csv
       └── requirements.txt   
```

## Dataset and Description 
This dataset encompasses 49 columns and 923 rows, detailing a wide array of attributes related to startups. It includes information on startup locations, funding histories, types of relationships, and indicators of success. The data provides insights into factors such as geographical regions, funding amounts and stages, and various relational dynamics, all of which contribute to assessing the likelihood of startup success or failure.  

The [Dataset](https://github.com/AnuLikithaImmadisetty/Startup-Success-Navigator/tree/main/Datasets) is available here! Below are the Descriptions of the columns available in the dataset:

- **Unnamed: 0**: An index column or placeholder for missing data.  
- **state_code**: The code representing the state where the startup is headquartered.
- **latitude**: The latitude coordinates indicating the startup's north-south position.
- **longitude**: The longitude coordinates indicating the startup's east-west position.
- **zip_code**: The postal code of the startup's headquarters.
- **id**: A unique identifier assigned to each startup.
- **city**: The city in which the startup is located.
- **Unnamed: 6**: Another unnamed column, potentially serving as a placeholder or for unused data.
- **name**: The name of the startup.
- **labels**: Tags or labels associated with the startup, likely describing its industry or business model.
- **founded_at**: The date on which the startup was founded.
- **closed_at**: The date on which the startup ceased operations, if applicable.
- **first_funding_at**: The date when the startup received its initial funding.
- **last_funding_at**: The date of the most recent funding round.
- **age_first_funding_year**: The age of the startup, in years, at the time of its first funding.
- **age_last_funding_year**: The age of the startup, in years, at the time of its most recent funding.
- **age_first_milestone_year**: The age of the startup when it achieved its first significant milestone.
- **age_last_milestone_year**: The age of the startup when it achieved its most recent major milestone.
- **relationships**: The number of professional connections or relationships established by the startup.
- **funding_rounds**: The total number of funding rounds the startup has participated in.
- **funding_total_usd**: The cumulative amount of funding raised by the startup, in US dollars.
- **milestones**: The total number of significant milestones the startup has reached.
- **state_code.1**: A secondary state code, possibly for verification or additional categorization.
- **is_CA**: A binary indicator for whether the startup is located in California (1 for Yes, 0 for No).
- **is_NY**: A binary indicator for whether the startup is located in New York (1 for Yes, 0 for No).
- **is_MA**: A binary indicator for whether the startup is located in Massachusetts (1 for Yes, 0 for No).
- **is_TX**: A binary indicator for whether the startup is located in Texas (1 for Yes, 0 for No).
- **is_otherstate**: A binary indicator for whether the startup is located in any state other than CA, NY, MA, or TX (1 for Yes, 0 for No).
- **category_code**: The industry or sector category of the startup (e.g., software, biotech, ecommerce).
- **is_software**: A binary indicator for whether the startup operates in the software industry (1 for Yes, 0 for No).
- **is_web**: A binary indicator for whether the startup operates in the web industry (1 for Yes, 0 for No).
- **is_mobile**: A binary indicator for whether the startup operates in the mobile industry (1 for Yes, 0 for No).
- **is_enterprise**: A binary indicator for whether the startup operates in the enterprise industry (1 for Yes, 0 for No).
- **is_advertising**: A binary indicator for whether the startup operates in the advertising industry (1 for Yes, 0 for No).
- **is_gamesvideo**: A binary indicator for whether the startup operates in the games or video industry (1 for Yes, 0 for No).
- **is_ecommerce**: A binary indicator for whether the startup operates in the ecommerce industry (1 for Yes, 0 for No).
- **is_biotech**: A binary indicator for whether the startup operates in the biotech industry (1 for Yes, 0 for No).
- **is_consulting**: A binary indicator for whether the startup operates in the consulting industry (1 for Yes, 0 for No).
- **is_othercategory**: A binary indicator for whether the startup operates in an industry other than those listed (1 for Yes, 0 for No).
- **object_id**: A unique identifier for each startup in the dataset.
- **has_VC**: A binary indicator for whether the startup has received venture capital funding (1 for Yes, 0 for No).
- **has_angel**: A binary indicator for whether the startup has received angel funding (1 for Yes, 0 for No).
- **has_roundA**: A binary indicator for whether the startup has completed a Series A funding round (1 for Yes, 0 for No).
- **has_roundB**: A binary indicator for whether the startup has completed a Series B funding round (1 for Yes, 0 for No).
- **has_roundC**: A binary indicator for whether the startup has completed a Series C funding round (1 for Yes, 0 for No).
- **has_roundD**: A binary indicator for whether the startup has completed a Series D funding round (1 for Yes, 0 for No).
- **avg_participants**: The average number of participants involved in the startup's funding rounds.
- **is_top500**: A binary indicator for whether the startup is ranked in the top 500 startups (1 for Yes, 0 for No).
- **status**: The current status of the startup (e.g., acquired, closed).

## Business Key Performance Indicators
1. **Total Funding Raised**: The aggregate amount of funding secured by startups across all funding rounds, expressed in USD.  
2. **Success Rate**: The proportion of startups that are currently operational, have been acquired, or have ceased operations, providing insights into overall startup success.  
3. **Venture Capital Acquisition Rate**: The percentage of startups that have successfully obtained venture capital (VC) funding.  
4. **Milestone Achievement**: The number of startups that have reached significant milestones, such as securing funding or establishing strategic partnerships.  
5. **Average Age at First Funding**: The mean age of startups at the time they receive their initial round of funding.  
6. **Startup Mortality Rate**: The ratio of startups that have closed compared to those that remain operational, indicating the startup survival rate.  
7. **Growth Trajectory**: The analysis of growth by examining the time elapsed between a startup's first and most recent funding or milestone achievement, providing insights into their development and scaling progress. 
  
## Features 
The Startup Sucess Navigator🚀 project incorporates the following components:
1. **Data Inspection**: Review and understand the dataset's structure and content.  
2. **Data Preprocessing for EDA**: Prepare the data for exploratory data analysis by handling missing values, outliers, and inconsistencies.  
3. **Exploratory Data Analysis (EDA)**: Perform an in-depth analysis to uncover patterns, trends, and relationships within the data.  
4. **Data Preprocessing for Modeling**: Transform and prepare the data for model training, including feature selection and scaling.  
5. **Modelling**: Develop and train predictive models to forecast startup success based on the analyzed data.  

## Tools and Technologies 
This project leverages a variety of machine learning models to identify the success of the startups. The models and scalings used include:

### Machine Learning Models:  
🤖🧠 **Logistic Regression:** A linear model for binary classification that estimates the probability of a class based on a logistic function. 📈

👥🔍 **K-Nearest Neighbors (KNN):** A non-parametric method that classifies a sample based on the majority class among its k-nearest neighbors. 👥

🔧🛡️ **Support Vector Machine (SVM):**
- **Linear Kernel:** Finds the optimal hyperplane that separates classes with the maximum margin. 🛡️
- **RBF Kernel:** Uses a radial basis function to handle non-linear boundaries. 🌐
- **Polynomial Kernel:** Computes the similarity between data points in a higher-dimensional space using polynomial functions. 🔢

🌳🛠️ **Random Forest:** An ensemble method that constructs multiple decision trees and combines their predictions to enhance accuracy and control overfitting. 🌳

🌲🔍  **Decision Tree:** A model that splits the data into subsets based on feature values to make decisions or predictions in a tree-like structure. 🌲

🚀📈 **XGBoost:** An efficient and scalable boosting method that builds decision trees sequentially to correct errors made by previous trees. 🚀

💪🔧 **AdaBoost:** An ensemble technique that combines weak classifiers to create a strong classifier by giving more weight to misclassified samples. 💪

📊🛠️ **Gradient Boosting:** A boosting technique that builds models sequentially, each one correcting errors of the previous model, to improve predictive performance. 📈

### Scaling Methods:

- **StandardScaler:** Scales features to have a mean of 0 and a standard deviation of 1 for normalization. 📉
- **MinMaxScaler:** Scales features to a range between 0 and 1 for feature scaling. 📊
- **RobustScaler:** Scales features using statistics that are robust to outliers, such as the median and interquartile range. 🛠️

Refer to the [Notebooks](https://github.com/AnuLikithaImmadisetty/Startup-Success-Navigator/tree/main/Notebooks) here!

## Results
Refer to the [Results](https://linktoresults) here!

## Documentation 
Refer to the [Documentation](https://linktodocumentation) here!

## Getting Started 
To get started with the Startup Success Navigator project, follow these steps:
1. **Clone the Repository:** Clone the repository to your local machine using Git: (`git clone https://github.com/AnuLikithaImmadisetty/Startup-Success-Navigator.git`)
2. **Navigate to the Project Directory:** Change to the project directory: (`cd Startup-Success-Navigator`)
3. **Install Dependencies:** Install the required dependencies using pip: (`pip install -r requirements.txt`)
4. **Prepare the Data:** Download the dataset and place it in the `/Datasets` folder and ensure the data is in the expected format as described in the data documentation.
5. **Run the Analysis:** Open the Jupyter notebooks in Google Collab or Visual Studio Code located in the `/Notebooks` folder, here you can explore various models and run the corresponding scripts to process the data, train the models, and make predictions.
6. **Evaluate the Models:** Review the evaluation metrics and results in the `/Notebooks` folder. Metrics which will analyze the performance of the models and compare their predictions. It will be available in the **METRICS.pdf** in the `/Documentations` folder.
  
## Usage 
To use the trained models for Startup Success Navigator🚀:
1. Format and preprocess your text data to align with the training data specifications.
2. Utilize the provided notebooks in the `/Notebooks` directory to load the trained model and generate predictions.
3. Examine the output to identify whether the startup was success or not?

## Contributing 
Contributions are welcome to this project. To contribute, please follow these steps:
1. Fork the Repository.
2. Create a New Branch (`git checkout -b feature/YourFeature`).
3. Make Your Changes.
4. Commit Your Changes (`git commit -m 'Add new feature'`).
5. Push to the Branch (`git push origin feature/YourFeature`).
6. Create a Pull Request.

## License 
This project is licensed under the MIT License. Refer to the LICENSE file included in the repository for details.






