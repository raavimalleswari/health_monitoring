import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Data Integration and Preprocessing
def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    # Drop rows with missing values
    df = df.dropna()
    
    # Normalize numerical features
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

# Exploratory Data Analysis
def plot_distributions(df, feature):
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

def correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

# Feature Engineering
def create_features(df):
    df['age_squared'] = df['age'] ** 2
    return df

# Time Series Analysis
def plot_time_series(df, feature):
    df[feature].plot()
    plt.title(f'Time Series of {feature}')
    plt.show()

# Supervised Machine Learning
def train_model(df, target):
    X = df.drop(columns=[target])
    y = df[target].astype(int)  # Ensure target is categorical
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy, X_test, y_test

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

# Reporting
def generate_report(df, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    summary = df.describe()
    summary.to_csv(os.path.join(output_path, 'summary.csv'))
    print("Report generated.")

def main():
    # File path to the dataset
    file_path = '/workspaces/project-raavimalleswari/heart.csv'
    
    # Data Integration and Preprocessing
    df = load_data(file_path)
    df = clean_data(df)
    
    # Exploratory Data Analysis
    plot_distributions(df, 'age')
    correlation_matrix(df)
    
    # Feature Engineering
    df = create_features(df)
    
    # Time Series Analysis
    plot_time_series(df, 'age')
    
    # Supervised Machine Learning
    model, accuracy, X_test, y_test = train_model(df, 'target')
    print(f'Model Accuracy: {accuracy}')
    
    # Model Evaluation
    evaluate_model(model, X_test, y_test)
    
    # Reporting
    generate_report(df, 'reports')

if __name__ == '__main__':
    main()