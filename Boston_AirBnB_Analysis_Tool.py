import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import simpledialog, messagebox
import warnings

#Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

#Set the working directory to the script's location
os.chdir(os.path.dirname(__file__))

def load_data(file_path):
    """
    Load and preprocess the Airbnb listings data.
    """
    print("\nLoading data...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        exit(1)

    #Select relevant columns based on original indices
    #Adjust the column indices as per the actual CSV structure
    df = df.iloc[:, [27, 33, 38, 39, 56]]
    df.columns = ['neighborhood', 'accommodates', 'amenities', 'price', 'reviews_last_12_months']

    #Clean and convert price to float
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

    #Fill missing reviews with 0 and convert to integer
    df['reviews_last_12_months'] = df['reviews_last_12_months'].fillna(0).astype(int)

    #Handle missing amenities
    df['amenities'] = df['amenities'].fillna('')

    #Feature engineering: Create binary features for amenities
    amenities_features = ['cable tv', 'dryer', 'dedicated workspace', 'indoor fireplace', 'wifi']
    for feature in amenities_features:
        df[feature.replace(' ', '_')] = df['amenities'].apply(
            lambda x: 1 if feature.lower() in x.lower() else 0
        )

    print("Data loaded and processed successfully.\n")
    return df

def calculate_estimated_income(df):
    """
    Calculate the estimated income based on reviews and price.
    """
    print("Calculating estimated income...")
    average_stay_length = 4
    review_rate = 0.72
    df['estimated_stays'] = df['reviews_last_12_months'] * review_rate
    df['estimated_income'] = df['estimated_stays'] * average_stay_length * df['price']
    print("Estimated income calculated.\n")
    return df

def filter_data(df, income_threshold=10000):
    """
    Filter listings with estimated income above a specified threshold.
    """
    print(f"Filtering listings with estimated income > ${income_threshold}...")
    filtered_df = df[df['estimated_income'] > income_threshold].copy()
    print(f"Number of listings after filtering: {len(filtered_df)}\n")
    return filtered_df

def prepare_model_data(df):
    """
    Prepare feature matrix X and target vector y for modeling.
    """
    print("Preparing data for modeling...")
    feature_cols = ['accommodates', 'cable_tv', 'dryer', 'dedicated_workspace', 'indoor_fireplace', 'wifi']
    X = df[feature_cols]
    y = df['estimated_income']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    print("Data split into training and testing sets.\n")
    return X_train, X_test, y_train, y_test, feature_cols

def train_model(X_train, y_train):
    """
    Train the Linear Regression model.
    """
    print("Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed.\n")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance.
    """
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    average_error = np.mean(predictions - y_test)
    income_mean = y_test.mean()
    error_percentage = (average_error / income_mean) * 100
    print(f"Average estimated income of sample set: ${income_mean:.2f}")
    print(f"Average amount of error: ${average_error:.2f}  Error Percentage: {error_percentage:.2f}%\n")
    return predictions, average_error, error_percentage

def display_feature_importance(model, feature_cols):
    """
    Display the importance of each feature in the model.
    """
    print("Feature Importance:")
    coefficients = model.coef_
    for feature, coef in zip(feature_cols, coefficients):
        print(f"Feature: {feature.replace('_', ' ').title()}, Value Increase By Addition: ${coef:.5f}")
    print("\nFeature Translation:")
    translations = {
        'accommodates': 'Each Additional Guest Capacity',
        'cable_tv': 'Having Cable TV',
        'dryer': 'Having a Dryer',
        'dedicated_workspace': 'Having a Dedicated Workspace',
        'indoor_fireplace': 'Having an Indoor Fireplace',
        'wifi': 'Having WiFi'
    }
    for idx, feature in enumerate(feature_cols):
        print(f"{idx}: {translations[feature]}")
    print("\n")

def get_user_input(feature_cols):
    """
    Collect user input for predicting estimated income.
    """
    user_input = {}
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    message = (
        "Enter the attributes of your potential Airbnb listing in the following prompts.\n"
        "After entering your response, press Enter/Return to continue to the next prompt."
    )
    messagebox.showinfo("Input Instructions", message)

    for feature in feature_cols:
        while True:
            try:
                if feature == 'accommodates':
                    prompt = "Enter the maximum number of guests for the listing:"
                    value = simpledialog.askinteger("Input", prompt, minvalue=1)
                else:
                    prompt = f"Does the listing have {feature.replace('_', ' ').title()}? (Enter 1 for Yes, 0 for No):"
                    value = simpledialog.askinteger("Input", prompt, minvalue=0, maxvalue=1)

                if value is None:
                    raise ValueError("Input cannot be empty.")

                user_input[feature] = value
                break
            except ValueError as ve:
                messagebox.showerror("Invalid Input", f"Please enter a valid value.\n{ve}")

    root.destroy()
    return user_input

def predict_income(model, user_input, feature_cols):
    """
    Predict the estimated income based on user input.
    """
    input_features = [user_input[feature] for feature in feature_cols]
    test_case = np.array([input_features])
    estimated_income = model.predict(test_case)[0]
    return estimated_income

def main():
    print("___Starting Boston Airbnb Analysis Tool 2022___\n")

    #Load and preprocess data
    data = load_data('listings.csv')

    #Calculate estimated income
    data = calculate_estimated_income(data)

    #Filter listings with estimated income > $10,000
    filtered_data = filter_data(data, income_threshold=10000)

    #Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_cols = prepare_model_data(filtered_data)

    #Train the model
    model = train_model(X_train, y_train)

    #Evaluate the model
    evaluate_model(model, X_test, y_test)

    #Display feature importance
    display_feature_importance(model, feature_cols)

    #Collect user input
    user_input = get_user_input(feature_cols)

    #Predict estimated income based on user input
    estimated_income = predict_income(model, user_input, feature_cols)

    #Display the results
    print("User Input:")
    for feature in feature_cols:
        display_name = feature.replace('_', ' ').title()
        print(f"{display_name}: {user_input[feature]}")
    print(f"\nEstimated Annual Income for the Listing: ${estimated_income:.2f}")

if __name__ == "__main__":
    main()
