import unittest
import pandas as pd
from health_monitoring import load_data, clean_data, create_features, train_model, generate_report  # Update the import statement as per your actual script file name
import os

class TestScript(unittest.TestCase):

    def setUp(self):
        # Sample dataframe for testing
        data = {'age': [29, 30, 24], 'trestbps': [130, 120, 110], 'chol': [200, 210, 190], 
                'thalach': [150, 160, 170], 'oldpeak': [1.0, 1.2, 0.8], 'target': [1, 0, 1]}
        self.df = pd.DataFrame(data)
    
    def test_load_data(self):
        # Use a sample CSV file path for testing
        df = load_data('/workspaces/project-raavimalleswari/heart.csv')  # You need to have a test CSV file
        self.assertIsInstance(df, pd.DataFrame)

    def test_clean_data(self):
        cleaned_df = clean_data(self.df)
        self.assertEqual(cleaned_df.isna().sum().sum(), 0)

    def test_create_features(self):
        df_with_features = create_features(self.df)
        self.assertIn('age_squared', df_with_features.columns)

    def test_train_model(self):
        model, accuracy, X_test, y_test = train_model(self.df, 'target')
        self.assertGreaterEqual(accuracy, 0)  # Check if we get a non-negative accuracy

    def test_generate_report(self):
        generate_report(self.df, 'test_reports')
        self.assertTrue(os.path.exists('test_reports/summary.csv'))

if __name__ == '__main__':
    unittest.main()