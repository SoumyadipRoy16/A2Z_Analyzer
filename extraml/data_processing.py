import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataProcessor:
    def load_data(self, train_file, test_file):
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        return train_data, test_data
    
    def process_data(self, train_data, test_data, target_column):
        # Handle missing values, duplicates, date columns, etc.
        # Split data
        X_train = train_data.drop(target_column, axis=1)
        y_train = train_data[target_column]
        X_test = test_data.drop(target_column, axis=1) if target_column in test_data.columns else test_data
        
        return X_train, y_train, X_test
    
    def check_missing_values(self, data):
        missing_values = data.isnull().sum()
        missing_percentages = 100 * missing_values / len(data)
        missing_table = pd.concat([missing_values, missing_percentages], axis=1, keys=['Total', 'Percent'])
        missing_table = missing_table[missing_table['Total'] > 0].sort_values('Total', ascending=False)
        
        if missing_table.empty:
            self.add_to_report("Missing Values", "No missing values found in the dataset.")
        else:
            self.add_to_report("Missing Values", f"Missing values found:\n{missing_table.to_html()}")

    def check_duplicates(self, data):
        n_duplicates = data.duplicated().sum()
        if n_duplicates > 0:
            self.add_to_report("Duplicate Values", f"Found {n_duplicates} duplicate rows. These will be removed.")
            data = data.drop_duplicates()
        else:
            self.add_to_report("Duplicate Values", "No duplicate rows found.")
        return data
    
    def remove_duplicate_report_entries(self):
        seen_titles = set()
        unique_report = []
        for item in self.report:
            if item['title'] not in seen_titles:
                seen_titles.add(item['title'])
                unique_report.append(item)
        self.report = unique_report

    def identify_categorical_columns(self, X):
        object_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        low_cardinality_cols = [col for col in numeric_cols if X[col].nunique() < 10 and col not in object_cols]
        return list(object_cols) + low_cardinality_cols
    
    def handle_missing_values(self, X):
        missing_values = X.isnull().sum()
        has_missing = missing_values.sum() > 0

        if not has_missing:
            return X  # Return the original dataframe if no missing values

        # Separate numerical and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Impute numerical columns with median
        if len(numeric_features) > 0 and missing_values[numeric_features].sum() > 0:
            num_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
            self.add_to_report("Handling Missing Values (Numeric)", "Numeric missing values were imputed using median strategy.")

        # Impute categorical columns with most frequent value
        if len(categorical_features) > 0 and missing_values[categorical_features].sum() > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])
            self.add_to_report("Handling Missing Values (Categorical)", "Categorical missing values were imputed using most frequent value strategy.")

        return X
    
