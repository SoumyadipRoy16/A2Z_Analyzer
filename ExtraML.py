import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from scipy.stats import uniform, randint
import base64   
from io import BytesIO          
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor 
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from collections import defaultdict
from scipy.stats import chi2_contingency, f_oneway
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64

class ExtraML:
    def __init__(self, problem_type='regression', target_column=None):
        self.problem_type = problem_type.lower()
        if self.problem_type != 'regression':
            raise ValueError("This implementation only supports regression problems.")
        
        self.models = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(),
            'MeanBaseline': DummyRegressor(strategy='mean'),
            'MedianBaseline': DummyRegressor(strategy='median')
        }
        
        self.best_model = None
        self.best_score = float('inf')
        self.best_params = None
        self.report = []
        self.target_column = target_column

    def add_to_report(self, title, content):
        self.report.append({'title': title, 'content': content})

    def capture_dataframe_head(self, df, title):
        self.add_to_report(title, df.head().to_string())

    def compute_correlation_ratio(self, categories, measurements):
        categories = pd.Categorical(categories)
        categories_unique = categories.categories
        measurements_grouped = [measurements[categories == cat] for cat in categories_unique]
        measurements_grouped = [group for group in measurements_grouped if len(group) > 0]
        
        ssb = sum(len(group) * (np.mean(group) - np.mean(measurements))**2 for group in measurements_grouped)
        sst = sum((x - np.mean(measurements))**2 for x in measurements)
        
        if sst == 0:
            return 0
        
        return np.sqrt(ssb / sst)

    def cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    def correlation_analysis(self, X):
        plots = []
        html_content = []

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.identify_categorical_columns(X)
        
        # Remove duplicates
        categorical_cols = [col for col in categorical_cols if col not in numeric_cols]

        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")

        # Numeric correlations
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr()
            self.add_to_report("Numeric Correlation Matrix", corr_matrix.to_string())
        else:
            self.add_to_report("Numeric Correlation Matrix", "Not enough numeric columns for correlation analysis.")

        # Categorical correlations
        if len(categorical_cols) > 1:
            cat_correlations = defaultdict(dict)
            for col1 in categorical_cols:
                for col2 in categorical_cols:
                    if col1 != col2:
                        cat_correlations[col1][col2] = self.cramers_v(X[col1], X[col2])
            cat_corr_matrix = pd.DataFrame(cat_correlations).fillna(1)
            self.add_to_report("Categorical Correlation Matrix", cat_corr_matrix.to_string())
        else:
            self.add_to_report("Categorical Correlation Matrix", "Not enough categorical columns for correlation analysis.")

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            num_cat_correlations = defaultdict(dict)
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    num_cat_correlations[num_col][cat_col] = self.compute_correlation_ratio(X[cat_col], X[num_col])

            num_cat_corr_matrix = pd.DataFrame(num_cat_correlations)
            
            # Custom color scale from light to dark
            colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
            
            fig = go.Figure(data=go.Heatmap(
                        z=num_cat_corr_matrix.values,
                        x=num_cat_corr_matrix.columns,
                        y=num_cat_corr_matrix.index,
                        colorscale=colors,
                        zmin=0, zmax=1))
            
            # Add text annotations
            for i, row in enumerate(num_cat_corr_matrix.values):
                for j, value in enumerate(row):
                    fig.add_annotation(
                        x=num_cat_corr_matrix.columns[j],
                        y=num_cat_corr_matrix.index[i],
                        text=f"{value:.2f}",
                        showarrow=False,
                        font=dict(color='black' if value < 0.7 else 'white')
                    )
            
            fig.update_layout(
                title="Correlation Ratio between Numeric and Categorical Features",
                width=1200,  # Increase from 1000
                height=1000, # Increase from 800
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                xaxis_side='top'
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(
                font=dict(size=10)  # Adjust this value as needed
            )

            # In your add_annotation loop:
            fig.add_annotation(
                x=num_cat_corr_matrix.columns[j],
                y=num_cat_corr_matrix.index[i],
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(color='black' if value < 0.7 else 'white', size=8)  # Adjust size here
            )

            fig.update_layout(
                xaxis=dict(rangeslider=dict(visible=True)),
                yaxis=dict(rangeslider=dict(visible=True))
            )
            fig.update_layout(
                dragmode='zoom',
                hovermode='closest'
            )
            plot_html = fig.to_html(full_html=False)
            plots.append(fig)
            html_content.append(plot_html)

            return plots, html_content
        else:
            print("Not enough numeric or categorical columns for correlation analysis")
            self.add_to_report("Numeric-Categorical Correlation Matrix", "Not enough numeric or categorical columns for correlation analysis.")

        print("Correlation analysis completed")

    def save_plot_to_report(self, title):
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_str = base64.b64encode(img_buf.getvalue()).decode()
        self.add_to_report(title, f'<img src="data:image/png;base64,{img_str}" alt="{title}">')
        plt.close()  

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
    
    def create_pair_plot(self, X, y):
        # Combine features and target
        data = pd.concat([X, y], axis=1)
        
        # Handle date columns
        date_columns = data.select_dtypes(include=['object']).columns[
            data.select_dtypes(include=['object']).apply(lambda col: col.str.match(r'\d{2}/\d{2}/\d{2}').all())
        ]
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], format='%d/%m/%y')
            data[f'{col}_year'] = data[col].dt.year
            data[f'{col}_month'] = data[col].dt.month
            data[f'{col}_day'] = data[col].dt.day
            data = data.drop(col, axis=1)
        
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        data = data[numeric_cols]
        
        # Select a subset of features if there are too many
        if len(data.columns) > 10:
            corr = data.corr()[y.name].abs().sort_values(ascending=False)
            top_features = corr.index[:9]  # Select top 9 features plus target
            data = data[top_features]
        
        # Create the pair plot
        fig = sns.pairplot(data, hue=y.name, diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.tight_layout()
        return fig

    def feature_engineering(self, X):
        categorical_cols = self.identify_categorical_columns(X)
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        return preprocessor
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        results = []
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            mse = mean_squared_error(y_val, model.predict(X_val))
            r2 = r2_score(y_val, model.predict(X_val))
            results.append({'model': name, 'mse': mse, 'r2': r2})
            if mse < self.best_score:
                self.best_score = mse
                self.best_model = name

        return results

    def hyperparameter_tuning(self, X_train, y_train):
        param_distributions = {
            'DecisionTree': {
                'max_depth': randint(1, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 20)
            },
            'KNN': {
                'n_neighbors': randint(1, 20),
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        }

        if self.best_model in param_distributions:
            random_search = RandomizedSearchCV(
                self.models[self.best_model],
                param_distributions=param_distributions[self.best_model],
                n_iter=50,
                cv=5,
                n_jobs=-1,
                random_state=42,
                scoring='neg_mean_squared_error'
            )

            random_search.fit(X_train, y_train)
            self.best_params = random_search.best_params_
            self.best_score = -random_search.best_score_

            self.add_to_report("Hyperparameter Tuning", 
                               f"Best Model: {self.best_model}\n"
                               f"Best Parameters: {self.best_params}\n"
                               f"Best MSE: {self.best_score}")
        else:
            self.add_to_report("Hyperparameter Tuning", 
                               f"No hyperparameter tuning performed for {self.best_model}")

    def visualize_model_comparison(self, results):
        df_results = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model', y='mse', data=df_results)
        plt.title('Model Comparison - Mean Squared Error')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error')
        plt.xticks(rotation=45)
        
        return plt.gcf()

    def handle_date_columns(self, X):
        date_columns = X.select_dtypes(include=['object']).columns[X.select_dtypes(include=['object']).apply(lambda col: col.str.match(r'\d{2}/\d{2}/\d{2}').all())]
        
        for col in date_columns:
            X[col] = pd.to_datetime(X[col], format='%d/%m/%y')
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X = X.drop(col, axis=1)
        
        # Only add to report if it hasn't been added before
        if not hasattr(self, 'date_columns_processed'):
            self.add_to_report("Date Column Handling", f"Processed date columns: {', '.join(date_columns)}")
            self.date_columns_processed = True
        
        return X

    def visualize_feature_importance(self, X, y):
        model = DecisionTreeRegressor(random_state=42)
            
        # Fit the model on the preprocessed data
        X_processed = self.feature_engineer.fit_transform(X)
        model.fit(X_processed, y)
            
        # Get feature names after preprocessing
        feature_names = (self.feature_engineer.named_transformers_['num'].get_feature_names_out().tolist() + 
                            self.feature_engineer.named_transformers_['cat'].get_feature_names_out().tolist())
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
            
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        return plt.gcf()

    def fit(self, train_data, test_data):
        plots = []
        html_content = []

        self.capture_dataframe_head(train_data, "Training Data Preview")
        self.capture_dataframe_head(test_data, "Test Data Preview")

        self.add_to_report("Data Overview", f"Training data shape: {train_data.shape}\nTest data shape: {test_data.shape}")

        if self.target_column is None:
            self.target_column = train_data.columns[-1]
            print(f"Automatically selected target column: {self.target_column}")
        else:
            if self.target_column not in train_data.columns:
                raise ValueError(f"Specified target column '{self.target_column}' not found in the training data.")
            print(f"User-specified target column: {self.target_column}")

        X_train = train_data.drop(self.target_column, axis=1)
        y_train = train_data[self.target_column]
        X_test = test_data.drop(self.target_column, axis=1) if self.target_column in test_data.columns else test_data

        # Check for missing values and duplicates
        self.check_missing_values(X_train)
        X_train = self.check_duplicates(X_train)
        
        # Handle missing values
        X_train = self.handle_missing_values(X_train)

        # Handle date columns
        X_train = self.handle_date_columns(X_train)
        X_test = self.handle_date_columns(X_test)

        # Create pair plot after handling date columns
        self.create_pair_plot(X_train, y_train)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        correlation_plots, correlation_html = self.correlation_analysis(X_train)
        plots.extend(correlation_plots)
        html_content.extend(correlation_html)

        # Feature engineering
        self.feature_engineer = self.feature_engineering(X_train)
        X_train_featured = self.feature_engineer.fit_transform(X_train)
        X_val_featured = self.feature_engineer.transform(X_val)

        # Train and evaluate models
        results = self.train_and_evaluate(X_train_featured, y_train, X_val_featured, y_val)
        
        model_comparison_plot = self.visualize_model_comparison(results)
        plots.append(model_comparison_plot)

        self.hyperparameter_tuning(X_train_featured, y_train)

        final_model = self.models[self.best_model]
        if self.best_params:
            final_model.set_params(**self.best_params)
        final_model.fit(X_train_featured, y_train)

        X_test_featured = self.feature_engineer.transform(X_test)

        y_pred = final_model.predict(X_test_featured)

        feature_importance_plot = self.visualize_feature_importance(X_train, y_train)
        plots.append(feature_importance_plot)

        mse = mean_squared_error(y_val, final_model.predict(X_val_featured))
        r2 = r2_score(y_val, final_model.predict(X_val_featured))
        performance_html = f"<p>Final Model Performance:<br>Mean Squared Error: {mse}<br>R-squared: {r2}</p>"
        html_content.append(performance_html)

        return y_pred, plots, html_content

    def generate_report(self, plots, html_content):
        report_content = "ExtraML Report\n\n"
        for item in self.report:
            report_content += f"{item['title']}\n"
            report_content += f"{item['content']}\n\n"
        
        report_content += "Plots and Visualizations:\n"
        for i, plot in enumerate(plots):
            img_buf = BytesIO()
            plot.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
            img_str = base64.b64encode(img_buf.getvalue()).decode()
            report_content += f'<img src="data:image/png;base64,{img_str}" alt="Plot {i+1}">\n\n'
        
        report_content += "HTML Content:\n"
        for html in html_content:
            report_content += f"{html}\n\n"
        
        return report_content

    def run_analysis(self, train_data, test_data):
        y_pred, plots, html_content = self.fit(train_data, test_data)
        self.remove_duplicate_report_entries()  
        return self.generate_report(plots, html_content)