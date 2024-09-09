from .data_processing import DataProcessor
from .visualization import Visualizer
from .modeling import Modeler
from .reporting import Reporter

class ExtraML:
    def __init__(self, problem_type='regression', target_column=None):
        self.problem_type = problem_type.lower()
        self.target_column = target_column
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        self.modeler = Modeler()
        self.reporter = Reporter()
        self.report = []

    def fit(self, train_file, test_file):
        # Load data
        train_data, test_data = self.data_processor.load_data(train_file, test_file)
        
        # Process data
        X_train, y_train, X_test = self.data_processor.process_data(train_data, test_data, self.target_column)
        
        # Visualize data
        self.visualizer.create_pair_plot(X_train, y_train)
        self.visualizer.create_box_plots(X_train)
        self.visualizer.correlation_analysis(X_train)
        
        # Model training and evaluation
        self.modeler.train_and_evaluate(X_train, y_train)
        
        # Generate report
        self.reporter.generate_html_report(self.report)

    def add_to_report(self, title, content):
        self.report.append({'title': title, 'content': content})

    def run_analysis(self, train_file, test_file):
        self.fit(train_file, test_file)
        print("Analysis complete. Check the HTML report for details.")