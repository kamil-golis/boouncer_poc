import pandas as pd 
import numpy as np 
from lightgbm import LGBMClassifier
import plotly.express as px
import plotly.graph_objects as go


class UncertaintyBooster:
    """
    Class intended to calculate and analyze uncertainty metrics for a boosting binary classification model.
    """
    def __init__(self
                 , df_test: pd.DataFrame
                 , model: LGBMClassifier
                 , feature_names: list
                 , target_name: str
                 , key_column_names: list
                 ):
        """
        Initializes the UncertaintyBooster instance with the necessary parameters.

        Args:
            df_test (pd.DataFrame): The test DataFrame containing features and target.
            model (LGBMClassifier): The LightGBM model to analyze.
            feature_names (list): List of feature column names.
            target_name (str): The target column name.
            key_column_names (list): List of key column names for identification (id columns).
        """
           
        self.df_test = df_test
        self.model = model
        self.feature_names = feature_names
        self.target_name = target_name
        self.key_column_names = key_column_names
        self.n_estimators = model.get_params()['n_estimators'] # Number of trees in the model - used for intermediate predictions calculations and plots
        self.intermediate_predictions_test = None # entity used to store intermediate predictions for the test set
        self.df_summary_test = df_test[self.key_column_names + [self.target_name]].copy()

    def create_master_key_column(self, separator='_'):
        """
        Creates a master key column by concatenating the specified key columns (MASTER_KEY).
        The default output column format is 'key1_key2_...'.

        Args:
            separator (str): The separator to use between key column values (default is '_').
        """
        self.df_summary_test['MASTER_KEY'] = self.df_summary_test[self.key_column_names].astype(str).agg(separator.join, axis=1)

    def get_final_predictions(self):
        """
        Generates final predictions for the test set.
        """
        self.df_summary_test['FINAL_PREDICTION'] = self.model.predict_proba(self.df_test[self.feature_names])[:, 1]

    def get_intermediate_predictions(self):
        """
        Generates intermediate predictions (over the trees) for the test set.
        """
        intermediate_predictions = []
        for n_trees in range(1, self.n_estimators + 1):
            prob = self.model.predict_proba(self.df_test[self.feature_names], num_iteration=n_trees)[:, 1]
            intermediate_predictions.append(prob)

        intermediate_predictions = np.array(intermediate_predictions)
        self.intermediate_predictions_test = intermediate_predictions

    def save_intermediate_predictions(self, file_path: str = '/cache/intermediate_predictions.npy'):
        """
        Saves the intermediate predictions of the UncertaintyBooster instance to a .npy file.

        Args:
            file_path (str): The file path where the .npy file will be saved.
        """
        if self.intermediate_predictions_test is None:
            raise ValueError("Intermediate predictions not found. Please run get_intermediate_predictions() first.")

        np.save(file_path, self.intermediate_predictions_test)
        print(f"Intermediate predictions saved to {file_path}")

    def load_intermediate_predictions(self, file_path: str = 'cache/intermediate_predictions.npy'):
        """
        Loads intermediate predictions from a .npy file into the UncertaintyBooster instance.

        Args:
            file_path (str): The file path from where the .npy file will be loaded.
        """
        try:
            self.intermediate_predictions_test = np.load(file_path, allow_pickle=True)
            print(f"Intermediate predictions loaded from {file_path}")     
        except Exception as e:
            print(f"Error loading intermediate predictions: {e}")

    def calculate_uncertainty_metrics(self):
        """ 
        Calculates uncertainty metrics based on intermediate predictions.
        - MAE over trees vs. final prediction
        - MAPE over trees vs. final prediction
        - Std Dev over trees
        - Coefficient of Variation over trees
        The outputs are stored in df_summary_test with appropriate column names.
        """
        if 'FINAL_PREDICTION' not in self.df_summary_test.columns:
            raise ValueError("Final predictions not found. Please run get_final_predictions() first.")
        if self.intermediate_predictions_test is None:
            raise ValueError("Intermediate predictions not found. Please run get_intermediate_predictions() first.")

        final_prediction = self.df_summary_test['FINAL_PREDICTION'].values
        mae_over_trees = np.mean(np.abs(self.intermediate_predictions_test - final_prediction), axis=0)
        mape_over_trees = np.mean(np.abs((self.intermediate_predictions_test - final_prediction) / (final_prediction + 1e-6)), axis=0)
        std_dev_over_trees = np.std(self.intermediate_predictions_test, axis=0)
        coef_var_over_trees = std_dev_over_trees / (np.mean(self.intermediate_predictions_test, axis=0) + 1e-6)  # Avoid division by zero

        self.df_summary_test['MAE_OVER_TREES'] = mae_over_trees
        self.df_summary_test['MAPE_OVER_TREES'] = mape_over_trees
        self.df_summary_test['STDDEV_OVER_TREES'] = std_dev_over_trees
        self.df_summary_test['COEFF_OF_VARIATION_OVER_TREES'] = coef_var_over_trees
        #create additional columns with deciles and percentiles for final prediction and uncertainty metrics
        for col in ['FINAL_PREDICTION', 'MAE_OVER_TREES', 'MAPE_OVER_TREES', 'STDDEV_OVER_TREES', 'COEFF_OF_VARIATION_OVER_TREES']:
            self.df_summary_test[f'{col}_DECILE'] = pd.qcut(self.df_summary_test[col], 10, labels=False, duplicates='drop') + 1
            self.df_summary_test[f'{col}_PERCENTILE'] = pd.qcut(self.df_summary_test[col], 100, labels=False, duplicates='drop') + 1

    def plot_predictions_distribution_for_key(self, key_values: str):
        """
        Generates a histogram of intermediate predictions for a given key,
        with the final prediction as horizontal line.

        Args:
            key_values (string): master key value, e.g. 'user_123_2025-09-15'
        """

        # get the index of the row matching the master key
        try:
            index = self.df_summary_test[self.df_summary_test['MASTER_KEY'] == key_values].index[0]
        except IndexError:
            print(f"No data found for key: {key_values}")
            return None
        
        # extract data for the corresponding observation
        final_prediction = self.df_summary_test.loc[index, 'FINAL_PREDICTION']
        target_value = self.df_summary_test.loc[index, self.target_name]
        intermediate_preds = self.intermediate_predictions_test[:, index]

        # determine histogram color based on target
        color_map = {0: 'rgba(173, 216, 230, 0.7)',  # Light blue for target 0
                     1: 'rgba(255, 182, 193, 0.7)'}  # Light red for target 1
        hist_color = color_map.get(target_value, 'gray') # Default to gray if target is not 0 or 1

        # create the histogram plot
        fig = go.Figure()

        # add histogram trace
        fig.add_trace(go.Histogram(
            x=intermediate_preds,
            name='Intermediate Predictions',
            marker_color=hist_color,
            opacity=0.6,
            bingroup=1
        ))
        
        # add vertical line for the final prediction
        fig.add_vline(
            x=final_prediction, 
            line_dash="dash", 
            line_color="black", 
            annotation_text="Final Prediction", 
            annotation_position="top right"
        )

        # update layout for a cleaner look
        fig.update_layout(
            title=f'Intermediate Predictions for Key: {key_values}, Target: {target_value}',
            xaxis_title='Prediction Value',
            yaxis_title='Count',
            bargap=0.05,
            height=800  
        )

        fig.show()

    def plot_predictions_convergence_for_key(self, key_values: str):
        """
        Generates a lineplot of the final prediction convergence through the trees for a given key,
        with the final prediction as vertical line.

        Args:
            key_values (string): master key value, e.g. 'user_123_2025-09-15'
        """

        # get the index of the row matching the master key
        try:
            index = self.df_summary_test[self.df_summary_test['MASTER_KEY'] == key_values].index[0]
        except IndexError:
            print(f"No data found for key: {key_values}")
            return None
        
        # extract data for the corresponding observation
        final_prediction = self.df_summary_test.loc[index, 'FINAL_PREDICTION']
        target_value = self.df_summary_test.loc[index, self.target_name]
        intermediate_preds = self.intermediate_predictions_test[:, index]

        # determine plot color based on target
        color_map = {0: 'rgba(0, 51, 153, 0.7)', # Dark blue for target 0
                     1: 'rgba(204, 0, 0, 0.7)'}  # Dark red for target 1
        hist_color = color_map.get(target_value, 'gray') # Default to gray if target is not 0 or 1

        # create the line plot
        fig = go.Figure()

        # add line trace
        fig.add_trace(go.Scatter(
            x=np.arange(1, self.n_estimators + 1),
            y=intermediate_preds,
            name='Intermediate Predictions',
            mode='lines',
            marker_color=hist_color,
            opacity=0.9
        ))
        
        # add horizontal line for the final prediction
        fig.add_hline(
            y=final_prediction, 
            line_dash="dash", 
            line_color="black", 
            annotation_text="Final Prediction", 
            annotation_position="bottom right",
            opacity = 0.5
        )

        # update layout for a cleaner look
        fig.update_layout(
            title=f'Intermediate Predictions for Key: {key_values}, Target: {target_value}',
            xaxis_title='Number of Trees (iteration)',
            yaxis_title='Prediction Value',
            height=800
        )

        fig.show()

    def plot_relative_predictions_convergence_for_keys(self, key_values: list):
        """
        Generates a lineplot of the final prediction relative convergence through the trees for given keys.
        The relative predictions are expressed as the MAPE between the intermediate prediction at each tree and the final prediction.

        Args:
            key_values (list or tuple): A list of values for the key columns, e.g., ['user_123', '2025-09-15']
        """
        
        # create the line plot
        fig = go.Figure()

        for id in key_values:
            # get the index of the row matching the master key
            try:
                index = self.df_summary_test[self.df_summary_test['MASTER_KEY'] == id].index[0]
            except IndexError:
                print(f"No data found for key: {id}")
                return None
            
            # extract data for the corresponding observation
            final_prediction = self.df_summary_test.loc[index, 'FINAL_PREDICTION']
            mape = self.df_summary_test.loc[index, 'MAPE_OVER_TREES'] 
            target_value = self.df_summary_test.loc[index, self.target_name]
            intermediate_preds = self.intermediate_predictions_test[:, index]

            # calculate relative predictions as MAPE
            relative_intermediate_preds = np.abs((intermediate_preds - final_prediction) / final_prediction) * 100

            # add line trace for intermediate predictions
            fig.add_trace(go.Scatter(
                x=np.arange(1, self.n_estimators + 1),
                y=relative_intermediate_preds,
                name=f'ID: {id} | Target: {target_value} | MAPE: {mape:.2f}%',
                mode='lines',
                opacity = 0.6
            ))

        # update layout for a cleaner look
        fig.update_layout(
            title=f'Relative Predictions Convergence',
            xaxis_title='Number of Trees (iteration)',
            yaxis_title='MAPE between Intermediate and Final Prediction',
            height=800
        )

        fig.show()

    def plot_uncertainty_vs_prediction_heatmap(self, uncertainty_metric: str = 'MAE_OVER_TREES', value_type: str = 'global_perc', 
                                            n_bins: int = 10):
        """
        Generates a heatmap with binned final predictions in rows and uncertainty metric in columns.
        
        Args:
            uncertainty_metric (str): The uncertainty metric to be used for the columns
            value_type (str): Type of value to display ('volume', 'global_perc', 'row_perc', 'col_perc')
            n_bins (int): Number of bins for discretizing continuous values
        """
        # Create bins for both metrics
        df = self.df_summary_test.copy()
        
        # Create bins using pandas qcut (equal-sized bins)
        df['FINAL_PREDICTION_BIN'] = pd.qcut(df['FINAL_PREDICTION'], 
                                        q=n_bins, 
                                        labels=[f"{x:.3f}-{y:.3f}" for x,y in 
                                                zip(pd.qcut(df['FINAL_PREDICTION'], q=n_bins).cat.categories.left,
                                                    pd.qcut(df['FINAL_PREDICTION'], q=n_bins).cat.categories.right)])
        
        df[f'{uncertainty_metric}_BIN'] = pd.qcut(df[uncertainty_metric], 
                                                q=n_bins,
                                                labels=[f"{x:.3f}-{y:.3f}" for x,y in 
                                                    zip(pd.qcut(df[uncertainty_metric], q=n_bins).cat.categories.left,
                                                        pd.qcut(df[uncertainty_metric], q=n_bins).cat.categories.right)])

        # Create pivot table
        pivot_table = pd.pivot_table(
            df,
            values='MASTER_KEY',
            index='FINAL_PREDICTION_BIN',
            columns=f'{uncertainty_metric}_BIN',
            aggfunc='count',
            fill_value=0
        )

        # Apply value type transformation
        if value_type == 'global_perc':
            pivot_table = (pivot_table / pivot_table.values.sum()) * 100
        elif value_type == 'row_perc':
            pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
        elif value_type == 'col_perc':
            pivot_table = pivot_table.div(pivot_table.sum(axis=0), axis=1) * 100

        # Create heatmap
        fig = px.imshow(
            pivot_table,
            labels=dict(
                x=f'{uncertainty_metric}', 
                y='Final Prediction', 
                color='Count' if value_type=='volume' else 'Percentage'
            ),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale='Viridis',
            aspect='auto'
        )
        
        # Add text annotations
        fig.update_traces(
            text=np.round(pivot_table.values, 2),
            texttemplate='%{text}',
            textfont={'size': 12},
            showscale=True
        )

        # Update layout
        fig.update_layout(
            title=f'Heatmap of Final Predictions vs {uncertainty_metric} ({value_type.replace("_", " ").title()})',
            xaxis_title=uncertainty_metric,
            yaxis_title='Final Prediction',
            xaxis={'tickangle': 45},
            height=800
        )
        
        fig.show()

    def plot_uncertainty_extended_confusion_matrix(self, uncertainty_metric: str = 'MAE_OVER_TREES', 
                                                final_prediction_threshold: float = 0.5,
                                                uncertainty_threshold: float = 0.1):
        """
        Generates a heatmap showing an uncertainty-extended confusion matrix.
        Matrix dimensions are: TARGET (1/0) x PREDICTION (1/0) x UNCERTAINTY (1/0)
        Each cell shows distribution between low/high uncertainty (sums to 100%).
        
        Args:
            uncertainty_metric (str): Metric to use for uncertainty calculation
            final_prediction_threshold (float): Threshold for converting probabilities to binary predictions
            uncertainty_threshold (float): Threshold for high/low uncertainty classification
        """
        df = self.df_summary_test.copy()
        
        # Create binary columns
        df['PRED_CLASS'] = (df['FINAL_PREDICTION'] >= final_prediction_threshold).astype(int)
        df['HIGH_UNCERTAINTY'] = (df[uncertainty_metric] >= uncertainty_threshold).astype(int)
        
        # Create combined categories for matrix
        df['CATEGORY'] = df.apply(lambda x: f"Target_{x[self.target_name]}_Pred_{x['PRED_CLASS']}_Uncert_{x['HIGH_UNCERTAINTY']}", axis=1)
        
        # Count occurrences
        category_counts = df['CATEGORY'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Create matrix data
        matrix_data = np.zeros((2, 2, 2))  # [target][prediction][uncertainty]
        
        for _, row in category_counts.iterrows():
            t, p, u = map(int, row['Category'].split('_')[1::2])
            matrix_data[t, p, u] = row['Count']
        
        # Create figure
        fig = go.Figure()
        
        # Calculate cell-wise percentages and create text annotations
        text_matrix = []
        values_matrix = []
        total_matrix = []
        
        for t in range(2):
            text_row = []
            values_row = []
            total_row = []
            for p in range(2):
                # Calculate total counts for this cell
                cell_total = matrix_data[t, p, 0] + matrix_data[t, p, 1]
                
                if cell_total > 0:
                    # Calculate percentages within this cell
                    low_unc_perc = (matrix_data[t, p, 0] / cell_total) * 100
                    high_unc_perc = (matrix_data[t, p, 1] / cell_total) * 100
                else:
                    low_unc_perc = high_unc_perc = 0
                
                # Create text annotation showing both percentages and total count
                text = f"Low: {low_unc_perc:.1f}%<br>High: {high_unc_perc:.2f}%<br>n={int(cell_total)}"
                text_row.append(text)
                values_row.append(high_unc_perc)  # Color based on high uncertainty percentage
                total_row.append(cell_total)
                
            text_matrix.append(text_row)
            values_matrix.append(values_row)
            total_matrix.append(total_row)
        
        # Create heatmap
        fig.add_trace(go.Heatmap(
            z=values_matrix,
            x=['Predicted 0', 'Predicted 1'],
            y=['Target 0', 'Target 1'],
            colorscale='RdYlBu_r',
            text=text_matrix,
            texttemplate='%{text}',
            textfont={"size": 14},
            showscale=True,
            colorbar=dict(title='% High Uncertainty<br>within Cell')
        ))

        # Update layout
        fig.update_layout(
            title=f'Uncertainty-Extended Confusion Matrix<br>{uncertainty_metric} (threshold={uncertainty_threshold})<br>Percentages within each Cell',
            xaxis_title='Predicted Class',
            yaxis_title='True Class',
            height=800,
            font=dict(size=12)
        )
        
        fig.show()