import os
import pickle

import optuna
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class HyperparameterOptimizer:

    def __init__(self, objective_function, save_filename):
        self.study = None
        self.objective_function = objective_function
        self.save_filename = save_filename
        self.last_trial = None

    def create_new_study(self, explore_randomly=True, exploration_weight=1.0):
        """
        Create a new study with a specified exploration-exploitation tradeoff.

        Args:
            exploration_weight (float): Indicates exploration/exploitation weight. Between 0.0 and 1.0. Higher values
             indicate higher.
        """
        if explore_randomly:
            sampler = optuna.samplers.RandomSampler()
        else:
            # Configure the TPESampler with exploration/exploitation tradeoff
            sampler = optuna.samplers.TPESampler(consider_prior=True,
                                                 prior_weight=exploration_weight,
                                                 n_startup_trials=int(exploration_weight * 10))

        # Create a study
        self.study = optuna.create_study(direction='minimize',
                                         study_name="voice_model_study",
                                         sampler=sampler)

    def try_load_study(self):
        if os.path.isfile(self.save_filename):
            with open(self.save_filename, "rb") as file_in:
                self.study = pickle.load(file_in)
            return True
        else:
            return False

    def save_study(self):
        with open(self.save_filename, "wb") as file_out:
            pickle.dump(self.study, file_out)

    def optimize(self, n_trials):
        for i in range(1, n_trials + 1):
            self.perform_trial()
            self.print_trial(self.last_trial, "This trial:")
            self.print_best_trial()

    def perform_trial(self):
        self.study.optimize(self.objective, n_trials=1)  # Invoke optimization of the objective function.
        self.save_study()

    def objective(self, trial):
        # Current trial number
        trial_number = trial.number
        print()
        print(f"Starting Trial {trial_number}")

        score = self.objective_function(trial)
        self.last_trial = trial
        self.last_trial.value = score
        return score

    def print_trial(self, trial, title_string):
        if trial is None:
            return
        print(title_string)
        print("  Number: ", trial.number)
        print("  Value:      ", trial.value)
        if hasattr(trial, 'validation_value'):
            print("  Validation: ", trial.validation_value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print()

    def print_best_trial(self):
        self.print_trial(self.study.best_trial, title_string="Best trial:")

    def print_all_trials(self):
        for trial in self.study.trials:
            print(f"Trial {trial.number}: Params = {trial.params}")

    def print_trial_param_counts(self):
        # Dictionary to store counts of each parameter value
        param_counts = {}

        for trial in self.study.trials:
            for param, value in trial.params.items():
                param_value_key = f"{param}({value})"
                if param_value_key in param_counts:
                    param_counts[param_value_key] += 1
                else:
                    param_counts[param_value_key] = 1

        # Sort the keys (parameter-value pairs) before printing
        sorted_param_keys = sorted(param_counts.keys())

        # Print the counts for each sorted parameter value
        for param_value_key in sorted_param_keys:
            print(f"{param_value_key}: {param_counts[param_value_key]}")

    def show_visualizations(self):
        figure = optuna.visualization.plot_contour(self.study)
        figure.show()
        figure = optuna.visualization.plot_param_importances(self.study)
        figure.show()
        figure = optuna.visualization.plot_edf([self.study])
        figure.show()
        figure = optuna.visualization.plot_intermediate_values(self.study)
        figure.show()
        figure = optuna.visualization.plot_optimization_history(self.study)
        figure.show()
        figure = optuna.visualization.plot_parallel_coordinate(self.study)
        figure.show()
        # figure = optuna.visualization.plot_pareto_front(study)  # `plot_pareto_front` function only supports 2 or 3 objective studies.
        # figure.show()
        figure = optuna.visualization.plot_slice(self.study)
        figure.show()
        figure = optuna.visualization.plot_slice(self.study)
        figure.update_layout(yaxis_range=[0, 1])  # Set y-axis range here
        figure.show()

    def show_pca_graphs(self, n_components=2):
        # Extract hyperparameters and objective values
        trial_params = [trial.params for trial in self.study.trials]
        objective_values = [trial.value for trial in self.study.trials]

        # Create a DataFrame from the hyperparameters
        df = pd.DataFrame(trial_params)

        # Encode categorical variables if necessary
        df = pd.get_dummies(df)

        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Apply PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_scaled)

        # Plot the results
        plt.figure(figsize=(8, 6))
        if n_components == 2:
            plt.scatter(principal_components[:, 0], principal_components[:, 1], c=objective_values, cmap='viridis')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
        elif n_components == 3:
            ax = plt.axes(projection='3d')
            ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=objective_values, cmap='viridis')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
        plt.colorbar(label='Objective Value')
        plt.title('Hyperparameter Optimization Results')
        plt.show()
