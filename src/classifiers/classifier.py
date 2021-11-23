
class Classifier:
    def __init__(self):
        # Initialize any internal variables in this function
        print("Unimplemented")

    def preprocess(self, data, training=False):
        # If training is true:
        #   Fit scalers if needed
        #   Identify features to drop
        #   etc.
        #
        # Preprocess data (no difference between training/testing here)
        print("Unimplemented")

    def train(self, data):
        # Preprocess data with training=True
        # Train the model
        print("Unimplemented")

    def predict(self, data):
        # Preprocess data
        # Predict y based on given data
        print("Unimplemented")

    @staticmethod
    def hyperparameter_tuning(param_grid):
        pass