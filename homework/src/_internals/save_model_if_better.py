# save_model_if_better.py
import os
import pickle

from homework.src._internals.compare_models import compare_models
from homework.src._internals.save_model import save_model


def save_model_if_better(model, x_test, y_test, save_path="models/estimator.pkl"):
    """Save the model if it performs better than the existing one."""
    best_model = None
    if os.path.exists(save_path):
        with open(save_path, "rb") as file:
            best_model = pickle.load(file)
    best_model = compare_models(model, best_model, x_test, y_test)
    save_model(best_model, save_path)


def compare_models(current_model, best_model, x_test, y_test):
    """Compare the current model with the best model and return the better one."""
    if best_model is None or current_model.score(x_test, y_test) > best_model.score(
        x_test, y_test
    ):
        return current_model
    return best_model
