import numpy as np
import pandas as pd
from functools import partial, reduce
from typing import List, Tuple, Callable
import math 

# Type aliases for clarity
str = str
float = float
int = int

# Constants
CLASS_LABELS: List[str] = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
ALPHA: float = 1.0

def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep="\t", header=None, quoting=3)

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[2].values
    y = np.array([CLASS_LABELS.index(label) for label in df[1]])
    return X, y

def calculate_class_probs(y: np.ndarray) -> np.ndarray:
    n_samples = len(y)
    n_classes = len(CLASS_LABELS)
    class_counts = np.bincount(y, minlength=n_classes)
    return np.log((class_counts + ALPHA) / (n_samples + n_classes * ALPHA))

def calculate_score(counts: List[int], class_probs: np.ndarray, true_prob: float, class_index: int) -> float:
    if class_index == 5:  # 'true' class
        return true_prob
    else:
        count_sum = sum(counts)
        term1 = np.log(1 - np.exp(true_prob))
        term2 = np.log((counts[class_index] + ALPHA) / (count_sum + 6 * ALPHA))
        return term1 + term2

def predict_single(counts: List[int], class_probs: np.ndarray) -> str:
    true_prob = class_probs[-1]
    scores = [calculate_score(counts, class_probs, true_prob, i) for i in range(len(CLASS_LABELS))]
    return CLASS_LABELS[np.argmax(scores)]

def predict(X: np.ndarray, count_data: List[np.ndarray], class_probs: np.ndarray) -> List[str]:
    predict_func = partial(predict_single, class_probs=class_probs)
    return list(map(predict_func, zip(*count_data)))

def calculate_accuracy(predictions: List[str], y_true: np.ndarray) -> float:
    correct = sum(1 for pred, true in zip(predictions, y_true) if CLASS_LABELS[true] == pred)
    return correct / len(predictions)

def count_misclassifications(predictions: List[str], y_true: np.ndarray) -> Tuple[dict, dict]:
    misclassified = [(pred, CLASS_LABELS[true]) for pred, true in zip(predictions, y_true) if pred != CLASS_LABELS[true]]
    predicted_as = {label: sum(1 for p, _ in misclassified if p == label) for label in CLASS_LABELS}
    misclassified_as = {label: sum(1 for _, t in misclassified if t == label) for label in CLASS_LABELS}
    return predicted_as, misclassified_as

def calculate_prediction_fractions(predictions: List[str]) -> dict:
    total = len(predictions)
    return {label: predictions.count(label) / total for label in CLASS_LABELS}

def write_predictions(predictions: List[str], output_file: str) -> None:
    with open(output_file, 'w') as f:
        f.write('\n'.join(predictions))

def main(train_file: str, test_file: str, out_file: str) -> None:
    # Prepare data
    X_train, y_train = prepare_data(read_data(train_file))
    X_test, y_test = prepare_data(read_data(test_file))
    
    # Train model
    class_probs = calculate_class_probs(y_train)
    
    # Prepare count data for prediction
    test_data = read_data(test_file)
    count_data = [test_data[12], test_data[9], test_data[8], test_data[10], test_data[11]]
    
    # Make predictions
    predictions = predict(X_test, count_data, class_probs)
    
    # Write predictions to file
    write_predictions(predictions, out_file)
    
    # Calculate and print metrics
    accuracy = calculate_accuracy(predictions, y_test)
    predicted_as, misclassified_as = count_misclassifications(predictions, y_test)
    prediction_fractions = calculate_prediction_fractions(predictions)
    
    print(f"Test Accuracy: {accuracy:.9f}")
    print("\nMisclassification Statistics:")
    print("Predicted as (misclassified):")
    for label, count in predicted_as.items():
        print(f"{label}: {count} times")
    print("\nMisclassified as:")
    for label, count in misclassified_as.items():
        print(f"{label}: {count} times")
    print("\nPrediction Fractions for Each Class:")
    for label, fraction in prediction_fractions.items():
        print(f"{label}: {fraction:.6f}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()
    
    main(args.train, args.test, args.out)
