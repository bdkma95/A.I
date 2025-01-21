import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file and return a tuple (evidence, labels).
    
    Each element in evidence is a list of 17 numeric features.
    Labels are integers (1 if the user made a purchase, 0 otherwise).
    
    Arguments:
        filename (str): Path to the CSV file.
    
    Returns:
        tuple: A tuple containing a list of evidence and a list of labels.
    """
    evidence = []
    labels = []

    # Open the CSV file
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Iterate through each row of the CSV file
        for row in reader:
            # Extract evidence as a list of 17 features
            evidence.append([
                int(row["Administrative"]),                       # Administrative
                float(row["Administrative_Duration"]),           # Administrative_Duration
                int(row["Informational"]),                       # Informational
                float(row["Informational_Duration"]),            # Informational_Duration
                int(row["ProductRelated"]),                      # ProductRelated
                float(row["ProductRelated_Duration"]),           # ProductRelated_Duration
                float(row["BounceRates"]),                       # BounceRates
                float(row["ExitRates"]),                         # ExitRates
                float(row["PageValues"]),                        # PageValues
                float(row["SpecialDay"]),                        # SpecialDay
                month_to_index(row["Month"]),                    # Month (converted to integer)
                int(row["OperatingSystems"]),                    # OperatingSystems
                int(row["Browser"]),                             # Browser
                int(row["Region"]),                              # Region
                int(row["TrafficType"]),                         # TrafficType
                int(row["VisitorType"] == "Returning"),          # VisitorType (1 if returning, 0 otherwise)
                int(row["Weekend"] == "TRUE")                    # Weekend (1 if TRUE, 0 otherwise)
            ])
            
            # Extract label (1 if Revenue is TRUE, 0 otherwise)
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return evidence, labels


def month_to_index(month):
    """
    Convert month name to an integer index (0-11).
    
    Arguments:
        month (str): Month name.
    
    Returns:
        int: Index of the month (0 for January, ..., 11 for December).
    """
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return months.index(month)


def month_to_index(month):
    """Convert month name or abbreviation to index (0-11)."""
    months_full = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Check if the month is a full name or an abbreviation
    if month in months_full:
        return months_full.index(month)
    elif month in months_abbr:
        return months_abbr.index(month)
    else:
        raise ValueError(f"Invalid month: {month}")


from sklearn.neighbors import KNeighborsClassifier

def train_model(evidence, labels):
    """
    Train a k-nearest-neighbor classifier (k=1) on the given evidence and labels.

    Arguments:
        evidence (list): A list of evidence lists, where each inner list contains numeric feature values.
        labels (list): A list of labels corresponding to each piece of evidence.

    Returns:
        KNeighborsClassifier: A fitted k-nearest-neighbor classifier.
    """
    # Initialize the KNeighborsClassifier with k=1
    model = KNeighborsClassifier(n_neighbors=1)
    
    # Fit the model using the provided evidence and labels
    model.fit(evidence, labels)
    
    return model


import numpy as np

def evaluate(labels, predictions):
    """
    Evaluate the performance of a classifier by calculating sensitivity and specificity.

    Arguments:
        labels (list): The true labels (0 or 1) for the testing set.
        predictions (list): The predicted labels (0 or 1) by the classifier.

    Returns:
        tuple: A tuple (sensitivity, specificity).
               sensitivity (float): True positive rate.
               specificity (float): True negative rate.
    """
    # Convert lists to numpy arrays for easier element-wise operations
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
    true_positive = np.sum((labels == 1) & (predictions == 1))
    true_negative = np.sum((labels == 0) & (predictions == 0))
    false_positive = np.sum((labels == 0) & (predictions == 1))
    false_negative = np.sum((labels == 1) & (predictions == 0))

    # Sensitivity: True Positive Rate = TP / (TP + FN)
    sensitivity = true_positive / (true_positive + false_negative)

    # Specificity: True Negative Rate = TN / (TN + FP)
    specificity = true_negative / (true_negative + false_positive)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
