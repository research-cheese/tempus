import numpy as np

from sklearn.metrics import classification_report, jaccard_score

class SegmentationMetrics:
    accuracy: float
    
    precision_foreground: float
    recall_foreground: float
    f1_foreground: float

    precision_background: float
    recall_background: float
    f1_background: float

    dice_score: float
    jaccard_score: float

def segmentation_classification_report(ground_truth, predicted):
    """
    Calculates the precision, recall, and F1 score for the predicted segmentation.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    flattened_ground_truth = ground_truth.flatten()
    flattened_predicted = predicted.flatten()
    return classification_report(flattened_ground_truth, flattened_predicted, labels=[0, 1], target_names=['background', 'foreground'], output_dict=True)

def segmentation_dice_score(ground_truth, predicted):
    """
    Calculates the Dice score for the predicted segmentation.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    intersection = np.logical_and(ground_truth, predicted)
    dice_score = (2*np.sum(intersection))/(np.sum(ground_truth)+np.sum(predicted))
    return dice_score

def segmentation_jaccard_score(ground_truth, predicted):
    """
    Calculates the Jaccard score for the predicted segmentation.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    flattened_ground_truth = ground_truth.flatten()
    flattened_predicted = predicted.flatten()

    jaccard = jaccard_score(flattened_ground_truth, flattened_predicted)
    return jaccard

def get_segmentation_metrics(ground_truth, predicted) -> SegmentationMetrics:
    """
    Returns a dictionary of segmentation metrics.

    ground_truth: 2D numpy array of 0s and 1s
    predicted: 2D numpy array of 0s and 1s
    """
    classification_report = segmentation_classification_report(ground_truth, predicted)
    dice_score = segmentation_dice_score(ground_truth, predicted)
    jaccard_score = segmentation_jaccard_score(ground_truth, predicted)

    segmentation_metrics = SegmentationMetrics()
    segmentation_metrics.accuracy = classification_report["accuracy"]
    segmentation_metrics.precision_background = classification_report["background"]["precision"]
    segmentation_metrics.recall_background = classification_report["background"]["recall"]
    segmentation_metrics.f1_background = classification_report["background"]["f1-score"]
    segmentation_metrics.precision_foreground = classification_report["foreground"]["precision"]
    segmentation_metrics.recall_foreground = classification_report["foreground"]["recall"]
    segmentation_metrics.f1_foreground = classification_report["foreground"]["f1-score"]
    segmentation_metrics.dice_score = dice_score
    segmentation_metrics.jaccard_score = jaccard_score

    return segmentation_metrics