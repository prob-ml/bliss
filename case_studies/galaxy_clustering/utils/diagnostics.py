import torch


def create_overlay(predictions, tile_size):
    """Creates prediction image to overlay on raw image.

    Args:
        predictions: raw predictions (of shape n_tiles x n_tiles)
        tile_size: tile size to create overlay

    Returns:
        overlay: predictions to overlay (green for true, red for false)
    """
    overlay = torch.zeros((predictions.size(0) * tile_size, predictions.size(1) * tile_size, 3))
    detection_color = torch.tensor([0.0, 1.0, 0.0])  # Green color
    non_detection_color = torch.tensor([1.0, 0.0, 0.0])  # Red color
    for i in range(predictions.size(0)):
        for j in range(predictions.size(1)):
            color = detection_color if predictions[i, j] == 1 else non_detection_color
            overlay[
                i * tile_size : (i + 1) * tile_size,
                j * tile_size : (j + 1) * tile_size,
            ] = color
    return overlay


def blend_images(original, overlay, alpha=0.5, img_crop=0):
    """Blends raw image with predictions overlay.

    Args:
        original: original raw image
        overlay: predictions image to overlay
        alpha: ratio of blending
        img_crop: size of image to crop out (based on tiles to crop)

    Returns:
        blended: blended image
    """
    # Ensure the original image is in float
    if original.max() > 1.0:
        original = original / 255.0
    # Blend the images
    blended = original * (1 - alpha) + overlay * alpha
    return blended[img_crop : blended.size(0) - img_crop, img_crop : blended.size(1) - img_crop]


def compute_metrics(est_cat, true_cat):
    """Computes metrics for arbitrary catalogs.

    Args:
        est_cat: predicted or estimated catalog
        true_cat: ground truth catalog

    Returns:
        accuracy, precision, recall, f1
    """
    true_positives = (est_cat & true_cat).sum(dtype=torch.float32)
    true_positives = (est_cat & true_cat).sum(dtype=torch.float32)
    false_positives = (est_cat & ~true_cat).sum(dtype=torch.float32)
    false_negatives = (~est_cat & true_cat).sum(dtype=torch.float32)
    true_negatives = (~est_cat & ~true_cat).sum(dtype=torch.float32)

    accuracy = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
    )
    precision = true_positives / (true_positives + false_positives + 1e-6)  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return accuracy.item(), precision.item(), recall.item(), f1.item()
