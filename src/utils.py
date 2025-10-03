from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_precision_recall_curve(model, test_dataloader, test_labels, model_type='fc', model_name='Autoencoder'):
    """
    Plot precision-recall curve for autoencoder anomaly detection
    """
    # Get reconstruction errors (these will be our "scores")
    test_errors = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            mel_specs, _, _ = batch
            
            if model_type == 'fc':
                x = mel_specs.squeeze(1).view(mel_specs.size(0), -1)
                x = x.to(next(model.parameters()).device)
                recon = model(x)
                errors = torch.mean((x - recon) ** 2, dim=1)
                
            elif model_type == 'conv':
                x = mel_specs.to(next(model.parameters()).device)
                recon = model(x)
                errors = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
                
            elif model_type == 'lstm':
                x = mel_specs.squeeze(1).permute(0, 2, 1).to(next(model.parameters()).device)
                recon = model(x)
                errors = torch.mean((x - recon) ** 2, dim=(1, 2))
            
            test_errors.extend(errors.cpu().numpy())
    
    # Filter valid labels
    test_errors = np.array(test_errors)
    valid_idx = test_labels != -1
    test_labels_valid = test_labels[valid_idx]
    test_errors_valid = test_errors[valid_idx]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(test_labels_valid, test_errors_valid)
    roc_auc = roc_auc_score(test_labels_valid, test_errors)
    avg_precision = average_precision_score(test_labels_valid, test_errors_valid)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return precision, recall, thresholds, avg_precision, roc_auc

def find_optimal_threshold_from_pr(precision, recall, thresholds, metric='f1', beta=2.0):
    """
    Find optimal threshold based on different metrics
    """
    if metric == 'f1':
        epsilon = 1e-8
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + epsilon)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = f1_scores[optimal_idx]
        print(f"Optimal threshold (F1): {optimal_threshold:.6f}, F1 Score: {optimal_score:.3f}")
    
    elif metric == 'fbeta':
        epsilon = 1e-8
        # F-beta score: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
        beta_squared = beta ** 2
        fbeta_scores = (1 + beta_squared) * (precision[:-1] * recall[:-1]) / (
            (beta_squared * precision[:-1]) + recall[:-1] + epsilon
        )
        optimal_idx = np.argmax(fbeta_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = fbeta_scores[optimal_idx]
        print(f"Optimal threshold (F{beta}-Score): {optimal_threshold:.6f}, F{beta} Score: {optimal_score:.3f}")
        if beta < 1:
            print(f"  (Beta={beta} emphasizes Precision)")
        elif beta > 1:
            print(f"  (Beta={beta} emphasizes Recall)")
        
    elif metric == 'balanced':
        # Balance precision and recall
        balanced_scores = (precision[:-1] + recall[:-1]) / 2
        optimal_idx = np.argmax(balanced_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = balanced_scores[optimal_idx]
        print(f"Optimal threshold (Balanced): {optimal_threshold:.6f}, Balanced Score: {optimal_score:.3f}")
    
    elif metric == 'intersection' or metric == 'intercept':
        # Find where precision and recall intersect (are closest to equal)
        diff = np.abs(precision[:-1] - recall[:-1])
        optimal_idx = np.argmin(diff)
        optimal_threshold = thresholds[optimal_idx]
        precision_at_intercept = precision[optimal_idx]
        recall_at_intercept = recall[optimal_idx]
        print(f"Optimal threshold (Precision-Recall Intersection): {optimal_threshold:.6f}")
        print(f"Precision: {precision_at_intercept:.3f}, Recall: {recall_at_intercept:.3f}")
        optimal_score = (precision_at_intercept + recall_at_intercept) / 2
    
    
    return optimal_threshold, optimal_idx

def plot_precision_recall_vs_threshold(model, test_dataloader, test_labels, model_type='fc', model_name='Autoencoder'):
    """
    Plot precision and recall vs threshold values
    """
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    # Get reconstruction errors (these will be our "scores")
    test_errors = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            mel_specs, _, _ = batch
            
            if model_type == 'fc':
                x = mel_specs.squeeze(1).view(mel_specs.size(0), -1)
                x = x.to(next(model.parameters()).device)
                recon = model(x)
                errors = torch.mean((x - recon) ** 2, dim=1)
                
            elif model_type == 'conv':
                x = mel_specs.to(next(model.parameters()).device)
                recon = model(x)
                errors = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
                
            elif model_type == 'lstm':
                x = mel_specs.squeeze(1).permute(0, 2, 1).to(next(model.parameters()).device)
                recon = model(x)
                errors = torch.mean((x - recon) ** 2, dim=(1, 2))
            
            test_errors.extend(errors.cpu().numpy())
    
    # Filter valid labels
    test_errors = np.array(test_errors)
    valid_idx = test_labels != -1
    test_labels_valid = test_labels[valid_idx]
    test_errors_valid = test_errors[valid_idx]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(test_labels_valid, test_errors_valid)
    
    # Plot precision and recall vs threshold
    plt.figure(figsize=(10, 6))
    
    # Plot precision vs threshold
    plt.plot(thresholds, precision[:-1], 'b-', linewidth=2, label='Precision')
    # Plot recall vs threshold  
    plt.plot(thresholds, recall[:-1], 'r-', linewidth=2, label='Recall')
    
    # Add F1 score vs threshold
    
    epsilon = 1e-8
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + epsilon)
    
    # Handle NaN values in F1 scores
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)
    
    # DEBUG: Check F1 scores
    print(f"F1 scores range: [{f1_scores.min():.3f}, {f1_scores.max():.3f}]")
    print(f"Valid F1 scores (>0): {np.sum(f1_scores > 0)}/{len(f1_scores)}")
    
    plt.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1 Score')
    
    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    
    
    # Mark optimal point
    plt.axvline(x=optimal_threshold, color='orange', linestyle='--', alpha=0.7, 
                label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.scatter(optimal_threshold, optimal_f1, color='orange', s=100, zorder=5)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision, Recall, and F1 vs Threshold - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(thresholds.min(), thresholds.max())
    plt.ylim(0, 1.05)
    
    # Add text annotation for optimal point
    plt.annotate(f'Optimal F1: {optimal_f1:.3f}', 
                xy=(optimal_threshold, optimal_f1), 
                xytext=(optimal_threshold + (thresholds.max()-thresholds.min())*0.1, optimal_f1 + 0.05),
                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                fontsize=10, color='orange')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal threshold: {optimal_threshold:.6f}")
    print(f"At optimal threshold - Precision: {precision[optimal_idx]:.3f}, Recall: {recall[optimal_idx]:.3f}, F1: {optimal_f1:.3f}")
    
    return precision, recall, thresholds, optimal_threshold