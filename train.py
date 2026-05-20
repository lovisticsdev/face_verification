import numpy as np
from joblib import load, dump, Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from skimage.transform import rotate
from scipy.ndimage import gaussian_filter

def plot_validation_scores(scores_dict, save_path):
    """Plot validation scores across different PCA components."""
    components = list(scores_dict.keys())
    scores = list(scores_dict.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(components, scores, marker='o', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance Across PCA Components')
    
    # Add value labels on points
    for i, score in enumerate(scores):
        plt.annotate(f'{score:.3f}', 
                    (components[i], scores[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Different', 'Same'],
                yticklabels=['Different', 'Same'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def normalize_images(X):
    """Apply basic normalization to improve feature quality."""
    X = X.reshape(-1, 62 * 47)
    X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return X_norm

def process_single_pair(img_pair, y_val):
    """Process a single pair of images for augmentation."""
    results = []
    labels = []
    
    # Original pair
    results.append(img_pair.flatten())
    labels.append(y_val)
    
    if y_val == 1:
        # Rotation only for positive pairs
        for angle in [-4, 4]:
            pair_rotated = []
            for img in img_pair:
                img_2d = img.reshape(62, 47)
                rotated = rotate(img_2d, angle, mode='reflect')
                pair_rotated.append(rotated.flatten())
            results.append(np.concatenate(pair_rotated))
            labels.append(y_val)
        
        # Single blur level
        pair_blurred = []
        for img in img_pair:
            img_2d = img.reshape(62, 47)
            blurred = gaussian_filter(img_2d, sigma=0.4)
            pair_blurred.append(blurred.flatten())
        results.append(np.concatenate(pair_blurred))
        labels.append(y_val)
    
    return results, labels

def augment_data(X, y):
    """Augment training data with parallel processing."""
    X = X.reshape(-1, 2, 62 * 47)
    
    # Process pairs in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_single_pair)(X[i], y[i]) 
        for i in range(len(X))
    )
    
    X_augmented = []
    y_augmented = []
    
    for x_list, y_list in results:
        X_augmented.extend(x_list)
        y_augmented.extend(y_list)
    
    return np.array(X_augmented), np.array(y_augmented)

def find_best_parameters(X, y):
    """Find best parameters using cross-validation."""
    # Create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    best_score = 0
    best_config = None
    validation_scores = {}
    
    # Test different PCA components
    for n_components in [96, 112]:
        print(f"\nTesting PCA components: {n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        # Grid search for SVM
        param_grid = {
            'C': [15.0, 20.0],
            'gamma': ['scale', 0.0001],
            'class_weight': ['balanced'],
            'probability': [True]
        }
        
        svm = SVC(kernel='rbf', random_state=42)
        grid_search = GridSearchCV(
            svm,
            param_grid,
            scoring='balanced_accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_pca, y_train)
        val_score = balanced_accuracy_score(y_val, grid_search.predict(X_val_pca))
        validation_scores[n_components] = val_score
        
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        print(f"Validation score: {val_score:.3f}")
        print(f"Parameters: {grid_search.best_params_}")
        
        if val_score > best_score:
            best_score = val_score
            best_config = {
                'n_components': n_components,
                'svm_params': grid_search.best_params_
            }
    
    # Plot validation scores
    plot_validation_scores(validation_scores, 'validation_scores.png')
    
    print("\nBest configuration:")
    print(f"PCA components: {best_config['n_components']}")
    print(f"SVM parameters: {best_config['svm_params']}")
    print(f"Best validation score: {best_score:.3f}")
    
    return best_config

def train_final_model(X, y, config):
    """Train final model using the best configuration."""
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=config['n_components'], random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Create and fit final model
    model = SVC(
        kernel='rbf',
        random_state=42,
        **config['svm_params']
    )
    model.fit(X_train_pca, y_train)
    
    # Generate confusion matrix
    y_pred = model.predict(X_test_pca)
    plot_confusion_matrix(y_test, y_pred, 'confusion_matrix.png')
    
    # Calculate and print final accuracy
    accuracy = balanced_accuracy_score(y_test, y_pred) * 100
    print(f"\nFinal balanced accuracy: {accuracy:.2f}%")
    
    return {
        'model': model,
        'scaler': scaler,
        'pca': pca
    }

def main():
    parser = argparse.ArgumentParser(description='Train face verification model')
    parser.add_argument('training_file', help='Path to training data file')
    parser.add_argument('model_file', help='Path to save trained model')
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading training data...")
    data = load(args.training_file)
    X = normalize_images(data['data'] / 255.0)
    y = data['target']
    
    # Augment data
    print("Augmenting training data...")
    X_aug, y_aug = augment_data(X, y)
    print(f"Original data shape: {X.shape}")
    print(f"Augmented data shape: {X_aug.shape}")
    
    # Find best parameters
    print("\nFinding best parameters...")
    best_config = find_best_parameters(X_aug, y_aug)
    
    # Train final model
    print("\nTraining final model...")
    model_components = train_final_model(X_aug, y_aug, best_config)
    
    print(f"\nSaving model to {args.model_file}...")
    dump(model_components, args.model_file)
    print("Done!")

if __name__ == "__main__":
    main()