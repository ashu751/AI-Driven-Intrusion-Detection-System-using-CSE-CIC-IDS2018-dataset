import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from joblib import dump, load
import time
import warnings
import gc  # For garbage collection
import matplotlib.ticker as mtick
from sklearn.calibration import calibration_curve
from collections import defaultdict
warnings.filterwarnings('ignore')

# Define the dataset directory
dataset_dir = r"D:\ML1.12"

# Function to load and preprocess the datasets in chunks
def load_and_preprocess_data(directory, sample_size=None):
    print("Loading and preprocessing data in chunks...")
    all_data = []
    
    # List all csv files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files.")
    
    # Define chunk size
    CHUNK_SIZE = 50000  # Adjusted chunk size
    
    # Process each CSV file in chunks
    for file in csv_files:
        print(f"Processing file: {file}")
        file_path = os.path.join(directory, file)
        
        try:
            # Initialize a counter for chunks
            chunk_count = 0
            file_data = []
            
            # Read and process the CSV file in chunks
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
                chunk_count += 1
                print(f"  - Processing chunk {chunk_count} with shape {chunk.shape}")
                
                # Clean the dataset
                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                chunk = chunk.dropna()
                
                # IMPORTANT FIX: Ensure 'Label' column exists and has both BENIGN and attack samples
                if 'Label' in chunk.columns:
                    # Print the original label values to verify
                    label_counts = chunk['Label'].value_counts()
                    print(f"  - Original label distribution: {label_counts.to_dict()}")
                    
                    # Correctly identify BENIGN traffic
                    benign_patterns = ['BENIGN', 'benign', 'normal', 'Normal']
                    
                    # Check if 'BENIGN' or related values exist in the labels
                    has_benign = any(pattern.lower() in ''.join(chunk['Label'].astype(str).values).lower() for pattern in benign_patterns)
                    
                    if not has_benign:
                        print("WARNING: No BENIGN traffic found in this chunk. Sampling might be biased.")
                    
                    # Binary classification: Normal vs Attack with verbose logging
                    chunk['Label'] = chunk['Label'].apply(lambda x: 0 if any(pattern.lower() in str(x).lower() for pattern in benign_patterns) else 1)
                    print(f"  - After conversion, class distribution: {chunk['Label'].value_counts().to_dict()}")
                else:
                    print(f"  - Warning: 'Label' column not found in {file}, skipping...")
                    continue
                
                # Convert Protocol column to numeric if it exists and is not already numeric
                if 'Protocol' in chunk.columns and chunk['Protocol'].dtype == object:
                    # Map protocol names to numbers
                    protocol_map = {'TCP': 6, 'UDP': 17, 'ICMP': 1}
                    chunk['Protocol'] = chunk['Protocol'].map(protocol_map).fillna(0).astype(int)
                
                # Take a sample from each chunk to manage memory
                if sample_size and len(chunk) > sample_size // len(csv_files) // 10:
                    # IMPORTANT FIX: Stratified sampling to maintain class distribution
                    try:
                        # Get stratified sample if both classes exist
                        if len(chunk['Label'].unique()) > 1:
                            sample_chunk = chunk.groupby('Label', group_keys=False).apply(
                                lambda x: x.sample(min(len(x), max(1, sample_size // len(csv_files) // 20)), random_state=42)
                            )
                        else:
                            sample_chunk = chunk.sample(n=min(len(chunk), sample_size // len(csv_files) // 10), random_state=42)
                        file_data.append(sample_chunk)
                    except Exception as e:
                        print(f"  - Error during sampling: {e}. Taking simple random sample.")
                        sample_chunk = chunk.sample(n=min(len(chunk), sample_size // len(csv_files) // 10), random_state=42)
                        file_data.append(sample_chunk)
                else:
                    file_data.append(chunk)
                
                # Force garbage collection
                gc.collect()
                
                # Limit the number of chunks per file to avoid memory issues
                if sample_size and chunk_count * CHUNK_SIZE >= sample_size // len(csv_files):
                    break
            
            # Combine all chunks for this file
            if file_data:
                file_df = pd.concat(file_data, ignore_index=True)
                print(f"  - File {file} processed with final shape {file_df.shape}")
                all_data.append(file_df)
                del file_data, file_df  # Free memory
                gc.collect()
            
        except Exception as e:
            print(f"  - Error processing {file}: {e}")
    
    # Combine all dataframes
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_data.shape}")
        
        # IMPORTANT FIX: Check if we have both classes in the final dataset
        class_distribution = combined_data['Label'].value_counts()
        print(f"Final class distribution: {class_distribution.to_dict()}")
        
        # If we're missing a class, try to create synthetic samples
        if len(class_distribution) < 2:
            missing_class = 0 if 1 in class_distribution.index else 1
            print(f"WARNING: Only one class ({1-missing_class}) present in the data.")
            print(f"Creating synthetic samples for class {missing_class}...")
            
            # Get a subset of samples to modify
            samples_to_modify = combined_data.sample(min(1000, len(combined_data)//10))
            
            # Modify some key fields to create synthetic samples of the missing class
            if missing_class == 0:  # Create synthetic benign traffic
                samples_to_modify['Label'] = 0
                samples_to_modify['Flow Duration'] = samples_to_modify['Flow Duration'] * 0.5
                samples_to_modify['Tot Fwd Pkts'] = samples_to_modify['Tot Fwd Pkts'] * 0.7
                samples_to_modify['Fwd Pkts/s'] = samples_to_modify['Fwd Pkts/s'] * 0.7
                samples_to_modify['SYN Flag Cnt'] = 1
                samples_to_modify['FIN Flag Cnt'] = 1
            else:  # Create synthetic attack traffic
                samples_to_modify['Label'] = 1
                samples_to_modify['Flow Duration'] = samples_to_modify['Flow Duration'] * 1.5
                samples_to_modify['Tot Fwd Pkts'] = samples_to_modify['Tot Fwd Pkts'] * 1.3
                samples_to_modify['SYN Flag Cnt'] = samples_to_modify['SYN Flag Cnt'] + 2
            
            # Add these synthetic samples to our dataset
            combined_data = pd.concat([combined_data, samples_to_modify], ignore_index=True)
            print(f"After adding synthetic samples, class distribution: {combined_data['Label'].value_counts().to_dict()}")
        
        # Take a sample from the combined data if needed
        if sample_size and len(combined_data) > sample_size:
            # Stratified sampling to maintain class distribution
            final_data = combined_data.groupby('Label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, sample_size * len(x) // len(combined_data))), random_state=42)
            )
            del combined_data, all_data  # Free memory
            gc.collect()
        else:
            final_data = combined_data
            del all_data  # Free memory
            gc.collect()
        
        print(f"Final data shape: {final_data.shape}")
        print(f"Final class distribution: {final_data['Label'].value_counts().to_dict()}")
        return final_data
    else:
        raise Exception("No valid data files were processed.")

# Function to perform feature selection and engineering
def feature_engineering(df):
    print("Performing feature engineering...")
    
    # First, check and convert data types to ensure numeric processing
    for col in df.columns:
        if col != 'Label':  # Skip the target column
            if df[col].dtype == object:  # If column is string/object type
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN values with column median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                except:
                    # If conversion fails, drop the column
                    print(f"  - Dropping non-numeric column: {col}")
                    df = df.drop(col, axis=1)
    
    # Drop irrelevant columns if they exist
    cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Print data types after conversion
    print("Data types after conversion:")
    print(df.dtypes.value_counts())
    
    # Check for remaining object columns and drop them
    object_cols = df.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        print(f"Dropping remaining object columns: {list(object_cols)}")
        df = df.drop(object_cols, axis=1)
    
    # Keep only a subset of features to reduce dimensionality
    # This list contains common network traffic features
    important_features = [
        'Dst Port', 'Protocol', 'Flow Duration', 
        'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
        'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
        'Flow IAT Mean', 'Flow IAT Std',
        'Fwd IAT Tot', 'Bwd IAT Tot',
        'Fwd Header Len', 'Bwd Header Len',
        'Fwd Pkts/s', 'Bwd Pkts/s',
        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
        'FIN Flag Cnt', 'SYN Flag Cnt', 
        'RST Flag Cnt', 'PSH Flag Cnt',
        'ACK Flag Cnt', 'URG Flag Cnt',
        'Down/Up Ratio', 'Pkt Size Avg',
        'Label'
    ]
    
    # Keep only the important features that exist in the dataframe
    available_features = [col for col in important_features if col in df.columns]
    if available_features:
        df = df[available_features]
        print(f"Selected {len(available_features)} features")
    else:
        # If none of our predefined features are available, keep all numeric columns
        print("None of the predefined features found. Using all numeric columns.")
    
    # Create derived features to help the model
    print("Creating derived features to help with classification...")
    try:
        # Ratio features often help with detection
        if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
            df['Fwd_Bwd_Ratio'] = df['Tot Fwd Pkts'] / (df['Tot Bwd Pkts'] + 1)
        
        if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
            df['Fwd_Bwd_Bytes_Ratio'] = df['TotLen Fwd Pkts'] / (df['TotLen Bwd Pkts'] + 1)
        
        if 'SYN Flag Cnt' in df.columns and 'ACK Flag Cnt' in df.columns:
            df['SYN_ACK_Ratio'] = df['SYN Flag Cnt'] / (df['ACK Flag Cnt'] + 1)
        
        # Flag combinations can indicate certain attacks
        if all(flag in df.columns for flag in ['SYN Flag Cnt', 'RST Flag Cnt', 'FIN Flag Cnt']):
            df['Unusual_Flag_Combo'] = ((df['SYN Flag Cnt'] > 0) & (df['RST Flag Cnt'] > 0)).astype(int)
            df['SYN_FIN_Combo'] = ((df['SYN Flag Cnt'] > 0) & (df['FIN Flag Cnt'] > 0)).astype(int)
        
        # Advanced packet statistics
        if 'Tot Fwd Pkts' in df.columns and 'Flow Duration' in df.columns:
            # Packets per second (if not already present)
            if 'Fwd Pkts/s' not in df.columns:
                df['Fwd Pkts/s'] = df['Tot Fwd Pkts'] / (df['Flow Duration'] / 1000000 + 0.001)
        
        if 'Tot Bwd Pkts' in df.columns and 'Flow Duration' in df.columns:
            # Backward packets per second (if not already present)
            if 'Bwd Pkts/s' not in df.columns:
                df['Bwd Pkts/s'] = df['Tot Bwd Pkts'] / (df['Flow Duration'] / 1000000 + 0.001)
        
        # Packet size statistics
        if all(col in df.columns for col in ['TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts']):
            # Average packet size calculation
            df['Avg_Fwd_Segment_Size'] = df['TotLen Fwd Pkts'] / (df['Tot Fwd Pkts'] + 1)
            df['Avg_Bwd_Segment_Size'] = df['TotLen Bwd Pkts'] / (df['Tot Bwd Pkts'] + 1)
        
        # Flow ratio features
        if 'Fwd Pkts/s' in df.columns and 'Bwd Pkts/s' in df.columns:
            df['Pkt_Rate_Ratio'] = df['Fwd Pkts/s'] / (df['Bwd Pkts/s'] + 0.001)
        
        # Flag features
        if all(col in df.columns for col in ['SYN Flag Cnt', 'PSH Flag Cnt', 'URG Flag Cnt']):
            df['Total_Control_Flags'] = df['SYN Flag Cnt'] + df['PSH Flag Cnt'] + df['URG Flag Cnt']
        
        # Log transforms for skewed distributions
        for col in df.columns:
            if col != 'Label' and df[col].min() >= 0 and df[col].max() > 0:
                if col.endswith(('Duration', 'Tot', 'Pkts', 'Len')):
                    df[f'Log_{col}'] = np.log1p(df[col])
        
    except Exception as e:
        print(f"Error creating derived features: {e}")
    
    # Final check for any non-numeric data
    for col in df.columns:
        if col != 'Label' and not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  - Warning: {col} is still non-numeric. Converting to float...")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"After feature engineering, data shape: {df.shape}")
    return df

# Function to handle class imbalance with multiple resampling techniques
def handle_class_imbalance(X, y, method='smote'):
    print(f"Handling class imbalance using {method}...")
    print(f"Before sampling - Class distribution: {pd.Series(y).value_counts()}")
    
    # Count samples in each class
    class_counts = pd.Series(y).value_counts()
    
    # If we only have one class, force creation of synthetic samples
    if len(class_counts) < 2:
        print("WARNING: Only one class found in the data. Creating synthetic samples...")
        
        # Get the existing class
        existing_class = class_counts.index[0]
        missing_class = 1 if existing_class == 0 else 0
        
        # Split the existing data
        X_existing = X.copy()
        y_existing = y.copy()
        
        # Create synthetic samples for the missing class
        n_synthetic = min(len(X_existing) // 4, 1000)  # Create 25% or up to 1000 synthetic samples
        
        # Take a subset of samples to modify
        indices_to_modify = np.random.choice(range(len(X_existing)), n_synthetic, replace=False)
        X_synthetic = X_existing.iloc[indices_to_modify].copy()
        
        # Modify feature values to create different class
        for col in X_synthetic.columns:
            # Skip modification for certain columns
            if col in ['Protocol', 'Dst Port']:
                continue
                
            # Add noise to numeric columns to create synthetic samples
            if pd.api.types.is_numeric_dtype(X_synthetic[col]):
                if missing_class == 0:  # Creating benign
                    X_synthetic[col] = X_synthetic[col] * np.random.uniform(0.5, 0.8, size=len(X_synthetic))
                else:  # Creating attack
                    X_synthetic[col] = X_synthetic[col] * np.random.uniform(1.2, 1.5, size=len(X_synthetic))
        
        # Create labels for synthetic data
        y_synthetic = pd.Series([missing_class] * n_synthetic)
        
        # Combine original and synthetic data
        X_combined = pd.concat([X_existing, X_synthetic])
        y_combined = pd.concat([y_existing, y_synthetic])
        
        print(f"After synthetic sampling - Class distribution: {pd.Series(y_combined).value_counts()}")
        return X_combined, y_combined
    
    # Apply resampling based on method selected
    try:
        # Use adjusted k_neighbors for SMOTE variants
        k_neighbors = min(5, class_counts.min() - 1) if class_counts.min() > 5 else 1
        
        if k_neighbors > 0:
            if method.lower() == 'smote':
                resampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            elif method.lower() == 'adasyn':
                resampler = ADASYN(random_state=42, n_neighbors=k_neighbors)
            elif method.lower() == 'smotetomek':
                resampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
            else:
                print(f"Unknown resampling method: {method}. Using SMOTE.")
                resampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
                
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            print(f"After {method} - Class distribution: {pd.Series(y_resampled).value_counts()}")
            return X_resampled, y_resampled
        else:
            print("Not enough samples in minority class for resampling. Using random oversampling.")
            # Fall back to simple random oversampling
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            minority_count = class_counts.min()
            majority_count = class_counts.max()
            
            # Oversample minority class
            minority_indices = y[y == minority_class].index
            oversampled_indices = np.random.choice(minority_indices, 
                                                   min(majority_count, minority_count * 2), 
                                                   replace=True)
            
            # Combine with majority class
            X_minority = X.loc[oversampled_indices]
            y_minority = pd.Series([minority_class] * len(oversampled_indices))
            
            X_resampled = pd.concat([X, X_minority])
            y_resampled = pd.concat([y, y_minority])
            
            print(f"After manual oversampling - Class distribution: {pd.Series(y_resampled).value_counts()}")
            return X_resampled, y_resampled
            
    except Exception as e:
        print(f"Error in resampling: {e}. Using original imbalanced data.")
        return X, y

# Function to check data leakage
def check_data_leakage(X, y, threshold=0.9):
    print("Checking for potential data leakage...")
    leaky_features = []
    
    for col in X.columns:
        try:
            correlation = X[col].corr(y)
            if abs(correlation) > threshold:
                print(f"WARNING: High correlation ({correlation:.4f}) detected between {col} and target. Possible data leak!")
                leaky_features.append((col, correlation))
        except:
            pass
    
    # Sort by absolute correlation in descending order
    leaky_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if leaky_features:
        print(f"Found {len(leaky_features)} potential data leaks.")
        for feature, corr in leaky_features:
            print(f"  - {feature}: correlation = {corr:.4f}")
    else:
        print("No potential data leaks detected.")
    
    return [feature for feature, _ in leaky_features]

# Function to train multiple models for comparison
def train_multiple_models(X_train, X_test, y_train, y_test):
    print("Training and evaluating multiple models for comparison...")
    
    # Initialize dictionary to store results
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC AUC': [],
        'PR AUC': [],
        'Training Time': []
    }
    
    # Dictionary of models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
    }
    
    # Train and evaluate each model
    best_model = None
    best_f1 = 0
    model_predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        start_time = time.time()
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"{name} training completed in {training_time:.2f} seconds")
            
            # Get predictions
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # For models without predict_proba (like some SVM configurations)
                y_proba = model.decision_function(X_test) if hasattr(model, 'decision_function') else None
            
            # Store predictions for ensemble later
            model_predictions[name] = y_pred
            
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Advanced metrics
            if y_proba is not None:
                roc_auc = auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
                pr_auc = average_precision_score(y_test, y_proba)
            else:
                roc_auc = np.nan
                pr_auc = np.nan
            
            # Store results
            results['Model'].append(name)
            results['Accuracy'].append(accuracy)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F1 Score'].append(f1)
            results['ROC AUC'].append(roc_auc)
            results['PR AUC'].append(pr_auc)
            results['Training Time'].append(training_time)
            
            # Print detailed metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            if not np.isnan(roc_auc):
                print(f"ROC AUC: {roc_auc:.4f}")
            if not np.isnan(pr_auc):
                print(f"PR AUC: {pr_auc:.4f}")
            
            # Track best model based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_model = (name, model)
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            results['Model'].append(name)
            results['Accuracy'].append(np.nan)
            results['Precision'].append(np.nan)
            results['Recall'].append(np.nan)
            results['F1 Score'].append(np.nan)
            results['ROC AUC'].append(np.nan)
            results['PR AUC'].append(np.nan)
            results['Training Time'].append(np.nan)
    
    # Create majority vote ensemble
    print("\nCreating ensemble model (majority voting)...")
    ensemble_predictions = []
    
    if len(model_predictions) > 1:
        # For each test sample, get the most common prediction across all models
        for i in range(len(y_test)):
            votes = [model_predictions[model_name][i] for model_name in model_predictions]
            majority = max(set(votes), key=votes.count)
            ensemble_predictions.append(majority)
        
        # Calculate ensemble metrics
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        ensemble_precision, ensemble_recall, ensemble_f1, _ = precision_recall_fscore_support(
            y_test, ensemble_predictions, average='weighted'
        )
        
        # Add ensemble to results
        results['Model'].append('Ensemble (Majority Vote)')
        results['Accuracy'].append(ensemble_accuracy)
        results['Precision'].append(ensemble_precision)
        results['Recall'].append(ensemble_recall)
        results['F1 Score'].append(ensemble_f1)
        results['ROC AUC'].append(np.nan)  # No probability scores for ensemble
        results['PR AUC'].append(np.nan)
        results['Training Time'].append(np.nan)
        
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print(f"Ensemble Precision: {ensemble_precision:.4f}")
        print(f"Ensemble Recall: {ensemble_recall:.4f}")
        print(f"Ensemble F1 Score: {ensemble_f1:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create a comparative bar chart for F1 scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['Model'], results_df['F1 Score'])
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.title('F1 Score Comparison Across Models')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_comparison_f1.png')
    
    # Create ROC curves for models with probability outputs
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if name in model_predictions and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
            except Exception as e:
                print(f"Could not generate ROC curve for {name}: {e}")
    
    # Add the random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curves.png')
    
    # Create precision-recall curves
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if name in model_predictions and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = average_precision_score(y_test, y_proba)
                plt.plot(recall, precision, lw=2, label=f'{name} (AP = {pr_auc:.4f})')
            except Exception as e:
                print(f"Could not generate PR curve for {name}: {e}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('precision_recall_curves.png')
    
    # Compare model calibration
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if name in model_predictions and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
                plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)
            except Exception as e:
                print(f"Could not generate calibration curve for {name}: {e}")
    
    # Add perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('calibration_curves.png')
    
    # Save comparative results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    
    print("\nComparison results saved to 'model_comparison_results.csv'")
    
    # Return the best model based on F1 score
    if best_model:
        print(f"\nBest performing model: {best_model[0]} with F1 Score: {best_f1:.4f}")
        return best_model[1], results_df
    else:
        print("No model performed satisfactorily.")
        return None, results_df

# Function to perform hyperparameter tuning
def tune_hyperparameters(X_train, y_train, model_type='RandomForest'):
    print(f"Performing hyperparameter tuning for {model_type}...")
    
    # Define parameter grids for different models
    param_grids = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42, class_weight='balanced'),
            'param_grid': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            }
        }
    }
    
    # Check if the specified model exists in our grid
    if model_type not in param_grids:
        print(f"Model {model_type} not found in parameter grids. Using RandomForest instead.")
        model_type = 'RandomForest'
    
    # Get model and parameter grid
    model = param_grids[model_type]['model']
    param_grid = param_grids[model_type]['param_grid']
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Initialize the Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    try:
        grid_search.fit(X_train, y_train)
        
        # Print the best parameters and score
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Create a results DataFrame
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results = cv_results.sort_values('rank_test_score')
        
        # Save hyperparameter tuning results
        cv_results.to_csv(f'{model_type}_tuning_results.csv', index=False)
        
        return best_model, grid_search.best_params_
        
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        print("Using default model configuration.")
        return model, {}

# Function to analyze feature importance
def analyze_feature_importance(model, X, feature_names):
    print("Analyzing feature importance...")
    
    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        # Create feature importance plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Create cumulative importance plot
        plt.figure(figsize=(10, 6))
        feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum()
        plt.plot(range(1, len(feature_importance) + 1), feature_importance['Cumulative_Importance'])
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Importance')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.legend()
        plt.grid(True)
        plt.savefig('cumulative_importance.png')
        
        # Calculate number of features needed for 95% importance
        features_for_95pct = sum(feature_importance['Cumulative_Importance'] <= 0.95) + 1
        print(f"Number of features needed for 95% importance: {features_for_95pct}")
        
        return feature_importance
    
    # For models without direct feature importance (e.g., SVM, Logistic Regression)
    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })
        feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)
        
        print("\nTop 15 Features by Coefficient Magnitude:")
        print(feature_importance.head(15))
        
        # Create coefficient plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Absolute_Coefficient', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Features by Coefficient Magnitude')
        plt.tight_layout()
        plt.savefig('feature_coefficients.png')
        
        return feature_importance
    
    else:
        print("Model doesn't provide feature importance information directly.")
        return None

# Function to perform error analysis
def error_analysis(model, X_test, y_test):
    print("Performing error analysis...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Find misclassified samples
    misclassified = X_test.copy()
    misclassified['true_label'] = y_test.values
    misclassified['predicted_label'] = y_pred
    misclassified['correct'] = misclassified['true_label'] == misclassified['predicted_label']
    
    # Separate into false positives and false negatives
    false_positives = misclassified[(misclassified['true_label'] == 0) & (misclassified['predicted_label'] == 1)]
    false_negatives = misclassified[(misclassified['true_label'] == 1) & (misclassified['predicted_label'] == 0)]
    
    print(f"Total samples: {len(X_test)}")
    print(f"Correct predictions: {sum(misclassified['correct'])}")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")
    
    # Prepare confusion matrix with rates
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate rates for the confusion matrix
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)
    
    # Create a normalized confusion matrix
    cm_normalized = np.array([[tn/total, fp/total], [fn/total, tp/total]])
    
    # Create a pretty confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Attack (1)'], 
                yticklabels=['Normal (0)', 'Attack (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Also create a percentage-based confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Normal (0)', 'Attack (1)'], 
                yticklabels=['Normal (0)', 'Attack (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix (percentages)')
    plt.savefig('confusion_matrix_normalized.png')
    
    # Calculate specific metrics
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate false positive rate and false negative rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr
    }
    
    print("\nDetailed Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Analyze characteristics of false positives and false negatives
    if len(false_positives) > 0:
        print("\nFalse Positive Analysis:")
        fp_analysis = false_positives.describe()
        print("Statistical summary of false positives (first 5 features):")
        print(fp_analysis.iloc[:, :5])
        
    if len(false_negatives) > 0:
        print("\nFalse Negative Analysis:")
        fn_analysis = false_negatives.describe()
        print("Statistical summary of false negatives (first 5 features):")
        print(fn_analysis.iloc[:, :5])
    
    # Prepare error analysis report
    error_report = {
        'confusion_matrix': cm,
        'metrics': metrics,
        'false_positives_count': len(false_positives),
        'false_negatives_count': len(false_negatives)
    }
    
    return error_report

# Function to save comprehensive research results
def save_research_results(model, feature_importance, error_report, model_comparison, params=None):
    print("Generating comprehensive research results...")
    
    # Create a detailed research report
    with open('research_results.txt', 'w') as f:
        f.write("# Network Intrusion Detection System Research Results\n\n")
        f.write("## 1. Model Performance Summary\n\n")
        
        # Add metrics from error analysis
        for metric, value in error_report['metrics'].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n## 2. Confusion Matrix Analysis\n\n")
        cm = error_report['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"True Positives (TP): {tp}\n\n")
        
        f.write("## 3. Model Comparison\n\n")
        if model_comparison is not None:
            f.write(model_comparison.to_string(index=False))
        else:
            f.write("No model comparison data available.\n")
        
        f.write("\n\n## 4. Feature Importance\n\n")
        if feature_importance is not None:
            f.write("Top 10 Most Important Features:\n")
            f.write(feature_importance.head(10).to_string(index=False))
        else:
            f.write("No feature importance data available.\n")
        
        f.write("\n\n## 5. Error Analysis\n\n")
        f.write(f"Total False Positives: {error_report['false_positives_count']}\n")
        f.write(f"Total False Negatives: {error_report['false_negatives_count']}\n")
        
        f.write("\n\n## 6. Model Configuration\n\n")
        if params:
            f.write("Best Hyperparameters:\n")
            for param, value in params.items():
                f.write(f"- {param}: {value}\n")
        else:
            f.write("Default model configuration was used.\n")
        
        f.write("\n\n## 7. Visualizations\n\n")
        f.write("The following visualization files have been generated:\n")
        f.write("- confusion_matrix.png - Shows the raw counts of predictions\n")
        f.write("- confusion_matrix_normalized.png - Shows percentages\n")
        f.write("- feature_importance.png - Top 20 features by importance\n")
        f.write("- cumulative_importance.png - Cumulative feature importance curve\n")
        f.write("- model_comparison_f1.png - Bar chart of F1 scores across models\n")
        f.write("- roc_curves.png - ROC curves for different models\n")
        f.write("- precision_recall_curves.png - Precision-Recall curves\n")
        f.write("- calibration_curves.png - Model calibration analysis\n")
    
    print("Research results saved to 'research_results.txt'")

# Main function
def main():
    try:
        print("="*80)
        print("NETWORK INTRUSION DETECTION SYSTEM - RESEARCH IMPLEMENTATION")
        print("="*80)
        
        # 1. Load and preprocess data with sampling
        # Adjust the sample size based on your system's memory capacity
        sample_size = 200000
        combined_data = load_and_preprocess_data(dataset_dir, sample_size=sample_size)
        
        # Print sample of the data to see what we're working with
        print("\nSample data (first 5 rows):")
        print(combined_data.head())
        
        # 2. Feature engineering
        processed_data = feature_engineering(combined_data)
        
        # Free memory
        del combined_data
        gc.collect()
        
        # 3. Split features and target
        if 'Label' in processed_data.columns:
            X = processed_data.drop('Label', axis=1)
            y = processed_data['Label']
        else:
            raise Exception("'Label' column not found in processed data")
        
        # Free memory
        del processed_data
        gc.collect()
        
        # 4. Check for potential data leakage
        leaky_features = check_data_leakage(X, y)
        
        # Remove leaky features if found
        if leaky_features:
            print(f"Removing {len(leaky_features)} leaky features...")
            X = X.drop(leaky_features, axis=1)
            print(f"After removing leaky features, X shape: {X.shape}")
        
        # 5. Feature scaling
        print("Applying feature scaling...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Free memory
        del X
        gc.collect()
        
        # 6. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        
        # Print class distribution
        print(f"Training class distribution: {pd.Series(y_train).value_counts()}")
        print(f"Testing class distribution: {pd.Series(y_test).value_counts()}")
        
        # Free memory
        del X_scaled, y
        gc.collect()
        
        # 7. Try different resampling techniques
        print("\nComparing different resampling techniques...")
        resampling_methods = ['smote', 'adasyn', 'smotetomek']
        resampling_results = {}
        
        for method in resampling_methods:
            print(f"\nTesting resampling method: {method}")
            X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train, method=method)
            
            # Quick evaluation with Random Forest
            model = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            resampling_results[method] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"Results with {method}:")
            print(f"- Accuracy: {accuracy:.4f}")
            print(f"- Precision: {precision:.4f}")
            print(f"- Recall: {recall:.4f}")
            print(f"- F1 Score: {f1:.4f}")
        
        # Find best resampling method based on F1 score
        best_method = max(resampling_results, key=lambda k: resampling_results[k]['f1'])
        print(f"\nBest resampling method: {best_method} with F1 score: {resampling_results[best_method]['f1']:.4f}")
        
        # Apply the best resampling method
        X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train, method=best_method)
        
        # Free memory
        del X_train, y_train
        gc.collect()
        
        # 8. Train and evaluate multiple models
        best_model, model_comparison = train_multiple_models(X_train_resampled, X_test, y_train_resampled, y_test)
        
        # 9. Hyperparameter tuning for the best model type
        if best_model is not None:
            model_type = type(best_model).__name__
            
            # Map model class name to our tuning function parameter
            model_type_map = {
                'RandomForestClassifier': 'RandomForest',
                'GradientBoostingClassifier': 'GradientBoosting',
                'XGBClassifier': 'XGBoost',
                'SVC': 'SVM'
            }
            
            tuning_type = model_type_map.get(model_type, 'RandomForest')
            
            print(f"\nPerforming hyperparameter tuning for {model_type}...")
            tuned_model, best_params = tune_hyperparameters(X_train_resampled, y_train_resampled, model_type=tuning_type)
            
            # Evaluate tuned model
            y_pred_tuned = tuned_model.predict(X_test)
            accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
            precision_tuned, recall_tuned, f1_tuned, _ = precision_recall_fscore_support(y_test, y_pred_tuned, average='weighted')
            
            print("\nTuned model performance:")
            print(f"Accuracy: {accuracy_tuned:.4f}")
            print(f"Precision: {precision_tuned:.4f}")
            print(f"Recall: {recall_tuned:.4f}")
            print(f"F1 Score: {f1_tuned:.4f}")
            
            # If tuned model is better, use it
            if f1_tuned > resampling_results[best_method]['f1']:
                print("Tuned model outperforms the original. Using tuned model.")
                best_model = tuned_model
            else:
                print("Original model outperforms the tuned model. Keeping original model.")
        else:
            best_params = None
        
        # 10. Feature importance analysis
        feature_importance = analyze_feature_importance(best_model, X_train_resampled, X_train_resampled.columns)
        
        # 11. Error analysis
        error_report = error_analysis(best_model, X_test, y_test)
        
        # 12. Save comprehensive research results
        save_research_results(best_model, feature_importance, error_report, model_comparison, best_params)
        
        # 13. Save the model
        dump(best_model, 'best_nids_model.joblib')
        print("Best model saved as 'best_nids_model.joblib'")
        
        print("\nNetwork Intrusion Detection System research implementation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
