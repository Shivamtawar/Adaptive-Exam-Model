import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

def preprocess_thinkplus_data(csv_file):
    """
    Enhanced preprocessing for adaptive quiz system with better feature engineering
    """
    # Load the CSV file
    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} questions")
    
    # Display basic info
    print("\n=== Dataset Overview ===")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # ========== DIFFICULTY ENCODING ==========
    difficulty_mapping = {
        'Very easy': 0,
        'very easy': 0,
        'Easy': 1,
        'easy': 1,
        'Moderate': 2,
        'moderate': 2,
        'Difficult': 3,
        'difficult': 3,
        'Hard': 3,
        'hard': 3
    }
    df['difficulty_numeric'] = df['difficulty'].map(difficulty_mapping)
    
    # Check for unmapped difficulties
    if df['difficulty_numeric'].isnull().any():
        print("\n⚠️ Warning: Some difficulty values couldn't be mapped:")
        print(df[df['difficulty_numeric'].isnull()]['difficulty'].unique())
    
    # ========== ANSWER ENCODING ==========
    answer_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
                     'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df['answer_numeric'] = df['answer'].str.lower().map(answer_mapping)
    
    # Check for unmapped answers
    if df['answer_numeric'].isnull().any():
        print("\n⚠️ Warning: Some answer values couldn't be mapped:")
        print(df[df['answer_numeric'].isnull()]['answer'].unique())
    
    # ========== TAG ENCODING ==========
    # Use LabelEncoder for tags if there are multiple categories
    if 'tag' in df.columns:
        le_tag = LabelEncoder()
        df['tag_encoded'] = le_tag.fit_transform(df['tag'].fillna('Unknown'))
        print(f"\nTag categories found: {le_tag.classes_}")
    else:
        df['tag_encoded'] = 1  # Default for single tag type
    
    # ========== OPTION PARSING ==========
    def parse_option_advanced(option):
        """
        Enhanced option parser that handles multiple formats:
        - Percentages: 25%, 50.5%
        - Currency: Rs.100, Rs. 1000
        - Plain numbers: 42, 3.14
        - Text: Returns hash of text for non-numeric options
        """
        if pd.isna(option):
            return 0
        
        option_str = str(option).strip()
        
        # Remove common prefixes/suffixes
        clean_option = option_str.replace('Rs.', '').replace('Rs', '').replace('%', '').strip()
        
        # Try to convert to float
        try:
            return float(clean_option)
        except ValueError:
            # For text options, return a hash value (for text-based questions)
            # This allows the model to still process them
            return hash(option_str) % 10000  # Keep hash within reasonable range
    
    # Parse all options
    print("\n=== Parsing Options ===")
    for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
        df[f'{opt}_numeric'] = df[opt].apply(parse_option_advanced)
        print(f"{opt}: {df[f'{opt}_numeric'].describe()}")
    
    # ========== CORRECT OPTION VALUE ==========
    df['correct_option_value'] = df.apply(
        lambda row: row[f'option_{row["answer"].lower()}_numeric'] 
        if pd.notna(row['answer']) else 0, 
        axis=1
    )
    
    # ========== FEATURE ENGINEERING ==========
    
    # 1. Question length (complexity indicator)
    df['question_length'] = df['question_text'].str.len()
    df['question_word_count'] = df['question_text'].str.split().str.len()
    
    # 2. Option statistics
    df['options_mean'] = df[['option_a_numeric', 'option_b_numeric', 
                              'option_c_numeric', 'option_d_numeric']].mean(axis=1)
    df['options_std'] = df[['option_a_numeric', 'option_b_numeric', 
                             'option_c_numeric', 'option_d_numeric']].std(axis=1)
    df['options_range'] = df[['option_a_numeric', 'option_b_numeric', 
                               'option_c_numeric', 'option_d_numeric']].max(axis=1) - \
                          df[['option_a_numeric', 'option_b_numeric', 
                               'option_c_numeric', 'option_d_numeric']].min(axis=1)
    
    # 3. Correct answer position (some patterns might exist)
    df['answer_position'] = df['answer_numeric']
    
    # 4. Difficulty distribution score (for balancing)
    difficulty_counts = df['difficulty_numeric'].value_counts()
    df['difficulty_rarity'] = df['difficulty_numeric'].map(
        lambda x: 1 / difficulty_counts.get(x, 1)
    )
    
    # ========== NORMALIZATION (OPTIONAL) ==========
    # Normalize numeric features for better model performance
    numeric_features = ['question_length', 'question_word_count', 'options_mean', 
                       'options_std', 'options_range']
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # ========== QUESTION CATEGORIZATION FOR ADAPTIVE TESTING ==========
    # Create bins for adaptive testing
    df['difficulty_bin'] = pd.cut(df['difficulty_numeric'], 
                                   bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
                                   labels=['very_easy', 'easy', 'moderate', 'difficult'])
    
    # ========== QUALITY CHECKS ==========
    print("\n=== Quality Checks ===")
    print(f"Questions per difficulty level:")
    print(df['difficulty_numeric'].value_counts().sort_index())
    print(f"\nAnswer distribution:")
    print(df['answer_numeric'].value_counts().sort_index())
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['question_text'], keep=False).sum()
    print(f"\nDuplicate questions: {duplicates}")
    
    # ========== SELECT FINAL COLUMNS ==========
    processed_df = df[[
        'id', 'question_text', 'option_a', 'option_b', 'option_c', 'option_d',
        'answer', 'difficulty',  # Keep original for display
        'difficulty_numeric', 'answer_numeric', 'tag_encoded',
        'option_a_numeric', 'option_b_numeric', 'option_c_numeric', 'option_d_numeric',
        'correct_option_value', 'question_length', 'question_word_count',
        'options_mean', 'options_std', 'options_range', 'answer_position',
        'difficulty_rarity', 'difficulty_bin'
    ]].copy()
    
    # Fill any remaining NaN values (excluding categorical columns)
    # Get numeric columns only
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)
    
    # Fill categorical columns with appropriate default
    if 'difficulty_bin' in processed_df.columns:
        processed_df['difficulty_bin'] = processed_df['difficulty_bin'].cat.add_categories(['unknown'])
        processed_df['difficulty_bin'] = processed_df['difficulty_bin'].fillna('unknown')
    
    # Fill string columns
    string_cols = processed_df.select_dtypes(include=['object']).columns
    processed_df[string_cols] = processed_df[string_cols].fillna('')
    
    print(f"\n✅ Preprocessing complete! Final shape: {processed_df.shape}")
    
    return processed_df, scaler


def create_train_test_split(df, test_size=0.2, stratify=True):
    """
    Create stratified train-test split maintaining difficulty distribution
    """
    from sklearn.model_selection import train_test_split
    
    if stratify:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['difficulty_numeric'],
            random_state=42
        )
    else:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            random_state=42
        )
    
    print(f"\n=== Train/Test Split ===")
    print(f"Training set: {len(train_df)} questions")
    print(f"Test set: {len(test_df)} questions")
    print(f"\nTraining difficulty distribution:")
    print(train_df['difficulty_numeric'].value_counts().sort_index())
    print(f"\nTest difficulty distribution:")
    print(test_df['difficulty_numeric'].value_counts().sort_index())
    
    return train_df, test_df


def save_processed_data(df, output_file='processed_thinkplus.csv'):
    """
    Save processed data with metadata
    """
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved processed data to: {output_file}")
    
    # Save metadata
    metadata = {
        'total_questions': len(df),
        'difficulty_distribution': df['difficulty_numeric'].value_counts().to_dict(),
        'answer_distribution': df['answer_numeric'].value_counts().to_dict(),
        'feature_columns': df.columns.tolist()
    }
    
    import json
    with open(output_file.replace('.csv', '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved metadata to: {output_file.replace('.csv', '_metadata.json')}")


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Process the data
    processed_df, scaler = preprocess_thinkplus_data('thinkplus_dataset.csv')
    
    # Create train-test split
    train_df, test_df = create_train_test_split(processed_df, test_size=0.2)
    
    # Save processed data
    save_processed_data(processed_df, 'processed_thinkplus_full.csv')
    save_processed_data(train_df, 'processed_thinkplus_train.csv')
    save_processed_data(test_df, 'processed_thinkplus_test.csv')
    
    print("\n" + "="*50)
    print("✨ PREPROCESSING COMPLETE ✨")
    print("="*50)
    print("\nNext steps:")
    print("1. Review the processed CSV files")
    print("2. Build the adaptive testing model")
    print("3. Implement the section-based testing system")