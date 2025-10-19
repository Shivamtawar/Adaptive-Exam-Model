import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RobustDatasetPreprocessor:
    """
    ROBUST preprocessor with aggressive option extraction and cleaning
    """
    
    def __init__(self, reference_scaler=None, reference_tag_encoder=None):
        self.scaler = reference_scaler or StandardScaler()
        self.tag_encoder = reference_tag_encoder or LabelEncoder()
        self.preprocessing_stats = {}
        
    def load_dataset(self, filepath, block_name):
        """Load dataset with error handling"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            print(f"✓ Loaded {block_name}: {len(df)} questions")
            return df
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')
            print(f"✓ Loaded {block_name}: {len(df)} questions (latin-1 encoding)")
            return df
        except Exception as e:
            print(f"✗ Error loading {block_name}: {str(e)}")
            return None
    
    def aggressive_option_extraction(self, question_text, existing_options):
        """
        ULTRA AGGRESSIVE: Extract ALL possible option patterns from question text
        """
        if pd.isna(question_text):
            return question_text, existing_options
        
        question_text = str(question_text).strip()
        extracted_opts = {}
        
        # PATTERN 1: (A)/ text (B)/ text (C)/ text (D) - MOST COMMON IN YOUR DATA
        # Example: "Children enjoy listening to (A)/ ghost stories (B)/ especially on Halloween night. (C)/ No error(D)"
        pattern1 = r'\(([A-Da-d])\)/\s*([^(]*?)(?=\s*\([A-Da-d]\)/|\s*\([A-Da-d]\)|$)'
        matches1 = re.findall(pattern1, question_text, re.IGNORECASE)
        
        if len(matches1) >= 3:  # At least 3 options found
            for i, match in enumerate(matches1):
                if i >= 4:  # Only take first 4
                    break
                label = match[0].lower()
                text = match[1].strip()
                
                # Clean the text
                text = text.rstrip('/')
                text = text.strip()
                
                if text:  # Only add non-empty options
                    extracted_opts[f'option_{label}'] = text
            
            if len(extracted_opts) >= 3:
                # Remove ALL option patterns from question
                # First remove the pattern with text
                cleaned_question = re.sub(r'\([A-Da-d]\)/[^(]*(?=\([A-Da-d]\)|$)', '', question_text, flags=re.IGNORECASE)
                # Then remove any remaining isolated (X)/ or (X) patterns
                cleaned_question = re.sub(r'\s*\([A-Da-d]\)/?', '', cleaned_question, flags=re.IGNORECASE)
                # Clean up extra spaces
                cleaned_question = ' '.join(cleaned_question.split())
                cleaned_question = cleaned_question.strip()
                
                print(f"    ✂️ Extracted {len(extracted_opts)} options (Pattern 1: (A)/ format)")
                
                # If we got at least 3 options, fill in missing ones
                for letter in ['a', 'b', 'c', 'd']:
                    if f'option_{letter}' not in extracted_opts:
                        extracted_opts[f'option_{letter}'] = ''
                
                return cleaned_question, extracted_opts
        # Pattern 2: Handle incomplete questions (like "One should")
        # These are questions where the full text is in the options
        if len(question_text.split()) < 5:  # Very short question
            # This might be an incomplete question, keep it but mark it
            print(f"    ⚠️ Very short question detected: '{question_text}'")
        
        # Pattern 3: a) b) c) d) or a. b. c. d.
        pattern2 = r'(?:^|\n)\s*([a-d])[\.\)]\s*([^\n]+?)(?=\n\s*[a-d][\.\)]|$)'
        matches2 = re.findall(pattern2, question_text, re.IGNORECASE | re.MULTILINE)
        
        if len(matches2) >= 4:
            for match in matches2[:4]:
                label = match[0].lower()
                text = match[1].strip()
                extracted_opts[f'option_{label}'] = text
            
            cleaned_question = re.sub(pattern2, '', question_text, flags=re.IGNORECASE | re.MULTILINE).strip()
            print(f"    ✂️ Extracted options (Pattern 2: a) format)")
            return cleaned_question, extracted_opts
        
        # Pattern 3: (a) (b) (c) (d)
        pattern3 = r'\(([a-d])\)\s*([^(]+?)(?=\s*\([a-d]\)|$)'
        matches3 = re.findall(pattern3, question_text, re.IGNORECASE)
        
        if len(matches3) >= 4:
            for match in matches3[:4]:
                label = match[0].lower()
                text = match[1].strip()
                extracted_opts[f'option_{label}'] = text
            
            cleaned_question = re.sub(pattern3, '', question_text, flags=re.IGNORECASE).strip()
            print(f"    ✂️ Extracted options (Pattern 3: (a) format)")
            return cleaned_question, extracted_opts
        
        # Pattern 4: Options at end without clear labels (last 4 lines)
        lines = question_text.split('\n')
        if len(lines) >= 5:
            potential_options = [line.strip() for line in lines[-4:] if line.strip()]
            if len(potential_options) == 4:
                # Verify these don't look like question parts
                if not any(line.endswith('?') for line in potential_options):
                    extracted_opts = {
                        'option_a': potential_options[0],
                        'option_b': potential_options[1],
                        'option_c': potential_options[2],
                        'option_d': potential_options[3]
                    }
                    cleaned_question = '\n'.join(lines[:-4]).strip()
                    
                    if cleaned_question:
                        print(f"    ✂️ Extracted options (Pattern 4: last 4 lines)")
                        return cleaned_question, extracted_opts
        
        # No extraction - return original
        return question_text, existing_options
    
    def clean_option_text(self, option_text):
        """
        Clean individual option text from common issues
        """
        if pd.isna(option_text):
            return ''
        
        text = str(option_text).strip()
        
        # Remove leading option labels if they somehow remain
        text = re.sub(r'^[a-d][\.\)]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\([a-d]\)\s*', '', text, flags=re.IGNORECASE)
        
        # Remove trailing slashes
        text = text.rstrip('/')
        
        # Fix encoding issues - replace common garbled characters
        text = text.replace('�', '')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def standardize_columns(self, df, block_name):
        """Standardize column names across different datasets"""
        rename_dict = {
            'Question Text': 'question_text',
            'Difficulty Level': 'difficulty',
            'Correct Answer': 'answer',
            'Chapter / Subtopic': 'tag',
            'ID': 'id'
        }
        
        existing_renames = {k: v for k, v in rename_dict.items() if k in df.columns}
        df = df.rename(columns=existing_renames)
        
        # Remove header rows
        if 'question_text' in df.columns:
            df = df[df['question_text'].astype(str).str.lower() != 'question text']
            df = df[df['question_text'].astype(str).str.strip() != '']
            df = df[~df['question_text'].isna()]
        
        if 'id' not in df.columns:
            df['id'] = [f"{block_name}_{i:04d}" for i in range(len(df))]
        
        if 'tag' not in df.columns:
            df['tag'] = block_name
        
        df = df.reset_index(drop=True)
        
        return df
    
    def parse_options_block7(self, df):
        """Parse options for Block 7 (VARC) with separate columns"""
        print("\n  📋 Parsing Block 7 options (separate columns)...")
        
        option_cols = ['Option A', 'Option B', 'Option C', 'Option D']
        
        for i, col in enumerate(option_cols):
            target_col = f'option_{chr(97+i)}'
            if col in df.columns:
                df[target_col] = df[col].fillna('').astype(str).str.strip()
            else:
                df[target_col] = ''
        
        return df
    
    def parse_options_standard(self, df):
        """Parse options for standard format"""
        print("\n  📋 Parsing standard format options...")
        
        possible_cols = ['Options / Answer Choices', 'Options', 'Answer Choices']
        options_col = None
        
        for col in possible_cols:
            if col in df.columns:
                options_col = col
                print(f"    Found options in: '{col}'")
                break
        
        if options_col is None:
            if all(col in df.columns for col in ['Option A', 'Option B', 'Option C', 'Option D']):
                return self.parse_options_block7(df)
            else:
                print("    ⚠ Creating empty option columns")
                for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
                    df[opt] = ''
                return df
        
        def split_options(options_str):
            if pd.isna(options_str) or options_str == '':
                return ['', '', '', '']
            
            options_str = str(options_str).strip()
            
            # Try multiple splitting methods
            pattern1 = re.findall(r'[a-d][\)\.]\s*([^a-d\)\.]+?)(?=[a-d][\)\.]|$)', options_str, re.IGNORECASE)
            if len(pattern1) >= 4:
                return [opt.strip() for opt in pattern1[:4]]
            
            if '\n' in options_str:
                opts = [opt.strip() for opt in options_str.split('\n') if opt.strip()]
                if len(opts) >= 4:
                    return opts[:4]
            
            if ';' in options_str:
                opts = [opt.strip() for opt in options_str.split(';') if opt.strip()]
                if len(opts) >= 4:
                    return opts[:4]
            
            # Comma-separated with parenthesis handling
            opts = []
            current = ''
            paren_depth = 0
            
            for char in options_str:
                if char == '(':
                    paren_depth += 1
                    current += char
                elif char == ')':
                    paren_depth -= 1
                    current += char
                elif char == ',' and paren_depth == 0:
                    if current.strip():
                        opts.append(current.strip())
                    current = ''
                else:
                    current += char
            
            if current.strip():
                opts.append(current.strip())
            
            cleaned_opts = []
            for opt in opts:
                opt_clean = re.sub(r'^[a-d][\)\.]\s*', '', opt, flags=re.IGNORECASE).strip()
                cleaned_opts.append(opt_clean)
            
            cleaned_opts = (cleaned_opts + ['', '', '', ''])[:4]
            return cleaned_opts
        
        options_df = df[options_col].apply(split_options).apply(pd.Series)
        options_df.columns = ['option_a', 'option_b', 'option_c', 'option_d']
        
        df = pd.concat([df.drop(options_col, axis=1), options_df], axis=1)
        
        return df
    
    def detect_and_remove_paragraph_questions(self, df):
        """Remove paragraph-based questions AND questions with embedded options that couldn't be extracted"""
        print("\n  🗑️ Detecting and removing problematic questions...")
        
        initial_count = len(df)
        questions_to_remove = []
        
        for idx in df.index:
            question_text = str(df.at[idx, 'question_text'])
            
            # Criteria 1: Very long text (likely contains passage)
            if len(question_text) > 800:
                questions_to_remove.append(idx)
                continue
            
            # Criteria 2: Contains passage markers
            passage_markers = [
                'passage:', 'paragraph:', 'read the following', 
                'directions:', 'context:', 'comprehension'
            ]
            if any(marker in question_text.lower() for marker in passage_markers):
                questions_to_remove.append(idx)
                continue
            
            # Criteria 3: Multiple questions in one (has Q1, Q2, Q3 patterns)
            if re.search(r'(?:Q\s*\d+|Question\s+\d+|^\d+\.)', question_text, re.IGNORECASE):
                questions_to_remove.append(idx)
                continue
            
            # NEW Criteria 4: Still contains embedded option markers (extraction failed)
            if re.search(r'\([A-Da-d]\)/', question_text, re.IGNORECASE):
                print(f"    ⚠️ Question still has embedded options: '{question_text[:100]}...'")
                questions_to_remove.append(idx)
                continue
            
            # NEW Criteria 5: Very short/incomplete questions (less than 3 words)
            if len(question_text.split()) < 3:
                print(f"    ⚠️ Question too short: '{question_text}'")
                questions_to_remove.append(idx)
                continue
        
        # Remove identified questions
        df = df.drop(questions_to_remove)
        removed_count = initial_count - len(df)
        
        print(f"    ✓ Removed {removed_count} problematic questions")
        print(f"    ✓ Remaining: {len(df)} clean questions")
        
        return df
    
    def process_questions_and_options(self, df):
        """Process questions to extract embedded options and clean text"""
        print("\n  🔧 Processing questions with AGGRESSIVE option extraction...")
        
        options_extracted = 0
        
        for idx in df.index:
            existing_options = {
                'option_a': df.at[idx, 'option_a'] if 'option_a' in df.columns else '',
                'option_b': df.at[idx, 'option_b'] if 'option_b' in df.columns else '',
                'option_c': df.at[idx, 'option_c'] if 'option_c' in df.columns else '',
                'option_d': df.at[idx, 'option_d'] if 'option_d' in df.columns else ''
            }
            
            # Try aggressive extraction
            cleaned_question, extracted_options = self.aggressive_option_extraction(
                df.at[idx, 'question_text'],
                existing_options
            )
            
            # Update if options were extracted
            if extracted_options != existing_options:
                df.at[idx, 'question_text'] = cleaned_question
                for opt_key, opt_value in extracted_options.items():
                    if opt_value:
                        df.at[idx, opt_key] = opt_value
                options_extracted += 1
            
            # Clean all option texts
            for opt_key in ['option_a', 'option_b', 'option_c', 'option_d']:
                if opt_key in df.columns:
                    df.at[idx, opt_key] = self.clean_option_text(df.at[idx, opt_key])
        
        print(f"    ✓ Extracted options from {options_extracted} questions")
        print(f"    ✓ Cleaned all option texts")
        
        return df
    
    def encode_difficulty(self, df, block_name):
        """Encode difficulty levels"""
        difficulty_mapping = {
            'Very Easy': 0, 'very easy': 0, 'VERY EASY': 0,
            'Easy': 1, 'easy': 1, 'EASY': 1,
            'Moderate': 2, 'moderate': 2, 'MODERATE': 2, 'Medium': 2, 'medium': 2,
            'Difficult': 3, 'difficult': 3, 'DIFFICULT': 3,
            'Hard': 3, 'hard': 3, 'HARD': 3, 'Very Hard': 3, 'very hard': 3
        }
        
        df['difficulty_numeric'] = df['difficulty'].map(difficulty_mapping)
        
        unmapped = df[df['difficulty_numeric'].isnull()]['difficulty'].unique()
        if len(unmapped) > 0:
            print(f"  ⚠ Unmapped difficulty values: {unmapped}")
            df['difficulty_numeric'] = df['difficulty_numeric'].fillna(2)
        
        return df
    
    def encode_answer(self, df, block_name):
        """Encode answer choices"""
        df['answer_normalized'] = df['answer'].astype(str).str.strip().str.lower()
        df['answer_normalized'] = df['answer_normalized'].str.replace(r'[()]', '', regex=True)
        
        answer_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        df['answer_numeric'] = df['answer_normalized'].map(answer_mapping)
        
        unmapped = df[df['answer_numeric'].isnull()]['answer'].unique()
        if len(unmapped) > 0:
            print(f"  ⚠ Unmapped answer values: {unmapped}")
            df['answer_numeric'] = df['answer_numeric'].fillna(0)
        
        return df
    
    def parse_option_to_numeric(self, option):
        """Advanced option parser"""
        if pd.isna(option) or option == '':
            return 0.0
        
        option_str = str(option).strip()
        
        if len(option_str) == 0:
            return 0.0
        
        clean_option = option_str
        for pattern in ['Rs.', 'Rs', '$', '₹', '%', '?', '!', 'INR']:
            clean_option = clean_option.replace(pattern, '').strip()
        
        try:
            return float(clean_option)
        except ValueError:
            pass
        
        numbers = re.findall(r'-?\d+\.?\d*', clean_option)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        text_value = len(option_str) * 100 + sum(ord(c) for c in option_str[:5])
        return float(text_value % 10000)
    
    def engineer_features(self, df):
        """Create all engineered features"""
        
        df['question_length'] = df['question_text'].fillna('').astype(str).str.len()
        df['question_word_count'] = df['question_text'].fillna('').astype(str).str.split().str.len()
        
        for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
            df[f'{opt}_numeric'] = df[opt].apply(self.parse_option_to_numeric)
        
        option_cols = ['option_a_numeric', 'option_b_numeric', 'option_c_numeric', 'option_d_numeric']
        df['options_mean'] = df[option_cols].mean(axis=1)
        df['options_std'] = df[option_cols].std(axis=1)
        df['options_range'] = df[option_cols].max(axis=1) - df[option_cols].min(axis=1)
        
        df[['options_std', 'options_range']] = df[['options_std', 'options_range']].replace([np.inf, -np.inf], 0)
        
        df['correct_option_value'] = df.apply(
            lambda row: row[f'option_{row["answer_normalized"]}_numeric'] 
            if row['answer_normalized'] in ['a', 'b', 'c', 'd'] else 0, 
            axis=1
        )
        
        df['answer_position'] = df['answer_numeric']
        df['difficulty_rarity'] = 0.25
        
        df['difficulty_bin'] = pd.cut(
            df['difficulty_numeric'], 
            bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
            labels=['very_easy', 'easy', 'moderate', 'difficult']
        )
        
        return df
    
    def preprocess_dataset(self, df, block_name, is_block7=False):
        """Main preprocessing pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing {block_name}")
        print(f"{'='*60}")
        
        df = self.standardize_columns(df, block_name)
        
        if is_block7:
            df = self.parse_options_block7(df)
        else:
            df = self.parse_options_standard(df)
        
        df = self.detect_and_remove_paragraph_questions(df)
        df = self.process_questions_and_options(df)
        df = self.encode_difficulty(df, block_name)
        df = self.encode_answer(df, block_name)
        df = self.engineer_features(df)
        
        self.print_quality_checks(df, block_name)
        
        return df
    
    def print_quality_checks(self, df, block_name):
        """Print quality check statistics"""
        print(f"\n📊 Quality Checks for {block_name}:")
        print(f"  Total questions: {len(df)}")
        
        for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
            empty = (df[opt].fillna('').astype(str).str.strip() == '').sum()
            if empty > 0:
                print(f"  ⚠ Empty {opt}: {empty} questions")
        
        print(f"\n  Difficulty distribution:")
        for level, count in df['difficulty_numeric'].value_counts().sort_index().items():
            level_name = ['Very Easy', 'Easy', 'Moderate', 'Difficult'][int(level)]
            print(f"    {level_name}: {count} ({count/len(df)*100:.1f}%)")
    
    def merge_and_finalize(self, dfs, block_names, output_file='processed_combined_datasets.csv'):
        """Merge datasets and apply global transformations"""
        print(f"\n{'='*60}")
        print("Merging and Finalizing Datasets")
        print(f"{'='*60}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Combined {len(combined_df)} questions from {len(dfs)} datasets")
        
        print("\n🧹 Final cleaning...")
        initial_count = len(combined_df)
        
        # Remove questions with empty options
        for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
            combined_df = combined_df[combined_df[opt].fillna('').astype(str).str.strip() != '']
        
        # Remove questions with encoding issues in question text
        combined_df = combined_df[~combined_df['question_text'].astype(str).str.contains('�')]
        
        # Remove questions with malformed options (like "4?(")
        for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
            combined_df = combined_df[~combined_df[opt].astype(str).str.match(r'^\d+\?\(')]
        
        filtered_count = initial_count - len(combined_df)
        print(f"  ✓ Removed {filtered_count} questions with issues")
        
        combined_df['id'] = [f"Q{i:06d}" for i in range(len(combined_df))]
        
        combined_df['tag_encoded'] = self.tag_encoder.fit_transform(combined_df['tag'].fillna('Unknown'))
        
        difficulty_counts = combined_df['difficulty_numeric'].value_counts()
        combined_df['difficulty_rarity'] = combined_df['difficulty_numeric'].map(
            lambda x: 1.0 / difficulty_counts.get(x, 1)
        )
        
        numeric_features = ['question_length', 'question_word_count',
                           'options_mean', 'options_std', 'options_range']
        
        for col in numeric_features:
            combined_df[col] = combined_df[col].replace([np.inf, -np.inf], 0)
        
        combined_df[numeric_features] = self.scaler.fit_transform(combined_df[numeric_features])
        
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)
        
        if 'difficulty_bin' in combined_df.columns:
            if not isinstance(combined_df['difficulty_bin'].dtype, pd.CategoricalDtype):
                combined_df['difficulty_bin'] = pd.Categorical(combined_df['difficulty_bin'])
            combined_df['difficulty_bin'] = combined_df['difficulty_bin'].cat.add_categories(['unknown'])
            combined_df['difficulty_bin'] = combined_df['difficulty_bin'].fillna('unknown')
        
        string_cols = combined_df.select_dtypes(include=['object']).columns
        combined_df[string_cols] = combined_df[string_cols].fillna('')
        
        final_columns = [
            'id', 'question_text', 'option_a', 'option_b', 'option_c', 'option_d',
            'answer', 'difficulty', 'difficulty_numeric', 'answer_numeric', 'tag_encoded',
            'option_a_numeric', 'option_b_numeric', 'option_c_numeric', 'option_d_numeric',
            'correct_option_value', 'question_length', 'question_word_count',
            'options_mean', 'options_std', 'options_range', 'answer_position',
            'difficulty_rarity', 'difficulty_bin'
        ]
        
        combined_df = combined_df[final_columns]
        
        combined_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved: {output_file}")
        
        sample_file = output_file.replace('.csv', '_sample.csv')
        combined_df.head(20).to_csv(sample_file, index=False)
        print(f"💾 Saved sample: {sample_file}")
        
        metadata = {
            'total_questions': len(combined_df),
            'datasets': block_names,
            'difficulty_distribution': {int(k): int(v) for k, v in combined_df['difficulty_numeric'].value_counts().to_dict().items()},
            'answer_distribution': {int(k): int(v) for k, v in combined_df['answer_numeric'].value_counts().to_dict().items()},
            'tag_categories': self.tag_encoder.classes_.tolist(),
            'feature_columns': final_columns,
            'preprocessing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fixes_applied': [
                'Aggressive option extraction from question text',
                'Removed encoding issues (� characters)',
                'Removed malformed options',
                'Cleaned all option texts',
                'Removed paragraph-based questions'
            ]
        }
        
        metadata_file = output_file.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved: {metadata_file}")
        
        print(f"\n{'='*60}")
        print("✨ Final Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total Questions: {len(combined_df)}")
        print(f"\n✅ ROBUST FIXES APPLIED:")
        print(f"  ✓ Aggressive option extraction (Pattern 1: (A)/ format)")
        print(f"  ✓ Removed encoding issues")
        print(f"  ✓ Removed malformed options")
        print(f"  ✓ Cleaned all option texts")
        
        return combined_df


def main():
    """Main execution function"""
    print("="*60)
    print("ROBUST Dataset Preprocessing Pipeline v4.0")
    print("Aggressive fixes for embedded options and encoding issues")
    print("="*60)
    
    preprocessor = RobustDatasetPreprocessor()
    
    datasets = [
        ('Block 1 Arithmetic.csv', 'Block_1_Arithmetic', False),
        ('BLOCK_2_NumberSystem_Arranged.csv', 'Block_2_NumberSystem', False),
        ('BLOCK - 7 VARC.csv', 'Block_7_VARC', True)
    ]
    
    processed_dfs = []
    block_names = []
    
    for filepath, block_name, is_block7 in datasets:
        if not Path(filepath).exists():
            print(f"\n⚠ Warning: {filepath} not found, skipping...")
            continue
        
        df = preprocessor.load_dataset(filepath, block_name)
        if df is not None:
            processed_df = preprocessor.preprocess_dataset(df, block_name, is_block7)
            processed_dfs.append(processed_df)
            block_names.append(block_name)
    
    if not processed_dfs:
        print("\n✗ Error: No datasets were successfully processed!")
        return
    
    combined_df = preprocessor.merge_and_finalize(
        processed_dfs, 
        block_names,
        'processed_combined_datasets.csv'
    )
    
    print("\n" + "="*60)
    print("✅ ROBUST PREPROCESSING COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("  1. Check 'processed_combined_datasets_sample.csv'")
    print("  2. Verify NO options are embedded in questions")
    print("  3. Verify NO encoding issues (� characters)")
    print("  4. Verify NO malformed options like '4?('")
    print("  5. Retrain your model with the cleaned data")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()