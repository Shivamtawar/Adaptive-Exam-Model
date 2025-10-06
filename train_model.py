"""
Adaptive Quiz System - ML Model Training (UPDATED with NaN filtering)
Save this as: adaptive_quiz_model.py

Updates:
- Filters out questions with NaN options during preprocessing
- Analytics tracking ready
- Same PKL file name maintained
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import expit

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# ============================================================================
# CELL 1: Load and Clean Data (Remove NaN options)
# ============================================================================
df = pd.read_csv('processed_thinkplus_full.csv')

print(f"Original Dataset Shape: {df.shape}")

# Filter out questions with NaN in any option column
option_cols = ['option_a', 'option_b', 'option_c', 'option_d']
df_clean = df.dropna(subset=option_cols)

print(f"After removing NaN options: {df_clean.shape}")
print(f"Removed {len(df) - len(df_clean)} questions with NaN options")

# Also remove questions with empty string options
for col in option_cols:
    df_clean = df_clean[df_clean[col].str.strip() != '']

print(f"After removing empty options: {df_clean.shape}")
print(f"\nDifficulty Distribution:")
print(df_clean['difficulty_numeric'].value_counts().sort_index())

# ============================================================================
# CELL 2: Prepare Features for ML Model
# ============================================================================
feature_cols = [
    'tag_encoded', 'option_a_numeric', 'option_b_numeric', 
    'option_c_numeric', 'option_d_numeric', 'correct_option_value',
    'question_length', 'question_word_count', 'options_mean', 
    'options_std', 'options_range', 'answer_position'
]

X = df_clean[feature_cols].copy()
y = df_clean['difficulty_numeric'].copy()
X = X.fillna(0)

print(f"Feature Matrix Shape: {X.shape}")
print(f"Target Shape: {y.shape}")

# ============================================================================
# CELL 3: Train-Test Split and Model Training
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_test = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# ============================================================================
# CELL 4: Adaptive Quiz System Classes
# ============================================================================
class AdaptiveQuizSystem:
    """Adaptive Quiz System with analytics tracking"""
    
    def __init__(self, questions_df, difficulty_model):
        self.questions_df = questions_df.copy()
        self.difficulty_model = difficulty_model
        self.user_ability = 0.5
        self.response_history = []
        self.asked_questions = set()
        self.start_time = datetime.now()
        self.question_times = []
        
    def estimate_ability(self, responses):
        if not responses:
            return 0.5
        
        ability = 0.5
        learning_rate = 0.2
        
        for i, (difficulty, correct) in enumerate(responses):
            if correct:
                target = ability + 0.4
            else:
                target = ability - 0.4
            
            ability = ability + learning_rate * (target - ability)
            learning_rate = min(0.3, learning_rate + 0.01)
        
        return np.clip(ability, 0.0, 3.0)
    
    def select_next_question(self, mode='adaptive', target_difficulty=None):
        available_questions = self.questions_df[
            ~self.questions_df['id'].isin(self.asked_questions)
        ].copy()
        
        if len(available_questions) == 0:
            return None
        
        if mode == 'adaptive':
            target_level = int(round(self.user_ability))
            target_level = np.clip(target_level, 0, 3)
            
            available_questions['difficulty_diff'] = abs(
                available_questions['difficulty_numeric'] - target_level
            )
            
            available_questions['score'] = available_questions['difficulty_diff'].apply(
                lambda x: 0 if x == 0 else (1.0 if x == 1 else 5.0)
            )
            
            available_questions['score'] += np.random.uniform(0, 0.3, len(available_questions))
            next_question = available_questions.nsmallest(1, 'score').iloc[0]
        else:
            next_question = available_questions.sample(1).iloc[0]
        
        return next_question
    
    def submit_answer(self, question_id, user_answer, correct_answer, time_spent):
        is_correct = (user_answer == correct_answer)
        question = self.questions_df[self.questions_df['id'] == question_id].iloc[0]
        difficulty = question['difficulty_numeric']
        
        self.response_history.append({
            'question_id': question_id,
            'difficulty': difficulty,
            'correct': is_correct,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'time_spent': time_spent,
            'timestamp': datetime.now().isoformat()
        })
        
        self.question_times.append(time_spent)
        self.asked_questions.add(question_id)
        
        responses = [(r['difficulty'], r['correct']) for r in self.response_history]
        self.user_ability = self.estimate_ability(responses)
        
        return is_correct, self.user_ability
    
    def get_analytics(self):
        if not self.response_history:
            return {}
        
        total = len(self.response_history)
        correct = sum(1 for r in self.response_history if r['correct'])
        total_time = sum(self.question_times)
        avg_time = total_time / total if total > 0 else 0
        
        difficulty_breakdown = {0: [], 1: [], 2: [], 3: []}
        for r in self.response_history:
            difficulty_breakdown[r['difficulty']].append(r['correct'])
        
        difficulty_stats = {}
        for diff, results in difficulty_breakdown.items():
            if results:
                difficulty_stats[diff] = {
                    'attempted': len(results),
                    'correct': sum(results),
                    'accuracy': sum(results) / len(results)
                }
        
        return {
            'total_questions': total,
            'correct_answers': correct,
            'wrong_answers': total - correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'total_time_seconds': int(total_time),
            'average_time_per_question': round(avg_time, 2),
            'fastest_answer': min(self.question_times),
            'slowest_answer': max(self.question_times),
            'current_ability': self.user_ability,
            'difficulty_breakdown': difficulty_stats,
            'response_history': self.response_history,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }


class SectionBasedQuizSystem:
    """Section-based quiz with analytics tracking"""
    
    def __init__(self, questions_df):
        self.questions_df = questions_df.copy()
        self.sections = {
            0: {'name': 'Very Easy', 'difficulty': 0, 'threshold': 6},
            1: {'name': 'Easy', 'difficulty': 1, 'threshold': 6},
            2: {'name': 'Moderate', 'difficulty': 2, 'threshold': 6},
            3: {'name': 'Difficult', 'difficulty': 3, 'threshold': 6}
        }
        self.current_section = 0
        self.section_questions = []
        self.section_answers = []
        self.asked_question_ids = set()
        self.completed_sections = []
        self.section_exhausted = False
        self.start_time = datetime.now()
        self.question_times = []
        
    def start_section(self, section_num, reset_section=False):
        if section_num >= len(self.sections):
            return None
        
        self.current_section = section_num
        
        if reset_section:
            self.section_questions = []
            self.section_answers = []
            self.section_exhausted = False
        
        section_difficulty = self.sections[section_num]['difficulty']
        available = self.questions_df[
            (self.questions_df['difficulty_numeric'] == section_difficulty) &
            (~self.questions_df['id'].isin(self.asked_question_ids))
        ]
        
        if len(available) == 0:
            self.section_exhausted = True
            return None
        
        if len(available) < 10:
            questions = available
        else:
            questions = available.sample(10, random_state=None)
        
        self.section_questions = questions.to_dict('records')
        for q in self.section_questions:
            self.asked_question_ids.add(q['id'])
        
        return self.section_questions
    
    def submit_section_answer(self, question_index, user_answer, time_spent):
        if question_index >= len(self.section_questions):
            return None
        
        question = self.section_questions[question_index]
        correct_answer = question['answer_numeric']
        is_correct = (user_answer == correct_answer)
        
        self.section_answers.append({
            'question_index': question_index,
            'question_id': question['id'],
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'time_spent': time_spent,
            'timestamp': datetime.now().isoformat()
        })
        
        self.question_times.append(time_spent)
        
        return is_correct
    
    def check_section_completion(self):
        total_answered = len(self.section_answers)
        
        if total_answered < 10:
            return False, False, False
        
        correct_count = sum(1 for a in self.section_answers if a['is_correct'])
        threshold = self.sections[self.current_section]['threshold']
        last_wrong = not self.section_answers[-1]['is_correct']
        
        if correct_count >= threshold:
            return True, True, False
        elif correct_count == threshold - 1 and last_wrong and total_answered == 10:
            return False, False, True
        else:
            return True, False, False
    
    def reload_section_questions(self):
        section_difficulty = self.sections[self.current_section]['difficulty']
        available = self.questions_df[
            (self.questions_df['difficulty_numeric'] == section_difficulty) &
            (~self.questions_df['id'].isin(self.asked_question_ids))
        ]
        
        if len(available) == 0:
            self.section_exhausted = True
            return None
        
        self.section_answers = []
        
        if len(available) < 10:
            questions = available
        else:
            questions = available.sample(10, random_state=None)
        
        self.section_questions = questions.to_dict('records')
        for q in self.section_questions:
            self.asked_question_ids.add(q['id'])
        
        return self.section_questions
    
    def get_11th_question(self):
        section_difficulty = self.sections[self.current_section]['difficulty']
        available = self.questions_df[
            (self.questions_df['difficulty_numeric'] == section_difficulty) &
            (~self.questions_df['id'].isin(self.asked_question_ids))
        ]
        
        if len(available) == 0:
            return None
        
        question = available.sample(1).iloc[0].to_dict()
        self.asked_question_ids.add(question['id'])
        self.section_questions.append(question)
        
        return question
    
    def proceed_to_next_section(self):
        self.completed_sections.append({
            'section': self.current_section,
            'correct': sum(1 for a in self.section_answers if a['is_correct']),
            'total': len(self.section_answers),
            'passed': sum(1 for a in self.section_answers if a['is_correct']) >= 6
        })
        
        self.current_section += 1
        
        if self.current_section >= len(self.sections):
            return None
        
        return self.start_section(self.current_section)
    
    def get_analytics(self):
        total_questions = sum(len(s.get('answers', [])) for s in self.completed_sections)
        total_correct = sum(s['correct'] for s in self.completed_sections)
        total_time = sum(self.question_times)
        
        return {
            'total_questions': total_questions,
            'correct_answers': total_correct,
            'wrong_answers': total_questions - total_correct,
            'total_time_seconds': int(total_time),
            'average_time_per_question': round(total_time / total_questions, 2) if total_questions > 0 else 0,
            'fastest_answer': min(self.question_times) if self.question_times else 0,
            'slowest_answer': max(self.question_times) if self.question_times else 0,
            'sections_completed': len(self.completed_sections),
            'sections_passed': sum(1 for s in self.completed_sections if s['passed']),
            'section_breakdown': self.completed_sections,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }


class QuizModelPackage:
    """Complete package for Flask API"""
    
    def __init__(self, difficulty_model, questions_df, feature_cols, scaler=None):
        self.difficulty_model = difficulty_model
        self.questions_df = questions_df
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.metadata = {
            'model_type': 'RandomForest',
            'n_questions': len(questions_df),
            'difficulty_levels': questions_df['difficulty_numeric'].nunique(),
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_accuracy': test_accuracy,
            'feature_importance': feature_importance.to_dict('records'),
            'version': '2.2_analytics_nan_filter'
        }
    
    def create_adaptive_quiz(self):
        return AdaptiveQuizSystem(self.questions_df, self.difficulty_model)
    
    def create_section_quiz(self):
        return SectionBasedQuizSystem(self.questions_df)
    
    def get_question_by_id(self, question_id):
        question = self.questions_df[self.questions_df['id'] == question_id]
        if len(question) == 0:
            return None
        return question.iloc[0].to_dict()
    
    def get_random_questions(self, n=10, difficulty=None):
        if difficulty is not None:
            filtered = self.questions_df[
                self.questions_df['difficulty_numeric'] == difficulty
            ]
        else:
            filtered = self.questions_df
        
        if len(filtered) < n:
            return filtered.to_dict('records')
        
        return filtered.sample(n).to_dict('records')


# ============================================================================
# CELL 5: Create and Save Model Package
# ============================================================================
model_package = QuizModelPackage(
    difficulty_model=rf_model,
    questions_df=df_clean,
    feature_cols=feature_cols,
    scaler=None
)

# Save with SAME filename as before
with open('adaptive_quiz_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✓ Model saved to 'adaptive_quiz_model.pkl'")

with open('model_metadata.json', 'w') as f:
    json.dump(model_package.metadata, f, indent=2)

print("✓ Metadata saved to 'model_metadata.json'")

import os
file_size = os.path.getsize('adaptive_quiz_model.pkl') / (1024 * 1024)
print(f"✓ Model file size: {file_size:.2f} MB")
print(f"✓ Clean questions: {len(df_clean)}")
print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)