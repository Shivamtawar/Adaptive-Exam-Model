"""
Adaptive Quiz System - ML Model Training
Save this as: adaptive_quiz_model.ipynb or .py

This notebook trains an adaptive quiz system that:
1. Predicts question difficulty
2. Estimates user ability
3. Selects optimal next questions
4. Implements section-based testing
"""

# ============================================================================
# CELL 1: Import Libraries
# ============================================================================
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# For Item Response Theory (IRT)
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function

import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# ============================================================================
# CELL 2: Load Preprocessed Data
# ============================================================================
# Load the preprocessed data
df = pd.read_csv('processed_thinkplus_full.csv')

print(f"Dataset Shape: {df.shape}")
print(f"\nDifficulty Distribution:")
print(df['difficulty_numeric'].value_counts().sort_index())
print(f"\nColumns: {df.columns.tolist()}")

# ============================================================================
# CELL 3: Prepare Features for ML Model
# ============================================================================
# Features for difficulty prediction
feature_cols = [
    'tag_encoded', 'option_a_numeric', 'option_b_numeric', 
    'option_c_numeric', 'option_d_numeric', 'correct_option_value',
    'question_length', 'question_word_count', 'options_mean', 
    'options_std', 'options_range', 'answer_position'
]

# Prepare X (features) and y (target)
X = df[feature_cols].copy()
y = df['difficulty_numeric'].copy()

# Handle any remaining NaN
X = X.fillna(0)

print(f"Feature Matrix Shape: {X.shape}")
print(f"Target Shape: {y.shape}")
print(f"\nFeatures used: {feature_cols}")

# ============================================================================
# CELL 4: Train-Test Split
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# CELL 5: Train Random Forest Model (Primary Model)
# ============================================================================
print("\n" + "="*60)
print("Training Random Forest Model...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluation
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n✅ Training Accuracy: {train_accuracy:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, 
                          target_names=['Very Easy', 'Easy', 'Moderate', 'Difficult']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# ============================================================================
# CELL 6: Adaptive Testing System - IRT Implementation
# ============================================================================
class AdaptiveQuizSystem:
    """
    Adaptive Quiz System using Item Response Theory (IRT)
    """
    
    def __init__(self, questions_df, difficulty_model):
        self.questions_df = questions_df.copy()
        self.difficulty_model = difficulty_model
        self.user_ability = 0.0  # Initial ability estimate (neutral)
        self.response_history = []
        self.asked_questions = set()
        
    def estimate_ability(self, responses):
        """
        Estimate user ability using Maximum Likelihood Estimation
        responses: list of (difficulty, correct) tuples
        """
        if not responses:
            return 0.0
        
        def negative_log_likelihood(ability):
            nll = 0
            for difficulty, correct in responses:
                # IRT 1-parameter logistic model (Rasch model)
                prob = expit(ability - difficulty)
                if correct:
                    nll -= np.log(prob + 1e-10)
                else:
                    nll -= np.log(1 - prob + 1e-10)
            return nll
        
        result = minimize(negative_log_likelihood, x0=0.0, method='BFGS')
        return result.x[0]
    
    def get_question_probability(self, question_difficulty, user_ability):
        """
        Calculate probability of correct answer using IRT
        """
        return expit(user_ability - question_difficulty)
    
    def select_next_question(self, mode='adaptive', target_difficulty=None):
        """
        Select next question based on current ability
        mode: 'adaptive' or 'fixed'
        """
        available_questions = self.questions_df[
            ~self.questions_df['id'].isin(self.asked_questions)
        ].copy()
        
        if len(available_questions) == 0:
            return None
        
        if mode == 'adaptive':
            # Select question closest to user's current ability
            available_questions['score'] = abs(
                available_questions['difficulty_numeric'] - self.user_ability
            )
            # Add some randomness to avoid predictability
            available_questions['score'] += np.random.normal(0, 0.1, len(available_questions))
            next_question = available_questions.nsmallest(1, 'score').iloc[0]
            
        elif mode == 'fixed' and target_difficulty is not None:
            # Select from specific difficulty level
            difficulty_questions = available_questions[
                available_questions['difficulty_numeric'] == target_difficulty
            ]
            if len(difficulty_questions) == 0:
                next_question = available_questions.sample(1).iloc[0]
            else:
                next_question = difficulty_questions.sample(1).iloc[0]
        else:
            # Random selection
            next_question = available_questions.sample(1).iloc[0]
        
        return next_question
    
    def submit_answer(self, question_id, user_answer, correct_answer):
        """
        Process user's answer and update ability
        """
        is_correct = (user_answer == correct_answer)
        
        question = self.questions_df[self.questions_df['id'] == question_id].iloc[0]
        difficulty = question['difficulty_numeric']
        
        self.response_history.append({
            'question_id': question_id,
            'difficulty': difficulty,
            'correct': is_correct,
            'user_answer': user_answer,
            'correct_answer': correct_answer
        })
        
        self.asked_questions.add(question_id)
        
        # Update user ability
        responses = [(r['difficulty'], r['correct']) for r in self.response_history]
        self.user_ability = self.estimate_ability(responses)
        
        return is_correct, self.user_ability
    
    def get_stats(self):
        """Get current quiz statistics"""
        if not self.response_history:
            return {
                'total_questions': 0,
                'correct_answers': 0,
                'accuracy': 0.0,
                'current_ability': self.user_ability
            }
        
        total = len(self.response_history)
        correct = sum(1 for r in self.response_history if r['correct'])
        
        return {
            'total_questions': total,
            'correct_answers': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'current_ability': self.user_ability,
            'response_history': self.response_history
        }

print("\n✅ Adaptive Quiz System class created!")

# ============================================================================
# CELL 7: Section-Based Testing System
# ============================================================================
class SectionBasedQuizSystem:
    """
    Section-based quiz with progression rules:
    - 3 sections (Easy, Moderate, Difficult)
    - 10 questions per section
    - Need 6/10 correct to proceed
    - If last question wrong, get 11th unique question
    """
    
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
        
    def start_section(self, section_num):
        """Start a new section"""
        if section_num >= len(self.sections):
            return None
        
        self.current_section = section_num
        self.section_questions = []
        self.section_answers = []
        
        # Get 10 questions from this difficulty level
        section_difficulty = self.sections[section_num]['difficulty']
        available = self.questions_df[
            (self.questions_df['difficulty_numeric'] == section_difficulty) &
            (~self.questions_df['id'].isin(self.asked_question_ids))
        ]
        
        if len(available) < 10:
            print(f"⚠️ Warning: Only {len(available)} questions available in section {section_num}")
            questions = available
        else:
            questions = available.sample(10, random_state=None)
        
        self.section_questions = questions.to_dict('records')
        for q in self.section_questions:
            self.asked_question_ids.add(q['id'])
        
        return self.section_questions
    
    def submit_section_answer(self, question_index, user_answer):
        """Submit answer for current section question"""
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
            'is_correct': is_correct
        })
        
        return is_correct
    
    def check_section_completion(self):
        """
        Check if section is complete and if user can proceed
        Returns: (is_complete, passed, needs_11th_question)
        """
        if len(self.section_answers) < 10:
            return False, False, False
        
        correct_count = sum(1 for a in self.section_answers if a['is_correct'])
        threshold = self.sections[self.current_section]['threshold']
        
        # Check if last question was wrong
        last_wrong = not self.section_answers[-1]['is_correct']
        
        if correct_count >= threshold:
            # Passed section
            return True, True, False
        elif correct_count == threshold - 1 and last_wrong:
            # Need 11th question (one short and last was wrong)
            return False, False, True
        else:
            # Failed section
            return True, False, False
    
    def get_11th_question(self):
        """Get unique 11th question from same difficulty"""
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
        """Move to next section"""
        self.completed_sections.append({
            'section': self.current_section,
            'correct': sum(1 for a in self.section_answers if a['is_correct']),
            'total': len(self.section_answers),
            'passed': sum(1 for a in self.section_answers if a['is_correct']) >= 6
        })
        
        self.current_section += 1
        
        if self.current_section >= len(self.sections):
            return None  # Quiz complete
        
        return self.start_section(self.current_section)
    
    def get_progress(self):
        """Get overall progress"""
        return {
            'current_section': self.current_section,
            'section_name': self.sections[self.current_section]['name'],
            'questions_answered': len(self.section_answers),
            'correct_in_section': sum(1 for a in self.section_answers if a['is_correct']),
            'completed_sections': self.completed_sections
        }

print("\n✅ Section-Based Quiz System class created!")

# ============================================================================
# CELL 8: Create Model Package for Flask
# ============================================================================
class QuizModelPackage:
    """
    Complete package for Flask API
    """
    
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
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def create_adaptive_quiz(self):
        """Factory method to create adaptive quiz instance"""
        return AdaptiveQuizSystem(self.questions_df, self.difficulty_model)
    
    def create_section_quiz(self):
        """Factory method to create section-based quiz instance"""
        return SectionBasedQuizSystem(self.questions_df)
    
    def get_question_by_id(self, question_id):
        """Retrieve question by ID"""
        question = self.questions_df[self.questions_df['id'] == question_id]
        if len(question) == 0:
            return None
        return question.iloc[0].to_dict()
    
    def get_random_questions(self, n=10, difficulty=None):
        """Get random questions, optionally filtered by difficulty"""
        if difficulty is not None:
            filtered = self.questions_df[
                self.questions_df['difficulty_numeric'] == difficulty
            ]
        else:
            filtered = self.questions_df
        
        if len(filtered) < n:
            return filtered.to_dict('records')
        
        return filtered.sample(n).to_dict('records')

# Create the package
model_package = QuizModelPackage(
    difficulty_model=rf_model,
    questions_df=df,
    feature_cols=feature_cols,
    scaler=None
)

print("\n✅ Model package created!")
print(f"Metadata: {json.dumps(model_package.metadata, indent=2)}")

# ============================================================================
# CELL 9: Save Model to PKL File
# ============================================================================
# Save the complete model package
with open('adaptive_quiz_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✅ Model saved to 'adaptive_quiz_model.pkl'")

# Also save metadata separately for reference
with open('model_metadata.json', 'w') as f:
    json.dump(model_package.metadata, f, indent=2)

print("✅ Metadata saved to 'model_metadata.json'")

# Get file size
import os
file_size = os.path.getsize('adaptive_quiz_model.pkl') / (1024 * 1024)  # MB
print(f"\n📦 Model file size: {file_size:.2f} MB")

# ============================================================================
# CELL 10: Test Loading the Model (Verification)
# ============================================================================
print("\n" + "="*60)
print("Testing Model Loading...")
print("="*60)

# Load the model
with open('adaptive_quiz_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("✅ Model loaded successfully!")
print(f"Model type: {loaded_model.metadata['model_type']}")
print(f"Number of questions: {loaded_model.metadata['n_questions']}")
print(f"Test accuracy: {loaded_model.metadata['test_accuracy']:.4f}")

# Test adaptive quiz creation
test_adaptive = loaded_model.create_adaptive_quiz()
print("\n✅ Adaptive quiz instance created!")

# Test section quiz creation
test_section = loaded_model.create_section_quiz()
print("✅ Section quiz instance created!")

# Test getting random questions
random_questions = loaded_model.get_random_questions(n=5)
print(f"\n✅ Retrieved {len(random_questions)} random questions")
print(f"First question ID: {random_questions[0]['id']}")

# ============================================================================
# CELL 11: Example Usage Demo
# ============================================================================
print("\n" + "="*60)
print("DEMO: Adaptive Quiz Flow")
print("="*60)

# Create adaptive quiz
demo_quiz = loaded_model.create_adaptive_quiz()

# Simulate 5 questions
for i in range(5):
    question = demo_quiz.select_next_question(mode='adaptive')
    if question is None:
        break
    
    print(f"\nQuestion {i+1}:")
    print(f"Difficulty: {question['difficulty']}")
    print(f"Current User Ability: {demo_quiz.user_ability:.2f}")
    
    # Simulate answer (randomly correct/incorrect)
    user_answer = np.random.randint(0, 4)
    is_correct, new_ability = demo_quiz.submit_answer(
        question['id'], 
        user_answer, 
        question['answer_numeric']
    )
    
    print(f"Answer: {'✅ Correct' if is_correct else '❌ Wrong'}")
    print(f"Updated Ability: {new_ability:.2f}")

stats = demo_quiz.get_stats()
print(f"\n📊 Final Stats:")
print(f"Total Questions: {stats['total_questions']}")
print(f"Correct Answers: {stats['correct_answers']}")
print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"Final Ability: {stats['current_ability']:.2f}")

print("\n" + "="*60)
print("✨ MODEL TRAINING COMPLETE! ✨")
print("="*60)
print("\nFiles created:")
print("  📦 adaptive_quiz_model.pkl (ready for Flask)")
print("  📄 model_metadata.json (model info)")
print("\nNext step: Integrate with Flask API!")