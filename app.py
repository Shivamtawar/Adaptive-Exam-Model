"""
Flask API for Adaptive Quiz System (FIXED for String IDs)
Save as: app.py

Fixes:
- Handles string IDs (e.g., 'Q000321') instead of requiring integers
- Properly converts NumPy types to native Python types
- Ensures all analytics data is JSON serializable
- Handles empty/NaN values gracefully
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.special import expit
import json
import dill
import sys
import types

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)

# Helper function to convert data to JSON-safe format
def make_json_safe(obj):
    """Recursively convert NumPy types to native Python types"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return 0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

# Helper function to safely convert to int (handles both string and numeric IDs)
def safe_to_int(value):
    """Safely convert value to int, handling string IDs"""
    if isinstance(value, (int, np.integer)):
        return int(value)
    elif isinstance(value, str):
        # If it's a string ID like 'Q000321', keep it as string
        return value
    else:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value

# ============================================================================
# DEFINE CLASSES
# ============================================================================

class AdaptiveQuizSystem:
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
        difficulty = int(question['difficulty_numeric'])
        
        self.response_history.append({
            'question_id': str(question_id),  # Keep as string
            'difficulty': int(difficulty),
            'correct': bool(is_correct),
            'user_answer': int(user_answer),
            'correct_answer': int(correct_answer),
            'time_spent': float(time_spent),
            'timestamp': datetime.now().isoformat()
        })
        
        self.question_times.append(float(time_spent))
        self.asked_questions.add(question_id)
        
        responses = [(r['difficulty'], r['correct']) for r in self.response_history]
        self.user_ability = self.estimate_ability(responses)
        
        return is_correct, self.user_ability
    
    def get_analytics(self):
        if not self.response_history:
            return {
                'total_questions': 0,
                'correct_answers': 0,
                'wrong_answers': 0,
                'accuracy': 0.0,
                'total_time_seconds': 0,
                'average_time_per_question': 0.0,
                'fastest_answer': 0.0,
                'slowest_answer': 0.0,
                'current_ability': float(self.user_ability),
                'difficulty_breakdown': {},
                'response_history': [],
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
        
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
                difficulty_stats[str(diff)] = {
                    'attempted': int(len(results)),
                    'correct': int(sum(results)),
                    'accuracy': float(sum(results) / len(results))
                }
        
        return {
            'total_questions': int(total),
            'correct_answers': int(correct),
            'wrong_answers': int(total - correct),
            'accuracy': float(correct / total if total > 0 else 0.0),
            'total_time_seconds': int(total_time),
            'average_time_per_question': float(round(avg_time, 2)),
            'fastest_answer': float(min(self.question_times)),
            'slowest_answer': float(max(self.question_times)),
            'current_ability': float(self.user_ability),
            'difficulty_breakdown': difficulty_stats,
            'response_history': self.response_history,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }


class SectionBasedQuizSystem:
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
        self.all_answers = []
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
    
    def submit_section_answer(self, question_index, user_answer, time_spent):
        if question_index >= len(self.section_questions):
            return None
        
        question = self.section_questions[question_index]
        correct_answer = int(question['answer_numeric'])
        is_correct = (user_answer == correct_answer)
        
        answer_record = {
            'question_index': int(question_index),
            'question_id': str(question['id']),  # Keep as string
            'user_answer': int(user_answer),
            'correct_answer': int(correct_answer),
            'is_correct': bool(is_correct),
            'time_spent': float(time_spent),
            'difficulty': int(question['difficulty_numeric']),
            'section': int(self.current_section),
            'timestamp': datetime.now().isoformat()
        }
        
        self.section_answers.append(answer_record)
        self.all_answers.append(answer_record)
        self.question_times.append(float(time_spent))
        
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
            'section': int(self.current_section),
            'section_name': self.sections[self.current_section]['name'],
            'correct': int(sum(1 for a in self.section_answers if a['is_correct'])),
            'total': int(len(self.section_answers)),
            'passed': bool(sum(1 for a in self.section_answers if a['is_correct']) >= 6)
        })
        
        self.current_section += 1
        self.section_answers = []
        
        if self.current_section >= len(self.sections):
            return None
        
        return self.start_section(self.current_section)
    
    def get_analytics(self):
        if not self.all_answers:
            return {
                'total_questions': 0,
                'correct_answers': 0,
                'wrong_answers': 0,
                'accuracy': 0.0,
                'total_time_seconds': 0,
                'average_time_per_question': 0.0,
                'fastest_answer': 0.0,
                'slowest_answer': 0.0,
                'sections_completed': 0,
                'sections_passed': 0,
                'section_breakdown': [],
                'response_history': [],
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
        
        total_questions = len(self.all_answers)
        total_correct = sum(1 for a in self.all_answers if a['is_correct'])
        total_time = sum(self.question_times)
        
        return {
            'total_questions': int(total_questions),
            'correct_answers': int(total_correct),
            'wrong_answers': int(total_questions - total_correct),
            'accuracy': float(total_correct / total_questions if total_questions > 0 else 0.0),
            'total_time_seconds': int(total_time),
            'average_time_per_question': float(round(total_time / total_questions, 2) if total_questions > 0 else 0),
            'fastest_answer': float(min(self.question_times)) if self.question_times else 0.0,
            'slowest_answer': float(max(self.question_times)) if self.question_times else 0.0,
            'sections_completed': int(len(self.completed_sections)),
            'sections_passed': int(sum(1 for s in self.completed_sections if s['passed'])),
            'section_breakdown': self.completed_sections,
            'response_history': self.all_answers,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }


class QuizModelPackage:
    def __init__(self, difficulty_model, questions_df, feature_cols, scaler=None):
        self.difficulty_model = difficulty_model
        self.questions_df = questions_df
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.metadata = {}
    
    def create_adaptive_quiz(self):
        return AdaptiveQuizSystem(self.questions_df, self.difficulty_model)
    
    def create_section_quiz(self):
        return SectionBasedQuizSystem(self.questions_df)
    
    def get_question_by_id(self, question_id):
        question = self.questions_df[self.questions_df['id'] == question_id]
        if len(question) == 0:
            return None
        return question.iloc[0].to_dict()


# Register classes for dill deserialization
def register_classes():
    # Create a module for deserialization compatibility
    if 'quiz_model_package' not in sys.modules:
        module = types.ModuleType('quiz_model_package')
        sys.modules['quiz_model_package'] = module
    else:
        module = sys.modules['quiz_model_package']
    
    # Assign classes to the module
    module.QuizModelPackage = QuizModelPackage
    module.AdaptiveQuizSystem = AdaptiveQuizSystem
    module.SectionBasedQuizSystem = SectionBasedQuizSystem
    
    # Also register in __main__ to handle Gunicorn's context
    if '__main__' not in sys.modules:
        sys.modules['__main__'] = sys.modules[__name__]
    sys.modules['__main__'].QuizModelPackage = QuizModelPackage
    sys.modules['__main__'].AdaptiveQuizSystem = AdaptiveQuizSystem
    sys.modules['__main__'].SectionBasedQuizSystem = SectionBasedQuizSystem

# Call the registration before loading the pickle file
register_classes()

# Load model
print("Loading model...")
with open("adaptive_quiz_model.pkl", "rb") as f:
    model_package = dill.load(f)
print("Model loaded!")

active_sessions = {}

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/adaptive')
def adaptive_page():
    return render_template('adaptive.html')

@app.route('/section')
def section_page():
    return render_template('section.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

def create_session_id():
    return str(uuid.uuid4())

def format_question(question_dict):
    """Format question for API response - handles string IDs"""
    return {
        'id': str(question_dict['id']),  # Keep as string
        'question': str(question_dict['question_text']),
        'options': {
            'a': str(question_dict['option_a']),
            'b': str(question_dict['option_b']),
            'c': str(question_dict['option_c']),
            'd': str(question_dict['option_d'])
        },
        'difficulty': str(question_dict['difficulty']),
        'difficulty_numeric': int(question_dict['difficulty_numeric']),
        'answer_numeric': int(question_dict['answer_numeric'])
    }

# ============================================================================
# ADAPTIVE QUIZ ENDPOINTS
# ============================================================================

@app.route('/api/adaptive/start', methods=['POST'])
def start_adaptive_quiz():
    try:
        data = request.json or {}
        session_id = create_session_id()
        
        quiz = model_package.create_adaptive_quiz()
        
        active_sessions[session_id] = {
            'type': 'adaptive',
            'quiz': quiz,
            'user_id': data.get('user_id'),
            'started_at': datetime.now().isoformat()
        }
        
        first_question = quiz.select_next_question(mode='adaptive')
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'question': format_question(first_question),
            'user_ability': float(quiz.user_ability)
        }), 200
        
    except Exception as e:
        print(f"Error in start_adaptive_quiz: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/adaptive/submit', methods=['POST'])
def submit_adaptive_answer():
    try:
        data = request.json
        session_id = data.get('session_id')
        question_id = data.get('question_id')  # Now a string
        user_answer = data.get('answer')
        time_spent = data.get('time_spent', 0)
        
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        
        if isinstance(user_answer, str):
            answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            user_answer = answer_map.get(user_answer.lower(), 0)
        
        question = model_package.get_question_by_id(question_id)
        if question is None:
            return jsonify({'success': False, 'error': 'Question not found'}), 404
        
        correct_answer = int(question['answer_numeric'])
        
        is_correct, new_ability = quiz.submit_answer(
            question_id, user_answer, correct_answer, time_spent
        )
        
        next_question = quiz.select_next_question(mode='adaptive')
        
        response = {
            'success': True,
            'is_correct': bool(is_correct),
            'correct_answer': ['a', 'b', 'c', 'd'][int(correct_answer)],
            'user_ability': float(new_ability)
        }
        
        if next_question is not None:
            response['next_question'] = format_question(next_question)
        else:
            response['quiz_complete'] = True
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in submit_adaptive_answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/adaptive/analytics/<session_id>', methods=['GET'])
def get_adaptive_analytics(session_id):
    try:
        print(f"\n=== Analytics Request for Session: {session_id} ===")
        
        if session_id not in active_sessions:
            print(f"Session not found in active_sessions")
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        
        print(f"Quiz type: {session['type']}")
        print(f"Response history length: {len(quiz.response_history)}")
        
        analytics = quiz.get_analytics()
        
        # Convert to JSON-safe format
        analytics = make_json_safe(analytics)
        
        print(f"Analytics generated successfully")
        print(f"Total questions: {analytics.get('total_questions')}")
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'quiz_type': 'adaptive'
        }), 200
        
    except Exception as e:
        print(f"Error in get_adaptive_analytics: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# SECTION QUIZ ENDPOINTS
# ============================================================================

@app.route('/api/section/start', methods=['POST'])
def start_section_quiz():
    try:
        data = request.json or {}
        session_id = create_session_id()
        
        quiz = model_package.create_section_quiz()
        questions = quiz.start_section(0)
        
        active_sessions[session_id] = {
            'type': 'section',
            'quiz': quiz,
            'user_id': data.get('user_id'),
            'started_at': datetime.now().isoformat(),
            'current_question_index': 0
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'section': 0,
            'section_name': quiz.sections[0]['name'],
            'question': format_question(questions[0]),
            'total_questions_in_section': len(questions),
            'passing_threshold': quiz.sections[0]['threshold']
        }), 200
        
    except Exception as e:
        print(f"Error in start_section_quiz: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/section/submit', methods=['POST'])
def submit_section_answer():
    try:
        data = request.json
        session_id = data.get('session_id')
        user_answer = data.get('answer')
        time_spent = data.get('time_spent', 0)
        
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        current_index = session['current_question_index']
        
        if isinstance(user_answer, str):
            answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            user_answer = answer_map.get(user_answer.lower(), 0)
        
        is_correct = quiz.submit_section_answer(current_index, user_answer, time_spent)
        
        correct_count = sum(1 for a in quiz.section_answers if a['is_correct'])
        total_answered = len(quiz.section_answers)
        
        response = {
            'success': True,
            'is_correct': is_correct,
            'question_index': current_index,
            'correct_count': correct_count,
            'total_answered': total_answered,
            'threshold': quiz.sections[quiz.current_section]['threshold'],
            'progress': {
                'correct_in_section': correct_count,
                'questions_answered': total_answered
            }
        }
        
        if total_answered >= 10:
            is_complete, passed, needs_11th = quiz.check_section_completion()
            
            if needs_11th:
                question_11 = quiz.get_11th_question()
                if question_11 is not None:
                    response['needs_11th_question'] = True
                    response['question_11'] = format_question(question_11)
                    session['current_question_index'] += 1
                else:
                    response['section_exhausted'] = True
                
            elif is_complete:
                response['section_complete'] = True
                response['section_passed'] = passed
                
                if passed:
                    next_questions = quiz.proceed_to_next_section()
                    if next_questions is not None:
                        response['next_section'] = quiz.current_section
                        response['section_name'] = quiz.sections[quiz.current_section]['name']
                        response['next_question'] = format_question(next_questions[0])
                        session['current_question_index'] = 0
                    else:
                        response['quiz_complete'] = True
                else:
                    new_questions = quiz.reload_section_questions()
                    if new_questions is not None:
                        response['section_failed'] = True
                        response['reload_section'] = True
                        response['next_question'] = format_question(new_questions[0])
                        session['current_question_index'] = 0
                    else:
                        response['section_exhausted'] = True
                        response['quiz_failed'] = True
        else:
            session['current_question_index'] += 1
            next_index = session['current_question_index']
            
            if next_index < len(quiz.section_questions):
                response['next_question'] = format_question(quiz.section_questions[next_index])
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in submit_section_answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/section/analytics/<session_id>', methods=['GET'])
def get_section_analytics(session_id):
    try:
        print(f"\n=== Section Analytics Request for Session: {session_id} ===")
        
        if session_id not in active_sessions:
            print(f"Session not found")
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        
        print(f"All answers length: {len(quiz.all_answers)}")
        print(f"Completed sections: {len(quiz.completed_sections)}")
        
        analytics = quiz.get_analytics()
        
        # Convert to JSON-safe format
        analytics = make_json_safe(analytics)
        
        print(f"Analytics generated successfully")
        print(f"Total questions: {analytics.get('total_questions')}")
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'quiz_type': 'section'
        }), 200
        
    except Exception as e:
        print(f"Error in get_section_analytics: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'model_questions': len(model_package.questions_df)
    }), 200


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ADAPTIVE QUIZ SYSTEM - Flask API (FIXED)")
    print("="*60)
    print("\nFixes:")
    print("  - All NumPy types converted to native Python types")
    print("  - Analytics properly serialized to JSON")
    print("  - Section quiz tracks all answers across sections")
    print("  - Better error handling and debugging")
    print("\nServer starting on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)