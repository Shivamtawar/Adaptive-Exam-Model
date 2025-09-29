"""
Flask API for Adaptive Quiz System
Save as: app.py
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    """Handle numpy types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration
app.json = NumpyJSONProvider(app)

# ============================================================================
# DEFINE CLASSES BEFORE LOADING MODEL (CRITICAL!)
# ============================================================================

class AdaptiveQuizSystem:
    """Adaptive Quiz System using Item Response Theory (IRT)"""
    
    def __init__(self, questions_df, difficulty_model):
        self.questions_df = questions_df.copy()
        self.difficulty_model = difficulty_model
        self.user_ability = 0.0
        self.response_history = []
        self.asked_questions = set()
        
    def estimate_ability(self, responses):
        if not responses:
            return 0.0
        
        def negative_log_likelihood(ability):
            nll = 0
            for difficulty, correct in responses:
                prob = expit(ability - difficulty)
                if correct:
                    nll -= np.log(prob + 1e-10)
                else:
                    nll -= np.log(1 - prob + 1e-10)
            return nll
        
        result = minimize(negative_log_likelihood, x0=0.0, method='BFGS')
        return result.x[0]
    
    def get_question_probability(self, question_difficulty, user_ability):
        return expit(user_ability - question_difficulty)
    
    def select_next_question(self, mode='adaptive', target_difficulty=None):
        available_questions = self.questions_df[
            ~self.questions_df['id'].isin(self.asked_questions)
        ].copy()
        
        if len(available_questions) == 0:
            return None
        
        if mode == 'adaptive':
            available_questions['score'] = abs(
                available_questions['difficulty_numeric'] - self.user_ability
            )
            available_questions['score'] += np.random.normal(0, 0.1, len(available_questions))
            next_question = available_questions.nsmallest(1, 'score').iloc[0]
        elif mode == 'fixed' and target_difficulty is not None:
            difficulty_questions = available_questions[
                available_questions['difficulty_numeric'] == target_difficulty
            ]
            if len(difficulty_questions) == 0:
                next_question = available_questions.sample(1).iloc[0]
            else:
                next_question = difficulty_questions.sample(1).iloc[0]
        else:
            next_question = available_questions.sample(1).iloc[0]
        
        return next_question
    
    def submit_answer(self, question_id, user_answer, correct_answer):
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
        responses = [(r['difficulty'], r['correct']) for r in self.response_history]
        self.user_ability = self.estimate_ability(responses)
        
        return is_correct, self.user_ability
    
    def get_stats(self):
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


class SectionBasedQuizSystem:
    """Section-based quiz with progression rules"""
    
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
        if section_num >= len(self.sections):
            return None
        
        self.current_section = section_num
        self.section_questions = []
        self.section_answers = []
        
        section_difficulty = self.sections[section_num]['difficulty']
        available = self.questions_df[
            (self.questions_df['difficulty_numeric'] == section_difficulty) &
            (~self.questions_df['id'].isin(self.asked_question_ids))
        ]
        
        if len(available) < 10:
            questions = available
        else:
            questions = available.sample(10, random_state=None)
        
        self.section_questions = questions.to_dict('records')
        for q in self.section_questions:
            self.asked_question_ids.add(q['id'])
        
        return self.section_questions
    
    def submit_section_answer(self, question_index, user_answer):
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
        if len(self.section_answers) < 10:
            return False, False, False
        
        correct_count = sum(1 for a in self.section_answers if a['is_correct'])
        threshold = self.sections[self.current_section]['threshold']
        last_wrong = not self.section_answers[-1]['is_correct']
        
        if correct_count >= threshold:
            return True, True, False
        elif correct_count == threshold - 1 and last_wrong:
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
            'section': self.current_section,
            'correct': sum(1 for a in self.section_answers if a['is_correct']),
            'total': len(self.section_answers),
            'passed': sum(1 for a in self.section_answers if a['is_correct']) >= 6
        })
        
        self.current_section += 1
        
        if self.current_section >= len(self.sections):
            return None
        
        return self.start_section(self.current_section)
    
    def get_progress(self):
        return {
            'current_section': self.current_section,
            'section_name': self.sections[self.current_section]['name'],
            'questions_answered': len(self.section_answers),
            'correct_in_section': sum(1 for a in self.section_answers if a['is_correct']),
            'completed_sections': self.completed_sections
        }


class QuizModelPackage:
    """Complete package for Flask API"""
    
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


# Load the model
print("Loading model...")
with open('adaptive_quiz_model.pkl', 'rb') as f:
    model_package = pickle.load(f)
print("✅ Model loaded successfully!")

# Store active quiz sessions (in production, use Redis or database)
active_sessions = {}

# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/')
def home():
    """Serve the main quiz interface"""
    return render_template('index.html')

@app.route('/adaptive')
def adaptive_page():
    """Adaptive quiz page"""
    return render_template('adaptive.html')

@app.route('/section')
def section_page():
    """Section-based quiz page"""
    return render_template('section.html')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())

def format_question(question_dict):
    """Format question for API response"""
    return {
        'id': int(question_dict['id']),
        'question': question_dict['question_text'],
        'options': {
            'a': question_dict['option_a'],
            'b': question_dict['option_b'],
            'c': question_dict['option_c'],
            'd': question_dict['option_d']
        },
        'difficulty': question_dict['difficulty']
    }

# ============================================================================
# ADAPTIVE QUIZ ENDPOINTS
# ============================================================================

@app.route('/api/adaptive/start', methods=['POST'])
def start_adaptive_quiz():
    """
    Start a new adaptive quiz session
    Body: { "user_id": "optional_user_id" }
    """
    try:
        data = request.json or {}
        session_id = create_session_id()
        
        # Create new adaptive quiz instance
        quiz = model_package.create_adaptive_quiz()
        
        # Store session
        active_sessions[session_id] = {
            'type': 'adaptive',
            'quiz': quiz,
            'user_id': data.get('user_id'),
            'started_at': datetime.now().isoformat(),
            'time_limit': 3600  # 1 hour in seconds
        }
        
        # Get first question
        first_question = quiz.select_next_question(mode='adaptive')
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'question': format_question(first_question),
            'user_ability': quiz.user_ability,
            'time_limit': 3600
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/adaptive/submit', methods=['POST'])
def submit_adaptive_answer():
    """
    Submit answer for adaptive quiz
    Body: {
        "session_id": "session_id",
        "question_id": 123,
        "answer": "a"  # or 0 for numeric
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        question_id = data.get('question_id')
        user_answer = data.get('answer')
        
        # Validate session
        if session_id not in active_sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session ID'
            }), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        
        # Convert answer to numeric if it's a letter
        if isinstance(user_answer, str):
            answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            user_answer = answer_map.get(user_answer.lower(), 0)
        
        # Get correct answer
        question = model_package.get_question_by_id(question_id)
        if question is None:
            return jsonify({
                'success': False,
                'error': 'Question not found'
            }), 404
        
        correct_answer = question['answer_numeric']
        
        # Submit answer
        is_correct, new_ability = quiz.submit_answer(
            question_id, user_answer, correct_answer
        )
        
        # Get next question
        next_question = quiz.select_next_question(mode='adaptive')
        
        # Get stats
        stats = quiz.get_stats()
        
        response = {
            'success': True,
            'is_correct': is_correct,
            'correct_answer': ['a', 'b', 'c', 'd'][correct_answer],
            'user_ability': new_ability,
            'stats': stats
        }
        
        # Add next question if available
        if next_question is not None:
            response['next_question'] = format_question(next_question)
        else:
            response['quiz_complete'] = True
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/adaptive/stats/<session_id>', methods=['GET'])
def get_adaptive_stats(session_id):
    """Get current quiz statistics"""
    try:
        if session_id not in active_sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session ID'
            }), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        stats = quiz.get_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# SECTION-BASED QUIZ ENDPOINTS
# ============================================================================

@app.route('/api/section/start', methods=['POST'])
def start_section_quiz():
    """
    Start a new section-based quiz
    Body: { "user_id": "optional_user_id" }
    """
    try:
        data = request.json or {}
        session_id = create_session_id()
        
        # Create new section quiz instance
        quiz = model_package.create_section_quiz()
        
        # Start first section
        questions = quiz.start_section(0)
        
        # Store session
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/section/submit', methods=['POST'])
def submit_section_answer():
    """
    Submit answer for section-based quiz
    Body: {
        "session_id": "session_id",
        "answer": "a"  # or 0 for numeric
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        user_answer = data.get('answer')
        
        # Validate session
        if session_id not in active_sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session ID'
            }), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        current_index = session['current_question_index']
        
        # Convert answer to numeric if it's a letter
        if isinstance(user_answer, str):
            answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            user_answer = answer_map.get(user_answer.lower(), 0)
        
        # Submit answer
        is_correct = quiz.submit_section_answer(current_index, user_answer)
        
        # Check section completion
        is_complete, passed, needs_11th = quiz.check_section_completion()
        
        response = {
            'success': True,
            'is_correct': is_correct,
            'question_index': current_index
        }
        
        if needs_11th:
            # Need 11th question
            question_11 = quiz.get_11th_question()
            response['needs_11th_question'] = True
            response['question_11'] = format_question(question_11)
            session['current_question_index'] += 1
            
        elif is_complete:
            # Section complete
            response['section_complete'] = True
            response['section_passed'] = passed
            
            if passed:
                # Move to next section
                next_questions = quiz.proceed_to_next_section()
                if next_questions is not None:
                    response['next_section'] = quiz.current_section
                    response['next_question'] = format_question(next_questions[0])
                    session['current_question_index'] = 0
                else:
                    response['quiz_complete'] = True
            else:
                response['quiz_failed'] = True
        else:
            # Continue with next question in section
            session['current_question_index'] += 1
            next_index = session['current_question_index']
            
            if next_index < len(quiz.section_questions):
                response['next_question'] = format_question(
                    quiz.section_questions[next_index]
                )
        
        # Add progress
        response['progress'] = quiz.get_progress()
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/section/progress/<session_id>', methods=['GET'])
def get_section_progress(session_id):
    """Get section quiz progress"""
    try:
        if session_id not in active_sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session ID'
            }), 404
        
        session = active_sessions[session_id]
        quiz = session['quiz']
        progress = quiz.get_progress()
        
        return jsonify({
            'success': True,
            'progress': progress
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# GENERAL ENDPOINTS
# ============================================================================

@app.route('/api/questions/random', methods=['GET'])
def get_random_questions():
    """
    Get random questions
    Query params: ?n=10&difficulty=1
    """
    try:
        n = int(request.args.get('n', 10))
        difficulty = request.args.get('difficulty')
        
        if difficulty is not None:
            difficulty = int(difficulty)
        
        questions = model_package.get_random_questions(n=n, difficulty=difficulty)
        
        formatted_questions = [format_question(q) for q in questions]
        
        return jsonify({
            'success': True,
            'questions': formatted_questions,
            'count': len(formatted_questions)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/question/<int:question_id>', methods=['GET'])
def get_question_by_id(question_id):
    """Get specific question by ID"""
    try:
        question = model_package.get_question_by_id(question_id)
        
        if question is None:
            return jsonify({
                'success': False,
                'error': 'Question not found'
            }), 404
        
        return jsonify({
            'success': True,
            'question': format_question(question)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model metadata and info"""
    try:
        return jsonify({
            'success': True,
            'metadata': model_package.metadata,
            'active_sessions': len(active_sessions)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/session/end/<session_id>', methods=['DELETE'])
def end_session(session_id):
    """End a quiz session and clean up"""
    try:
        if session_id in active_sessions:
            session = active_sessions[session_id]
            quiz = session['quiz']
            
            # Get final stats
            if session['type'] == 'adaptive':
                final_stats = quiz.get_stats()
            else:
                final_stats = quiz.get_progress()
            
            # Remove session
            del active_sessions[session_id]
            
            return jsonify({
                'success': True,
                'message': 'Session ended',
                'final_stats': final_stats
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'model_loaded': model_package is not None,
        'active_sessions': len(active_sessions)
    }), 200

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Adaptive Quiz API Server")
    print("="*60)
    print(f"Model: {model_package.metadata['model_type']}")
    print(f"Questions: {model_package.metadata['n_questions']}")
    print(f"Accuracy: {model_package.metadata['test_accuracy']:.2%}")
    print("="*60)
    print("\n📡 Starting server on http://localhost:5000")
    print("\n📚 API Endpoints:")
    print("  Adaptive Quiz:")
    print("    POST   /api/adaptive/start")
    print("    POST   /api/adaptive/submit")
    print("    GET    /api/adaptive/stats/<session_id>")
    print("\n  Section Quiz:")
    print("    POST   /api/section/start")
    print("    POST   /api/section/submit")
    print("    GET    /api/section/progress/<session_id>")
    print("\n  General:")
    print("    GET    /api/questions/random")
    print("    GET    /api/question/<id>")
    print("    GET    /api/model/info")
    print("    DELETE /api/session/end/<session_id>")
    print("    GET    /api/health")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)