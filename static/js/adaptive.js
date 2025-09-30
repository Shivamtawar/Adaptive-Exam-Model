// Adaptive Quiz JavaScript
const API_BASE = 'https://adaptive-exam-model.onrender.com/api';

let sessionId = null;
let currentQuestion = null;
let selectedAnswer = null;
let questionCount = 0;
let correctCount = 0;
let timerInterval = null;
let timeRemaining = 3600; // 1 hour

// Initialize quiz on page load
document.addEventListener('DOMContentLoaded', () => {
    startQuiz();
    startTimer();
});

// Start the quiz
async function startQuiz() {
    try {
        const response = await fetch(`${API_BASE}/adaptive/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'user_' + Date.now() })
        });
        
        const data = await response.json();
        
        if (data.success) {
            sessionId = data.session_id;
            currentQuestion = data.question;
            displayQuestion(currentQuestion);
            updateAbility(data.user_ability);
        } else {
            alert('Error starting quiz: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to start quiz. Make sure the Flask server is running.');
    }
}

// Display question
function displayQuestion(question) {
    questionCount++;
    
    document.getElementById('questionNumber').textContent = `Question #${questionCount}`;
    document.getElementById('questionText').textContent = question.question;
    
    // Set difficulty badge
    const badge = document.getElementById('difficultyBadge');
    badge.textContent = question.difficulty;
    badge.className = 'difficulty-badge ' + question.difficulty.toLowerCase().replace(' ', '-');
    
    // Display options
    const optionsGrid = document.getElementById('optionsGrid');
    optionsGrid.innerHTML = '';
    
    const letters = ['a', 'b', 'c', 'd'];
    letters.forEach((letter, index) => {
        const optionBtn = document.createElement('div');
        optionBtn.className = 'option-btn';
        optionBtn.onclick = () => selectOption(letter, optionBtn);
        
        optionBtn.innerHTML = `
            <div class="option-letter">${letter.toUpperCase()}</div>
            <div class="option-text">${question.options[letter]}</div>
        `;
        
        optionsGrid.appendChild(optionBtn);
    });
    
    // Reset selection
    selectedAnswer = null;
    document.getElementById('submitBtn').disabled = true;
    
    // Update stats
    document.getElementById('questionCount').textContent = questionCount;
}

// Select option
function selectOption(letter, element) {
    // Remove previous selection
    document.querySelectorAll('.option-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    // Add selection
    element.classList.add('selected');
    selectedAnswer = letter;
    document.getElementById('submitBtn').disabled = false;
}

// Submit answer
async function submitAnswer() {
    if (!selectedAnswer) return;
    
    document.getElementById('submitBtn').disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/adaptive/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                question_id: currentQuestion.id,
                answer: selectedAnswer
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update correct count
            if (data.is_correct) {
                correctCount++;
            }
            
            // Update stats
            updateStats(data.stats);
            updateAbility(data.user_ability);
            
            // Show feedback
            showFeedback(data.is_correct, data.correct_answer);
            
            // Store next question
            if (data.next_question) {
                currentQuestion = data.next_question;
            } else {
                // Quiz complete
                currentQuestion = null;
            }
        } else {
            alert('Error submitting answer: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to submit answer.');
    }
}

// Show feedback modal
function showFeedback(isCorrect, correctAnswer) {
    const modal = document.getElementById('feedbackModal');
    const icon = document.getElementById('feedbackIcon');
    const title = document.getElementById('feedbackTitle');
    const message = document.getElementById('feedbackMessage');
    
    icon.className = 'feedback-icon ' + (isCorrect ? 'correct' : 'wrong');
    title.textContent = isCorrect ? 'Correct! 🎉' : 'Incorrect';
    message.textContent = isCorrect 
        ? 'Great job! Moving to the next question...' 
        : `The correct answer was: ${correctAnswer.toUpperCase()}`;
    
    modal.classList.add('show');
}

// Next question
function nextQuestion() {
    const modal = document.getElementById('feedbackModal');
    modal.classList.remove('show');
    
    if (currentQuestion) {
        displayQuestion(currentQuestion);
    } else {
        showResults();
    }
}

// Update stats
function updateStats(stats) {
    document.getElementById('questionCount').textContent = stats.total_questions;
    document.getElementById('correctCount').textContent = stats.correct_answers;
    document.getElementById('accuracy').textContent = (stats.accuracy * 100).toFixed(0) + '%';
}

// Update ability meter
function updateAbility(ability) {
    const fill = document.getElementById('abilityFill');
    const value = document.getElementById('abilityValue');
    
    // Map ability (-3 to 3) to percentage (0 to 100)
    const percentage = ((ability + 3) / 6) * 100;
    fill.style.width = percentage + '%';
    value.textContent = ability.toFixed(2);
}

// Timer
function startTimer() {
    timerInterval = setInterval(() => {
        timeRemaining--;
        
        const minutes = Math.floor(timeRemaining / 60);
        const seconds = timeRemaining % 60;
        
        document.getElementById('timer').textContent = 
            `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        if (timeRemaining <= 0) {
            clearInterval(timerInterval);
            showResults();
        }
    }, 1000);
}

// Show final results
async function showResults() {
    clearInterval(timerInterval);
    
    try {
        const response = await fetch(`${API_BASE}/adaptive/stats/${sessionId}`);
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            
            document.getElementById('totalQuestions').textContent = stats.total_questions;
            document.getElementById('totalCorrect').textContent = stats.correct_answers;
            document.getElementById('finalAccuracy').textContent = (stats.accuracy * 100).toFixed(1) + '%';
            document.getElementById('finalAbility').textContent = stats.current_ability.toFixed(2);
            
            document.getElementById('resultsModal').classList.add('show');
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key >= '1' && e.key <= '4') {
        const options = document.querySelectorAll('.option-btn');
        const index = parseInt(e.key) - 1;
        if (options[index]) {
            options[index].click();
        }
    } else if (e.key === 'Enter' && !document.getElementById('submitBtn').disabled) {
        submitAnswer();
    }
});