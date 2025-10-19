// Adaptive Quiz JavaScript
const API_BASE = 'http://localhost:5000/api';

let questionStartTime = null;
let sessionId = null;
let currentQuestion = null;
let selectedAnswer = null;
let questionCount = 0;
let correctCount = 0;
let timerInterval = null;
let timeRemaining = 3600;

document.addEventListener('DOMContentLoaded', () => {
    startQuiz();
    startTimer();
});

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
        alert('Failed to start quiz.');
    }
    questionStartTime = Date.now();
}

function displayQuestion(question) {
    questionCount++;
    document.getElementById('questionNumber').textContent = `Question #${questionCount}`;
    document.getElementById('questionText').textContent = question.question;
    
    const badge = document.getElementById('difficultyBadge');
    badge.textContent = question.difficulty;
    badge.className = 'difficulty-badge ' + question.difficulty.toLowerCase().replace(' ', '-');
    
    const optionsGrid = document.getElementById('optionsGrid');
    optionsGrid.innerHTML = '';
    
    const letters = ['a', 'b', 'c', 'd'];
    letters.forEach((letter) => {
        const optionBtn = document.createElement('div');
        optionBtn.className = 'option-btn';
        optionBtn.onclick = () => selectOption(letter, optionBtn);
        optionBtn.innerHTML = `
            <div class="option-letter">${letter.toUpperCase()}</div>
            <div class="option-text">${question.options[letter]}</div>
        `;
        optionsGrid.appendChild(optionBtn);
    });
    
    selectedAnswer = null;
    document.getElementById('submitBtn').disabled = true;
    document.getElementById('questionCount').textContent = questionCount;
    document.getElementById('correctCount').textContent = correctCount;
    const accuracy = questionCount > 0 ? (correctCount / questionCount * 100) : 0;
    document.getElementById('accuracy').textContent = accuracy.toFixed(0) + '%';
    questionStartTime = Date.now();
}

function selectOption(letter, element) {
    document.querySelectorAll('.option-btn').forEach(btn => btn.classList.remove('selected'));
    element.classList.add('selected');
    selectedAnswer = letter;
    document.getElementById('submitBtn').disabled = false;
}

async function submitAnswer() {
    if (!selectedAnswer) return;
    
    const timeSpent = (Date.now() - questionStartTime) / 1000;
    document.getElementById('submitBtn').disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/adaptive/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                question_id: currentQuestion.id,
                answer: selectedAnswer,
                time_spent: timeSpent
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.is_correct) correctCount++;
            document.getElementById('correctCount').textContent = correctCount;
            const accuracy = questionCount > 0 ? (correctCount / questionCount * 100) : 0;
            document.getElementById('accuracy').textContent = accuracy.toFixed(0) + '%';
            updateAbility(data.user_ability);
            showFeedback(data.is_correct, data.correct_answer);
            
            if (data.next_question) {
                currentQuestion = data.next_question;
            } else {
                currentQuestion = null;
                sessionStorage.setItem('lastSessionId', sessionId);
                sessionStorage.setItem('lastQuizType', 'adaptive');
            }
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to submit answer.');
    }
}

function showFeedback(isCorrect, correctAnswer) {
    const modal = document.getElementById('feedbackModal');
    const icon = document.getElementById('feedbackIcon');
    const title = document.getElementById('feedbackTitle');
    const message = document.getElementById('feedbackMessage');
    
    icon.className = 'feedback-icon ' + (isCorrect ? 'correct' : 'wrong');
    title.textContent = isCorrect ? 'Correct! 🎉' : 'Incorrect';
    message.textContent = isCorrect ? 'Great job!' : `Correct answer: ${correctAnswer.toUpperCase()}`;
    modal.classList.add('show');
}

function nextQuestion() {
    const modal = document.getElementById('feedbackModal');
    modal.classList.remove('show');
    
    if (currentQuestion) {
        displayQuestion(currentQuestion);
    } else {
        showResults();
    }
}

function updateAbility(ability) {
    const fill = document.getElementById('abilityFill');
    const value = document.getElementById('abilityValue');
    const percentage = ((ability + 3) / 6) * 100;
    fill.style.width = percentage + '%';
    value.textContent = ability.toFixed(2);
}

function startTimer() {
    timerInterval = setInterval(() => {
        timeRemaining--;
        const minutes = Math.floor(timeRemaining / 60);
        const seconds = timeRemaining % 60;
        document.getElementById('timer').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        if (timeRemaining <= 0) {
            clearInterval(timerInterval);
            endExam();
        }
    }, 1000);
}

async function endExam() {
    if (!confirm('Are you sure you want to end the exam?')) return;
    clearInterval(timerInterval);
    if (sessionId) {
        sessionStorage.setItem('lastSessionId', sessionId);
        sessionStorage.setItem('lastQuizType', 'adaptive');
        setTimeout(() => showResults(), 500);
    }
}

function showResults() {
    if (!sessionId) {
        alert('No session ID found.');
        return;
    }
    clearInterval(timerInterval);
    window.location.href = `/analytics?session=${sessionId}&type=adaptive`;
}

document.addEventListener('keydown', (e) => {
    if (e.key >= '1' && e.key <= '4') {
        const options = document.querySelectorAll('.option-btn');
        const index = parseInt(e.key) - 1;
        if (options[index]) options[index].click();
    } else if (e.key === 'Enter' && !document.getElementById('submitBtn').disabled) {
        submitAnswer();
    }
});