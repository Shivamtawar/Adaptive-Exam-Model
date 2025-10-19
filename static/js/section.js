// Section-Based Quiz JavaScript
const API_BASE = 'https://adaptive-exam-model.onrender.com/api';

let questionStartTime = null;
let sessionId = null;
let currentQuestion = null;
let selectedAnswer = null;
let currentSection = 0;
let questionIndex = 0;
let sectionScore = 0;

document.addEventListener('DOMContentLoaded', () => {
    startQuiz();
});

async function startQuiz() {
    try {
        const response = await fetch(`${API_BASE}/section/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'user_' + Date.now() })
        });
        
        const data = await response.json();
        
        if (data.success) {
            sessionId = data.session_id;
            currentSection = data.section;
            currentQuestion = data.question;
            sessionStorage.setItem('lastSessionId', sessionId);
            sessionStorage.setItem('lastQuizType', 'section');
            updateSectionUI(data.section, data.section_name);
            displayQuestion(currentQuestion);
        } else {
            alert('Error starting quiz: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to start quiz.');
    }
    questionStartTime = Date.now();
}

function updateSectionUI(sectionNum, sectionName) {
    document.getElementById('sectionTitle').textContent = `Section ${sectionNum + 1}: ${sectionName}`;
    for (let i = 0; i < 4; i++) {
        const step = document.getElementById(`section${i}`);
        if (i < sectionNum) {
            step.classList.add('completed');
            step.classList.remove('active');
        } else if (i === sectionNum) {
            step.classList.add('active');
            step.classList.remove('completed');
        } else {
            step.classList.remove('completed', 'active');
        }
    }
}

function displayQuestion(question) {
    questionIndex++;
    document.getElementById('questionNumber').textContent = `Question #${questionIndex}`;
    document.getElementById('questionText').textContent = question.question;
    
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
    updateProgress();
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
        const response = await fetch(`${API_BASE}/section/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                answer: selectedAnswer,
                time_spent: timeSpent
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.is_correct) sectionScore++;
            updateProgress();
            showFeedback(data.is_correct);
            window.currentResponseData = data;
            
            if (data.quiz_complete || data.quiz_failed) {
                sessionStorage.setItem('lastSessionId', sessionId);
                sessionStorage.setItem('lastQuizType', 'section');
            }
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to submit answer.');
    }
}

function showFeedback(isCorrect) {
    const modal = document.getElementById('feedbackModal');
    const icon = document.getElementById('feedbackIcon');
    const title = document.getElementById('feedbackTitle');
    const message = document.getElementById('feedbackMessage');
    
    icon.className = 'feedback-icon ' + (isCorrect ? 'correct' : 'wrong');
    title.textContent = isCorrect ? 'Correct! ✅' : 'Incorrect ❌';
    message.textContent = isCorrect ? 'Great job!' : 'Keep trying!';
    modal.classList.add('show');
}

function handleNext() {
    const modal = document.getElementById('feedbackModal');
    modal.classList.remove('show');
    
    const data = window.currentResponseData;
    
    if (data.needs_11th_question) {
        currentQuestion = data.question_11;
        displayQuestion(currentQuestion);
    } else if (data.section_complete) {
        showSectionResult(data);
    } else if (data.next_question) {
        currentQuestion = data.next_question;
        displayQuestion(currentQuestion);
    }
}

function showSectionResult(data) {
    const modal = document.getElementById('sectionCompleteModal');
    const icon = document.getElementById('sectionIcon');
    const title = document.getElementById('sectionResultTitle');
    const message = document.getElementById('sectionResultMessage');
    
    icon.className = 'feedback-icon ' + (data.section_passed ? 'pass' : 'fail');
    title.textContent = data.section_passed ? 'Section Passed! 🎉' : 'Section Failed 😢';
    message.textContent = data.section_passed ? 'Congratulations!' : 'You need 6/10 to pass.';
    
    const correctInSection = data.progress?.correct_in_section || sectionScore;
    const questionsAnswered = data.progress?.questions_answered || questionIndex;
    
    document.getElementById('sectionCorrect').textContent = correctInSection;
    document.getElementById('sectionTotal').textContent = questionsAnswered;
    window.sectionCompleteData = data;
    modal.classList.add('show');
}

function handleSectionComplete() {
    const modal = document.getElementById('sectionCompleteModal');
    modal.classList.remove('show');
    
    const data = window.sectionCompleteData;
    
    if (data.quiz_complete) {
        showFinalResults(data);
    } else if (data.quiz_failed) {
        showFailedQuiz();
    } else if (data.next_section !== undefined) {
        currentSection = data.next_section;
        currentQuestion = data.next_question;
        questionIndex = 0;
        sectionScore = 0;
        updateSectionUI(currentSection, data.section_name);
        displayQuestion(currentQuestion);
    }
}

function updateProgress() {
    const progressText = `${questionIndex}/10`;
    const scoreText = `${sectionScore}/10`;
    const progressPercent = (questionIndex / 10) * 100;
    
    document.getElementById('questionProgress').textContent = progressText;
    document.getElementById('sectionScore').textContent = scoreText;
    document.getElementById('progressFill').style.width = progressPercent + '%';
}

async function endExam() {
    if (!confirm('Are you sure you want to end the exam?')) return;
    if (sessionId) {
        sessionStorage.setItem('lastSessionId', sessionId);
        sessionStorage.setItem('lastQuizType', 'section');
        setTimeout(() => showFinalResults(), 500);
    }
}

function showFinalResults(data) {
    if (!sessionId) {
        alert('No session ID found.');
        return;
    }
    window.location.href = `/analytics?session=${sessionId}&type=section`;
}

function showFailedQuiz() {
    const modal = document.getElementById('resultsModal');
    document.getElementById('finalResultTitle').textContent = '😢 Quiz Failed';
    document.getElementById('sectionsSummary').innerHTML = `
        <p style="text-align: center; color: #666; padding: 20px;">
            You didn't pass. Try again!
        </p>
    `;
    modal.classList.add('show');
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