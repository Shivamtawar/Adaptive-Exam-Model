// Section-Based Quiz JavaScript
const API_BASE = 'https://adaptive-exam-model.onrender.com/api';

let sessionId = null;
let currentQuestion = null;
let selectedAnswer = null;
let currentSection = 0;
let questionIndex = 0;
let sectionScore = 0;

// Initialize quiz
document.addEventListener('DOMContentLoaded', () => {
    startQuiz();
});

// Start quiz
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
            
            updateSectionUI(data.section, data.section_name);
            displayQuestion(currentQuestion);
        } else {
            alert('Error starting quiz: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to start quiz. Make sure the Flask server is running.');
    }
}

// Update section UI
function updateSectionUI(sectionNum, sectionName) {
    document.getElementById('sectionTitle').textContent = `Section ${sectionNum + 1}: ${sectionName}`;
    
    // Update section steps
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

// Display question
function displayQuestion(question) {
    questionIndex++;
    
    document.getElementById('questionNumber').textContent = `Question #${questionIndex}`;
    document.getElementById('questionText').textContent = question.question;
    
    // Display options
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
    
    // Reset selection
    selectedAnswer = null;
    document.getElementById('submitBtn').disabled = true;
    
    // Update progress
    updateProgress();
}

// Select option
function selectOption(letter, element) {
    document.querySelectorAll('.option-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    element.classList.add('selected');
    selectedAnswer = letter;
    document.getElementById('submitBtn').disabled = false;
}

// Submit answer
async function submitAnswer() {
    if (!selectedAnswer) return;
    
    document.getElementById('submitBtn').disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/section/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                answer: selectedAnswer
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update score
            if (data.is_correct) {
                sectionScore++;
            }
            
            updateProgress();
            
            // Show feedback
            showFeedback(data.is_correct);
            
            // Store response data
            window.currentResponseData = data;
        } else {
            alert('Error submitting answer: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to submit answer.');
    }
}

// Show feedback
function showFeedback(isCorrect) {
    const modal = document.getElementById('feedbackModal');
    const icon = document.getElementById('feedbackIcon');
    const title = document.getElementById('feedbackTitle');
    const message = document.getElementById('feedbackMessage');
    
    icon.className = 'feedback-icon ' + (isCorrect ? 'correct' : 'wrong');
    title.textContent = isCorrect ? 'Correct! ✅' : 'Incorrect ❌';
    message.textContent = isCorrect 
        ? 'Great job!' 
        : 'Keep trying!';
    
    modal.classList.add('show');
}

// Handle next after feedback
function handleNext() {
    const modal = document.getElementById('feedbackModal');
    modal.classList.remove('show');
    
    const data = window.currentResponseData;
    
    if (data.needs_11th_question) {
        // Show 11th question
        currentQuestion = data.question_11;
        displayQuestion(currentQuestion);
    } else if (data.section_complete) {
        // Show section result
        showSectionResult(data);
    } else if (data.next_question) {
        // Next question in section
        currentQuestion = data.next_question;
        displayQuestion(currentQuestion);
    }
}

// Show section result
function showSectionResult(data) {
    const modal = document.getElementById('sectionCompleteModal');
    const icon = document.getElementById('sectionIcon');
    const title = document.getElementById('sectionResultTitle');
    const message = document.getElementById('sectionResultMessage');
    
    icon.className = 'feedback-icon ' + (data.section_passed ? 'pass' : 'fail');
    title.textContent = data.section_passed ? 'Section Passed! 🎉' : 'Section Failed 😢';
    message.textContent = data.section_passed 
        ? 'Congratulations! You can proceed to the next section.' 
        : 'You need 6/10 to pass. Try again!';
    
    document.getElementById('sectionCorrect').textContent = data.progress.correct_in_section;
    document.getElementById('sectionTotal').textContent = data.progress.questions_answered;
    
    // Store data for next action
    window.sectionCompleteData = data;
    
    modal.classList.add('show');
}

// Handle section complete
function handleSectionComplete() {
    const modal = document.getElementById('sectionCompleteModal');
    modal.classList.remove('show');
    
    const data = window.sectionCompleteData;
    
    if (data.quiz_complete) {
        showFinalResults(data);
    } else if (data.quiz_failed) {
        showFailedQuiz();
    } else if (data.next_section !== undefined) {
        // Move to next section
        currentSection = data.next_section;
        currentQuestion = data.next_question;
        questionIndex = 0;
        sectionScore = 0;
        
        updateSectionUI(currentSection, data.section_name);
        displayQuestion(currentQuestion);
    }
}

// Update progress
function updateProgress() {
    const progressText = `${questionIndex}/10`;
    const scoreText = `${sectionScore}/10`;
    const progressPercent = (questionIndex / 10) * 100;
    
    document.getElementById('questionProgress').textContent = progressText;
    document.getElementById('sectionScore').textContent = scoreText;
    document.getElementById('progressFill').style.width = progressPercent + '%';
}

// Show final results
function showFinalResults(data) {
    const modal = document.getElementById('resultsModal');
    const summary = document.getElementById('sectionsSummary');
    
    summary.innerHTML = '';
    
    data.progress.completed_sections.forEach((section, index) => {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'section-result';
        
        const sectionNames = ['Very Easy', 'Easy', 'Moderate', 'Difficult'];
        const passed = section.passed ? '✅' : '❌';
        
        sectionDiv.innerHTML = `
            <div>
                <strong>Section ${index + 1}: ${sectionNames[section.section]}</strong>
            </div>
            <div>
                ${section.correct}/${section.total} ${passed}
            </div>
        `;
        
        summary.appendChild(sectionDiv);
    });
    
    document.getElementById('finalResultTitle').textContent = '🎉 Congratulations! Quiz Complete!';
    modal.classList.add('show');
}

// Show failed quiz
function showFailedQuiz() {
    const modal = document.getElementById('resultsModal');
    document.getElementById('finalResultTitle').textContent = '😢 Quiz Failed';
    document.getElementById('sectionsSummary').innerHTML = `
        <p style="text-align: center; color: #666; padding: 20px;">
            You didn't pass this section. Don't worry, practice makes perfect!<br>
            Try again to improve your score.
        </p>
    `;
    modal.classList.add('show');
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