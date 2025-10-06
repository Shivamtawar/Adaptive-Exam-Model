const API_BASE = 'https://adaptive-exam-model.onrender.com/api';
let analyticsData = null;
let charts = {};

document.addEventListener('DOMContentLoaded', () => {
    loadAnalytics();
});

async function loadAnalytics() {
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session') || sessionStorage.getItem('lastSessionId');
    const quizType = urlParams.get('type') || sessionStorage.getItem('lastQuizType') || 'adaptive';
    
    console.log('Loading analytics:', { sessionId, quizType });
    
    if (!sessionId) {
        console.error('No session ID');
        showError();
        return;
    }
    
    try {
        const endpoint = quizType === 'adaptive' 
            ? `${API_BASE}/adaptive/analytics/${sessionId}`
            : `${API_BASE}/section/analytics/${sessionId}`;
        
        console.log('Fetching:', endpoint);
        
        const response = await fetch(endpoint);
        const data = await response.json();
        
        console.log('Response:', data);
        
        if (data.success && data.analytics) {
            analyticsData = data.analytics;
            displayAnalytics();
        } else {
            console.error('Failed:', data);
            showError();
        }
    } catch (error) {
        console.error('Error:', error);
        showError();
    }
}

function displayAnalytics() {
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('analyticsContent').style.display = 'block';
    
    updateSummaryCards();
    createPerformanceChart();
    createTimeChart();
    createDifficultyChart();
    createTimelineChart();
    populateDetailedTable();
    generateInsights();
}

function updateSummaryCards() {
    document.getElementById('totalQuestions').textContent = analyticsData.total_questions || 0;
    document.getElementById('correctAnswers').textContent = analyticsData.correct_answers || 0;
    
    const accuracy = analyticsData.accuracy || 0;
    document.getElementById('accuracyRate').textContent = (accuracy * 100).toFixed(1) + '%';
    
    const totalSeconds = analyticsData.total_time_seconds || 0;
    const totalMinutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = Math.floor(totalSeconds % 60);
    document.getElementById('totalTime').textContent = `${totalMinutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    const correct = analyticsData.correct_answers || 0;
    const wrong = analyticsData.wrong_answers || 0;
    
    charts.performance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Correct', 'Wrong'],
            datasets: [{
                data: [correct, wrong],
                backgroundColor: ['#10b981', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 20, font: { size: 14 } }
                }
            }
        }
    });
}

function createTimeChart() {
    const ctx = document.getElementById('timeChart').getContext('2d');
    const fastest = analyticsData.fastest_answer || 0;
    const average = analyticsData.average_time_per_question || 0;
    const slowest = analyticsData.slowest_answer || 0;
    
    charts.time = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Fastest', 'Average', 'Slowest'],
            datasets: [{
                label: 'Time (seconds)',
                data: [fastest, average, slowest],
                backgroundColor: ['#3b82f6', '#8b5cf6', '#f59e0b'],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { callback: (value) => value + 's' }
                }
            }
        }
    });
}

function createDifficultyChart() {
    const ctx = document.getElementById('difficultyChart').getContext('2d');
    const labels = ['Very Easy', 'Easy', 'Moderate', 'Difficult'];
    let attempted = [0, 0, 0, 0];
    let correct = [0, 0, 0, 0];
    
    if (analyticsData.difficulty_breakdown) {
        Object.keys(analyticsData.difficulty_breakdown).forEach(level => {
            const idx = parseInt(level);
            const data = analyticsData.difficulty_breakdown[level];
            if (idx >= 0 && idx < 4) {
                attempted[idx] = data.attempted || 0;
                correct[idx] = data.correct || 0;
            }
        });
    } else if (analyticsData.section_breakdown) {
        analyticsData.section_breakdown.forEach((section, idx) => {
            if (idx < 4) {
                attempted[idx] = section.total || 0;
                correct[idx] = section.correct || 0;
            }
        });
    }
    
    charts.difficulty = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Attempted', data: attempted, backgroundColor: '#94a3b8' },
                { label: 'Correct', data: correct, backgroundColor: '#10b981' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { position: 'bottom', labels: { padding: 15 } } },
            scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
        }
    });
}

function createTimelineChart() {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    let labels = [];
    let correctness = [];
    let times = [];
    
    if (analyticsData.response_history && analyticsData.response_history.length > 0) {
        analyticsData.response_history.forEach((response, index) => {
            labels.push(`Q${index + 1}`);
            correctness.push(response.correct ? 1 : 0);
            times.push(response.time_spent || 0);
        });
    } else {
        labels = ['No Data'];
        correctness = [0];
        times = [0];
    }
    
    charts.timeline = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Correctness',
                    data: correctness,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                },
                {
                    label: 'Time (s)',
                    data: times,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'bottom', labels: { padding: 15, font: { size: 11 } } } },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 0,
                    max: 1,
                    ticks: { callback: (value) => value === 1 ? 'Correct' : (value === 0 ? 'Wrong' : '') }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { callback: (value) => value + 's' }
                }
            }
        }
    });
}

function populateDetailedTable() {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    
    if (!analyticsData.response_history || analyticsData.response_history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No data available</td></tr>';
        return;
    }
    
    const difficultyLabels = ['Very Easy', 'Easy', 'Moderate', 'Difficult'];
    const answerLabels = ['A', 'B', 'C', 'D'];
    
    analyticsData.response_history.forEach((response, index) => {
        const row = document.createElement('tr');
        row.className = response.correct ? 'correct-row' : 'wrong-row';
        
        const difficultyLabel = difficultyLabels[response.difficulty] || 'N/A';
        const userAnswerLabel = answerLabels[response.user_answer] || 'N/A';
        
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${response.question_id || 'N/A'}</td>
            <td><span class="difficulty-tag ${difficultyLabel.toLowerCase().replace(' ', '-')}">${difficultyLabel}</span></td>
            <td>${userAnswerLabel}</td>
            <td>${response.correct ? '✅ Correct' : '❌ Wrong'}</td>
            <td>${(response.time_spent || 0).toFixed(1)}s</td>
        `;
        tbody.appendChild(row);
    });
}

function generateInsights() {
    const insightsGrid = document.getElementById('insightsGrid');
    insightsGrid.innerHTML = '';
    const insights = [];
    
    const accuracy = analyticsData.accuracy || 0;
    if (accuracy >= 0.8) {
        insights.push({ icon: '🌟', title: 'Excellent Performance', description: `${(accuracy * 100).toFixed(1)}% accuracy!` });
    } else if (accuracy >= 0.6) {
        insights.push({ icon: '👍', title: 'Good Performance', description: `${(accuracy * 100).toFixed(1)}% correct.` });
    } else {
        insights.push({ icon: '💪', title: 'Room for Improvement', description: `${(accuracy * 100).toFixed(1)}% scored.` });
    }
    
    const avgTime = analyticsData.average_time_per_question || 0;
    if (avgTime > 0 && avgTime < 30) {
        insights.push({ icon: '⚡', title: 'Quick Thinker', description: `Average ${avgTime.toFixed(1)}s per question.` });
    } else if (avgTime >= 60) {
        insights.push({ icon: '🤔', title: 'Thoughtful Approach', description: `${avgTime.toFixed(1)}s per question.` });
    }
    
    if (analyticsData.response_history && analyticsData.response_history.length > 0) {
        const streak = calculateStreak(analyticsData.response_history);
        if (streak >= 5) {
            insights.push({ icon: '🔥', title: 'Hot Streak', description: `${streak}-question winning streak!` });
        }
    }
    
    if (insights.length === 0) {
        insights.push({ icon: '📊', title: 'Getting Started', description: 'Complete more questions for detailed insights.' });
    }
    
    insights.forEach(insight => {
        const card = document.createElement('div');
        card.className = 'insight-card';
        card.innerHTML = `
            <div class="insight-icon">${insight.icon}</div>
            <h4>${insight.title}</h4>
            <p>${insight.description}</p>
        `;
        insightsGrid.appendChild(card);
    });
}

function calculateStreak(history) {
    let maxStreak = 0;
    let currentStreak = 0;
    history.forEach(response => {
        if (response.correct) {
            currentStreak++;
            maxStreak = Math.max(maxStreak, currentStreak);
        } else {
            currentStreak = 0;
        }
    });
    return maxStreak;
}

function showError() {
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('errorState').style.display = 'flex';
}

async function loadAnalytics() {
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session') || sessionStorage.getItem('lastSessionId');
    const quizType = urlParams.get('type') || sessionStorage.getItem('lastQuizType') || 'adaptive';
    
    console.log('=== ANALYTICS DEBUG ===');
    console.log('Session ID:', sessionId);
    console.log('Quiz Type:', quizType);
    console.log('SessionStorage lastSessionId:', sessionStorage.getItem('lastSessionId'));
    console.log('SessionStorage lastQuizType:', sessionStorage.getItem('lastQuizType'));
    
    if (!sessionId) {
        console.error('❌ No session ID found');
        showError();
        return;
    }
    
    try {
        const endpoint = quizType === 'adaptive' 
            ? `${API_BASE}/adaptive/analytics/${sessionId}`
            : `${API_BASE}/section/analytics/${sessionId}`;
        
        console.log('Fetching from:', endpoint);
        
        const response = await fetch(endpoint);
        console.log('Response status:', response.status);
        console.log('Response ok:', response.ok);
        
        const data = await response.json();
        console.log('Full API Response:', JSON.stringify(data, null, 2));
        
        if (data.success && data.analytics) {
            console.log('✅ Analytics data found');
            console.log('Analytics object:', data.analytics);
            analyticsData = data.analytics;
            displayAnalytics();
        } else {
            console.error('❌ Analytics fetch failed');
            console.error('Success:', data.success);
            console.error('Error message:', data.error);
            console.error('Analytics present:', !!data.analytics);
            showError();
        }
    } catch (error) {
        console.error('❌ Exception caught:', error);
        showError();
    }
}