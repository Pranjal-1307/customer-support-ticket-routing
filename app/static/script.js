let categoryChartInstance = null;
let priorityChartInstance = null;
let departmentChartInstance = null;
let feedbackModalInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Modal
    const modalEl = document.getElementById('feedbackModal');
    if (modalEl && typeof bootstrap !== 'undefined') {
        feedbackModalInstance = new bootstrap.Modal(modalEl);
    }

    // Quick Sample Prompts listener
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const promptText = e.target.getAttribute('data-sample');
            document.getElementById('ticketText').value = promptText;
        });
    });

    // Form Submit listener
    const ticketForm = document.getElementById('ticketForm');
    if (ticketForm) {
        ticketForm.addEventListener('submit', handleTicketSubmit);
    }

    // Feedback Form listener
    const feedbackForm = document.getElementById('feedbackForm');
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', handleFeedbackSubmit);
    }
});

async function handleTicketSubmit(event) {
    event.preventDefault();
    const ticketText = document.getElementById('ticketText').value.trim();
    const submitBtn = document.getElementById('submitBtn');
    const resultCard = document.getElementById('resultCard');

    if (!ticketText) return;

    // Show loading UI
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing ticket with NLP...`;

    try {
        const response = await fetch('/api/tickets/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticket_text: ticketText })
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            document.getElementById('resTicketId').innerText = data.ticket_id;
            document.getElementById('resCategory').innerText = data.category;
            document.getElementById('resPriority').innerText = data.priority;
            document.getElementById('resDepartment').innerText = data.department;
            document.getElementById('ticketStatus').innerText = data.ticket_status || 'Open';

            const catConf = data.category_confidence !== undefined ? data.category_confidence : data.confidence;
            const priConf = data.priority_confidence !== undefined ? data.priority_confidence : 0;

            const catConfEl = document.getElementById('resCategoryConfidence');
            if (catConfEl) catConfEl.innerText = `${catConf}%`;

            const catBar = document.getElementById('catConfidenceBar');
            if (catBar) catBar.style.width = `${catConf}%`;

            const priConfEl = document.getElementById('resPriorityConfidence');
            if (priConfEl) priConfEl.innerText = `${priConf}%`;

            const priBar = document.getElementById('priConfidenceBar');
            if (priBar) {
                priBar.style.width = `${priConf}%`;
                priBar.className = 'progress-bar progress-bar-striped progress-bar-animated ';
                if (data.priority === 'Low') priBar.classList.add('bg-success');
                else if (data.priority === 'Medium') priBar.classList.add('bg-warning');
                else if (data.priority === 'High') priBar.classList.add('bg-danger');
                else priBar.classList.add('bg-dark');
            }

            // Set priority badge color
            const priBadge = document.getElementById('resPriority');
            priBadge.className = 'badge fs-6 px-3 py-1 ';
            if (data.priority === 'Low') priBadge.classList.add('bg-success');
            else if (data.priority === 'Medium') priBadge.classList.add('bg-warning', 'text-dark');
            else if (data.priority === 'High') priBadge.classList.add('bg-danger');
            else priBadge.classList.add('bg-dark');

            // Render Priority Intelligence Signals (M3)
            const explanationEl = document.getElementById('resExplanationText');
            if (explanationEl && data.priority_explanation) {
                explanationEl.innerText = data.priority_explanation;
            }

            const signalsListEl = document.getElementById('resSignalsList');
            if (signalsListEl && data.priority_signals) {
                const sigs = data.priority_signals;
                const sigKeys = [
                    { key: 'urgency', label: 'Urgency', icon: 'fa-bolt' },
                    { key: 'severity', label: 'Severity', icon: 'fa-triangle-exclamation' },
                    { key: 'financial_impact', label: 'Financial Impact', icon: 'fa-dollar-sign' },
                    { key: 'security_risk', label: 'Security Risk', icon: 'fa-shield-halved' },
                    { key: 'service_impact', label: 'Service Impact', icon: 'fa-server' },
                    { key: 'time_sensitivity', label: 'Time Sensitivity', icon: 'fa-clock' }
                ];

                signalsListEl.innerHTML = sigKeys.map(s => {
                    const score = sigs[s.key] !== undefined ? sigs[s.key] : 0.0;
                    const pct = Math.round(score * 100);
                    const colorClass = score >= 0.7 ? 'text-danger' : score >= 0.4 ? 'text-warning' : 'text-secondary';
                    const barColor = score >= 0.7 ? 'bg-danger' : score >= 0.4 ? 'bg-warning' : 'bg-primary';

                    return `
                        <div class="col-md-4 col-6">
                            <div class="p-2 bg-white rounded border small">
                                <div class="d-flex justify-content-between align-items-center mb-1">
                                    <span class="text-truncate fw-semibold"><i class="fa-solid ${s.icon} ${colorClass} me-1"></i>${s.label}</span>
                                    <span class="fw-bold ${colorClass}">${pct}%</span>
                                </div>
                                <div class="progress" style="height: 4px;">
                                    <div class="progress-bar ${barColor}" style="width: ${pct}%"></div>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');
            }

            // Legacy backward-compat elements
            const legacyConf = document.getElementById('resConfidence');
            if (legacyConf) legacyConf.innerText = `${catConf}%`;
            const legacyBar = document.getElementById('confidenceBar');
            if (legacyBar) legacyBar.style.width = `${catConf}%`;

            resultCard.classList.remove('d-none');
            resultCard.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert(`Error: ${data.message || 'Failed to submit ticket.'}`);
        }
    } catch (err) {
        alert(`Connection error: ${err.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i class="fa-solid fa-robot me-2"></i>Analyze & Submit Ticket`;
    }
}

async function loadAdminStats() {
    try {
        const response = await fetch('/api/admin/stats');
        const data = await response.json();

        // Stats Overview
        document.getElementById('statTotalTickets').innerText = data.total_tickets || 0;
        
        const fbStats = data.feedback_stats || {};
        const valCount = fbStats.validated_feedback_count || 0;
        const pendCount = fbStats.pending_validation_count || 0;
        
        const valEl = document.getElementById('statValidatedFeedback');
        if (valEl) valEl.innerText = valCount;

        const pendEl = document.getElementById('statPendingFeedback');
        if (pendEl) pendEl.innerText = `Pending: ${pendCount}`;

        // Render Charts
        renderCategoryChart(data.category_counts || {});
        renderPriorityChart(data.priority_counts || {});
        renderDepartmentChart(data.department_counts || {});

        // Render Recent Tickets Table
        renderRecentTicketsTable(data.recent_tickets || []);

        // Render Feedback Table
        renderFeedbackTable(fbStats.recent_feedback || []);
    } catch (err) {
        console.error("Error loading admin stats:", err);
    }
}

function renderCategoryChart(catData) {
    const ctx = document.getElementById('categoryChart').getContext('2d');
    if (categoryChartInstance) categoryChartInstance.destroy();

    const labels = Object.keys(catData);
    const counts = Object.values(catData);

    categoryChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Tickets',
                data: counts,
                backgroundColor: 'rgba(37, 99, 235, 0.7)',
                borderColor: '#2563eb',
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

function renderPriorityChart(priData) {
    const ctx = document.getElementById('priorityChart').getContext('2d');
    if (priorityChartInstance) priorityChartInstance.destroy();

    const labels = ['Low', 'Medium', 'High', 'Critical'];
    const counts = labels.map(p => priData[p] || 0);

    priorityChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#881337']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

function renderDepartmentChart(deptData) {
    const ctx = document.getElementById('departmentChart').getContext('2d');
    if (departmentChartInstance) departmentChartInstance.destroy();

    const labels = Object.keys(deptData);
    const counts = Object.values(deptData);

    departmentChartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: [
                    '#3b82f6', '#10b981', '#8b5cf6', '#f59e0b',
                    '#ec4899', '#6366f1', '#14b8a6', '#f97316'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

function renderRecentTicketsTable(tickets) {
    const tbody = document.getElementById('recentTicketsBody');
    if (!tickets || tickets.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No tickets submitted yet.</td></tr>`;
        return;
    }

    tbody.innerHTML = tickets.map(t => {
        const catConf = t.category_confidence !== undefined && t.category_confidence !== null ? t.category_confidence : t.confidence;
        const catConfStr = catConf !== undefined && catConf !== null ? `${catConf}%` : '--';

        const priBadgeClass = t.priority === 'Low' ? 'bg-success' : t.priority === 'Medium' ? 'bg-warning text-dark' : t.priority === 'High' ? 'bg-danger' : 'bg-dark';

        const escapedText = (t.ticket_text || '').replace(/"/g, '&quot;');

        return `
            <tr>
                <td class="font-monospace fw-bold text-primary">${t.ticket_id}</td>
                <td class="text-truncate" style="max-width: 200px;" title="${escapedText}">${t.ticket_text}</td>
                <td><span class="badge bg-light text-dark border">${t.category}</span></td>
                <td><span class="badge ${priBadgeClass}">${t.priority}</span></td>
                <td class="fw-semibold">${t.department}</td>
                <td><span class="fw-semibold text-primary">${catConfStr}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline-primary rounded-pill px-3" onclick="openFeedbackModal('${t.ticket_id}', '${escapedText}', '${t.priority}')">
                        <i class="fa-solid fa-pen-to-square me-1"></i>Correct
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function renderFeedbackTable(feedbackList) {
    const tbody = document.getElementById('feedbackTableBody');
    if (!tbody) return;

    if (!feedbackList || feedbackList.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">No feedback recorded yet.</td></tr>`;
        return;
    }

    tbody.innerHTML = feedbackList.map(fb => {
        const isValidated = fb.validated === 1;
        const statusBadge = isValidated 
            ? `<span class="badge bg-success"><i class="fa-solid fa-check me-1"></i>Validated</span>`
            : `<span class="badge bg-warning text-dark"><i class="fa-solid fa-clock me-1"></i>Pending Review</span>`;

        const actionBtn = isValidated
            ? `<span class="text-muted small">Ready for Gate</span>`
            : `<button class="btn btn-sm btn-success rounded-pill px-2" onclick="validateFeedback('${fb.feedback_id}')"><i class="fa-solid fa-check me-1"></i>Validate</button>`;

        return `
            <tr>
                <td class="font-monospace small">${fb.feedback_id}</td>
                <td class="font-monospace fw-bold text-primary">${fb.ticket_id}</td>
                <td><span class="badge bg-secondary">${fb.original_priority}</span></td>
                <td><span class="badge bg-primary">${fb.corrected_priority}</span></td>
                <td class="small text-muted">${fb.reason || '--'}</td>
                <td class="small">${fb.reviewer || 'agent'}</td>
                <td>${statusBadge}</td>
                <td>${actionBtn}</td>
            </tr>
        `;
    }).join('');
}

function openFeedbackModal(ticketId, ticketText, originalPriority) {
    document.getElementById('modalTicketId').value = ticketId;
    document.getElementById('modalDisplayTicketId').innerText = ticketId;
    document.getElementById('modalTicketText').innerText = ticketText;
    document.getElementById('modalOriginalPriority').innerText = originalPriority;
    document.getElementById('modalCorrectedPriority').value = originalPriority === 'High' ? 'Medium' : 'High';
    document.getElementById('modalReason').value = '';

    if (feedbackModalInstance) {
        feedbackModalInstance.show();
    } else {
        const modalEl = document.getElementById('feedbackModal');
        feedbackModalInstance = new bootstrap.Modal(modalEl);
        feedbackModalInstance.show();
    }
}

async function handleFeedbackSubmit(e) {
    e.preventDefault();
    const ticketId = document.getElementById('modalTicketId').value;
    const correctedPriority = document.getElementById('modalCorrectedPriority').value;
    const reason = document.getElementById('modalReason').value.trim();
    const reviewer = document.getElementById('modalReviewer').value.trim() || 'admin_supervisor';
    const submitBtn = document.getElementById('modalSubmitBtn');

    submitBtn.disabled = true;
    submitBtn.innerText = 'Saving...';

    try {
        const res = await fetch('/api/feedback/priority', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticket_id: ticketId,
                corrected_priority: correctedPriority,
                reason: reason,
                reviewer: reviewer,
                validated: 0
            })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            if (feedbackModalInstance) feedbackModalInstance.hide();
            alert(`Priority correction saved! Feedback ID: ${data.feedback_id}`);
            loadAdminStats();
        } else {
            alert(`Error: ${data.message || 'Failed to submit feedback.'}`);
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Save Feedback';
    }
}

async function validateFeedback(feedbackId) {
    try {
        const res = await fetch('/api/feedback/priority/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feedback_id: feedbackId, validated: 1 })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            loadAdminStats();
        } else {
            alert(`Failed: ${data.message}`);
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

async function retrainModels() {
    const retrainBtn = document.getElementById('retrainBtn');
    retrainBtn.disabled = true;
    retrainBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>Retraining Models...`;

    try {
        const response = await fetch('/api/admin/retrain', { method: 'POST' });
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            alert(`Retraining successful! Best Model: ${data.metrics.best_category_model_name} (F1: ${(data.metrics.best_category_f1 * 100).toFixed(1)}%)`);
            loadAdminStats();
        } else {
            alert(`Retraining failed: ${data.message}`);
        }
    } catch (err) {
        alert(`Error retraining models: ${err.message}`);
    } finally {
        retrainBtn.disabled = false;
        retrainBtn.innerHTML = `<i class="fa-solid fa-arrows-rotate me-2"></i>Retrain ML Models`;
    }
}
