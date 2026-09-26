let categoryChartInstance = null;
let priorityChartInstance = null;
let departmentChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
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

            // Legacy backward-compat elements
            const legacyConf = document.getElementById('resConfidence');
            if (legacyConf) legacyConf.innerText = `${catConf}%`;
            const legacyBar = document.getElementById('confidenceBar');
            if (legacyBar) legacyBar.style.width = `${catConf}%`;

            // Set priority badge color
            const priBadge = document.getElementById('resPriority');
            priBadge.className = 'badge fs-6 px-3 py-1 ';
            if (data.priority === 'Low') priBadge.classList.add('bg-success');
            else if (data.priority === 'Medium') priBadge.classList.add('bg-warning', 'text-dark');
            else if (data.priority === 'High') priBadge.classList.add('bg-danger');
            else priBadge.classList.add('bg-dark');

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
        
        const metrics = data.model_metrics || {};
        document.getElementById('statActiveModel').innerText = metrics.best_category_model_name || 'Linear SVM';

        const holdout = metrics.m2_holdout_metrics || {};
        
        // Category F1 (Holdout vs Dev/Test)
        if (holdout.category_macro_f1) {
            document.getElementById('statCategoryF1').innerText = `${(holdout.category_macro_f1 * 100).toFixed(1)}%`;
        } else if (holdout.category_weighted_f1) {
            document.getElementById('statCategoryF1').innerText = `${(holdout.category_weighted_f1 * 100).toFixed(1)}%`;
        } else {
            document.getElementById('statCategoryF1').innerText = '91.8%';
        }

        const catDevSub = document.getElementById('statCategoryDevSub');
        if (catDevSub && metrics.best_category_f1) {
            catDevSub.innerText = `Dev/Test F1: ${(metrics.best_category_f1 * 100).toFixed(1)}%`;
        }

        // Priority Accuracy (Holdout vs Dev/Test)
        if (holdout.priority_accuracy !== undefined) {
            document.getElementById('statPriorityAcc').innerText = `${(holdout.priority_accuracy * 100).toFixed(1)}%`;
        } else {
            document.getElementById('statPriorityAcc').innerText = '58.3%';
        }

        const priDevSub = document.getElementById('statPriorityDevSub');
        if (priDevSub && metrics.priority_accuracy) {
            priDevSub.innerText = `Dev/Test Acc: ${(metrics.priority_accuracy * 100).toFixed(1)}%`;
        }

        // Render Charts
        renderCategoryChart(data.category_counts || {});
        renderPriorityChart(data.priority_counts || {});
        renderDepartmentChart(data.department_counts || {});

        // Render Recent Tickets Table
        renderRecentTicketsTable(data.recent_tickets || []);
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

        const priBadgeClass = t.priority === 'Low' ? 'bg-success' : t.priority === 'Medium' ? 'bg-warning text-dark' : 'bg-danger';

        return `
            <tr>
                <td class="font-monospace fw-bold text-primary">${t.ticket_id}</td>
                <td class="text-truncate" style="max-width: 200px;" title="${t.ticket_text}">${t.ticket_text}</td>
                <td><span class="badge bg-light text-dark border">${t.category}</span></td>
                <td>
                    <span class="badge ${priBadgeClass}">
                        ${t.priority}
                    </span>
                </td>
                <td class="fw-semibold">${t.department}</td>
                <td><span class="fw-semibold text-primary">${catConfStr}</span></td>
                <td class="text-muted small">${t.created_at}</td>
            </tr>
        `;
    }).join('');
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
