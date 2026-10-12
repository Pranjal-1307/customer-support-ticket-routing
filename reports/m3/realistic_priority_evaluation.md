# Phase 7: Realistic Priority Evaluation Report

## 1. Objective
Assess the real-world operational reliability of Priority prediction on curated representative support tickets spanning Low, Medium, High, and Critical priorities with explicit ground-truth domain justifications.

> **Evaluation Rule:** These 12 scenarios are strictly reserved for testing and verification; none were added or relabeled in training sets.

---

## 2. Comparative Benchmark Table

| Ticket Text | Expected Ground Truth | Domain Reasoning | M2 Baseline Prediction | M3 Candidate Prediction | Extracted Key Signals | Explanation | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **"Where is my order?"** | **Low** | Routine status inquiry; no defect or financial risk. | Medium (89.8%) | Medium (90.4%) | Urg: 0, Fin: 0, Sec: 0 | Routine inquiry | Baseline Fail |
| **"Can you tell me my order status?"** | **Low** | Routine product check. | Medium (39.1%) | **Low** (63.2%) | Urg: 0, Fin: 0, Sec: 0 | Routine inquiry | **M3 Fixed** |
| **"Where can I find tracking information?"** | **Low** | Navigational guidance. | **Low** (90.4%) | **Low** (92.6%) | Urg: 0, Fin: 0, Sec: 0 | Routine inquiry | Both Match |
| **"My package is delayed by two days."** | **Medium** | Minor delivery delay. | High (60.8%) | **Medium** (73.6%) | Delay: 0.65 | Routine delay | **M3 Fixed** |
| **"I have been waiting for my refund."** | **Medium** | Standard refund inquiry. | **Medium** (90.6%) | **Medium** (95.7%) | Fin: 0.75 | Refund inquiry | Both Match |
| **"I cannot update my billing address."** | **Medium** | Non-blocking profile change. | **Medium** (82.2%) | **Medium** (71.2%) | Urg: 0, Sec: 0 | Routine inquiry | Both Match |
| **"I was charged twice for the same order."** | **High** | Direct financial impact / double charge. | Medium (78.4%) | Medium (74.5%) | Fin: 0.95 | Financial impact | Training bias |
| **"I cannot access my account and I need it today."** | **High** | Account access with strict same-day deadline. | **High** (61.1%) | **High** (86.0%) | Time: 0.90 | Time sensitivity | **M3 Improved** |
| **"My payment failed during an important purchase."** | **High** | Transaction failure on key purchase. | **High** (99.0%) | **High** (98.3%) | Fin: 0.75 | Billing issue | Both Match |
| **"My account has been hacked."** | **Critical** | Severe security breach. | High (61.7%) | High (55.0%) | Sec: 1.00 | Security risk | Training bias |
| **"There are unauthorized transactions on my account."** | **Critical** | Financial fraud / compromised account. | High (72.6%) | Medium (63.4%) | Fin: 0.95, Sec: 0.85 | Financial / Sec | Training bias |
| **"Our entire business service is unavailable."** | **Critical** | Complete enterprise outage / downtime. | Medium (50.2%) | **Critical** (99.0%) | Svc: 1.00 | Service disruption | **M3 Fixed** |

---

## 3. Accuracy Summary

| Model | Realistic Scenario Accuracy | Key Qualitative Improvements |
| :--- | :--- | :--- |
| **M2 Production Baseline** | **41.7%** (5 / 12) | Failed on major outages (assigned Medium), routine status (assigned High), minor delays (assigned High). |
| **M3 Candidate (TF-IDF + Signals)** | **66.7%** (8 / 12) | Successfully resolved enterprise outages to Critical (99.0%), routine queries to Low (63.2%), and minor delays to Medium (73.6%). |

---

## 4. Analysis & Next Steps
While M3 Candidate significantly improves realistic accuracy from **41.7% → 66.7%**, synthetic training data priors still pull specific double-charge and hacking cases toward High/Medium. This demonstrates why **Phase 9–12 Human Feedback and Controlled Retraining** is required for production operation.
