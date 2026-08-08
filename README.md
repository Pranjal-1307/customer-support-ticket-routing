# Customer Support Ticket Routing System using NLP

A complete, production-grade automated Customer Support Ticket Routing System built using **Python, NLP, Machine Learning, Flask, SQLite, and Bootstrap**.

---

## 📌 Project Overview
Customer support departments handle thousands of incoming tickets daily across multiple channels. Manual categorization and routing of support tickets lead to high resolution times, routing errors, agent inefficiency, and poor customer satisfaction.

This system leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to automatically analyze the text of customer support tickets, classify the underlying issue, predict the urgency/priority level, and dynamically route the ticket to the appropriate department team.

---

## 🎯 Objectives
1. **Automated Categorization**: Classify tickets into 8 distinct categories (Billing, Technical Support, Refund, Shipping, Account, Complaint, Product Inquiry, Cancellation).
2. **Priority Prediction**: Determine priority levels (Low, Medium, High, Critical) based on textual signals.
3. **Smart Routing**: Instantly route tickets to target department teams.
4. **Interactive Dashboard**: Provide real-time analytics, metrics overview, and model performance comparisons for support managers.
5. **Model Retraining**: Enable one-click retraining of machine learning classifiers.

---

## 🚀 Key Features
- **Modern Web Interface**: Responsive Bootstrap 5 portal for ticket submission and real-time classification visual feedback.
- **Robust NLP Pipeline**: Lowercasing, noise & URL removal, tokenization, stopword filtering, lemmatization, and TF-IDF feature extraction.
- **Model Comparison**: Automatically evaluates Logistic Regression, Naive Bayes, Random Forest, and Linear SVM, selecting the optimal model based on weighted F1-Score.
- **Database Persistence**: SQLite storage for tracking submitted tickets and administrative logging.
- **Admin Analytics Dashboard**: Interactive Chart.js charts for category distribution, priority breakdown, and department workload.

---

## 🏗️ Architecture & Workflow
```
Customer → Submit Ticket Text → Text Preprocessing → TF-IDF Extraction
                                                          ↓
                                                ML Classifier Models
                                                          ↓
                                           ┌──────────────┴──────────────┐
                                    Category Prediction          Priority Prediction
                                           │                             │
                                           └──────────────┬──────────────┘
                                                          ↓
                                                    Routing Engine
                                                          ↓
                                                  Department/Agent
                                                          ↓
                                                 SQLite / Admin Dashboard
```

---

## 📦 Six Core Modules

### 1. User Module
- Ticket submission interface.
- Unique ticket ID generation (`TICK-XXXXXX`).
- Real-time display of predicted Category, Priority, assigned Department, and Confidence score.

### 2. NLP Processing Module (`utils/preprocess.py`)
- Lowercasing and HTML/URL stripping.
- Punctuation & special character filtering.
- Word Tokenization using NLTK.
- English stopword removal.
- WordNet Lemmatization.
- TF-IDF N-gram (1, 2) vectorization.

### 3. Classification Module (`scripts/train_models.py`)
- Predicts one of 8 categories:
  - `Billing`, `Technical Support`, `Refund`, `Shipping`, `Account`, `Complaint`, `Product Inquiry`, `Cancellation`
- Evaluates 4 candidate classifiers:
  - Logistic Regression
  - Naive Bayes (MultinomialNB)
  - Random Forest Classifier
  - Linear SVM (LinearSVC)
- Auto-selects the best classifier using weighted F1-score.

### 4. Priority Module
- Predicts 4 priority levels:
  - `Low`, `Medium`, `High`, `Critical`

### 5. Routing Module (`utils/route.py`)
Maps categories directly to target operational teams:
- **Billing** → Billing Team
- **Technical Support** → Technical Support Team
- **Refund** → Refund Team
- **Shipping** → Logistics Team
- **Account** → Account Support Team
- **Complaint** → Customer Care Team
- **Product Inquiry** → Product Support Team
- **Cancellation** → Retention Team

### 6. Admin Module (`app/app.py` & Dashboard)
- Aggregated ticket counts & real-time statistics.
- Interactive Chart.js visual charts.
- Historical submitted tickets table.
- One-click ML model retraining trigger (`/api/admin/retrain`).

---

## 📂 Project Structure
```
Customer-Support-Ticket-Routing-System/
│
├── dataset/
│   ├── tickets.csv          # Full 5,200 synthetic customer ticket dataset
│   ├── train.csv            # 80% Stratified Training split
│   └── test.csv             # 20% Stratified Testing split
│
├── notebooks/
│   ├── Data_Preprocessing.ipynb  # Text preprocessing walkthrough
│   ├── EDA.ipynb                 # Exploratory data analysis & charts
│   └── Model_Training.ipynb      # Model evaluation & confusion matrix
│
├── models/
│   ├── vectorizer.pkl           # Fitted TfidfVectorizer artifact
│   ├── label_encoder.pkl        # LabelEncoder for category labels
│   ├── ticket_classifier.pkl    # Best Category Classifier model
│   ├── priority_classifier.pkl  # Trained Priority Classifier model
│   └── model_metrics.json       # JSON benchmarks & confusion matrices
│
├── app/
│   ├── app.py                   # Flask server & SQLite database handler
│   ├── templates/
│   │   └── index.html           # Bootstrap UI (Customer Portal & Admin Dashboard)
│   └── static/
│       ├── style.css            # Custom CSS styles & badge definitions
│       └── script.js            # AJAX submit, Chart.js, & admin actions
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py            # NLP cleaning & lemmatization pipeline
│   ├── predict.py               # Inference wrapper & confidence scoring
│   └── route.py                 # Category to Department routing engine
│
├── scripts/
│   ├── generate_data.py         # Dataset generator script
│   └── train_models.py          # ML training and model comparison script
│
├── tickets.db                   # SQLite database (auto-created)
├── requirements.txt             # Python dependencies
├── README.md                    # Detailed documentation
└── main.py                      # Application entrypoint
```

---

## 🛠️ Installation & Execution

### 1. Prerequisites
- Python 3.10+ (Recommended Python 3.12)
- pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
Run the root entry point:
```bash
python main.py
```
*`main.py` will automatically check for datasets and trained model artifacts, generate data and train models if missing, set up SQLite database, and launch the web server on `http://127.0.0.1:5000`.*

---

## 🧪 Sample Predictions

| Input Ticket Text | Predicted Category | Priority | Department | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| *"My credit card was charged twice."* | **Billing** | Medium | Billing Team | 91.4% |
| *"My internet is not working."* | **Technical Support** | High | Technical Support Team | 96.2% |
| *"I need a refund for item delivered broken."* | **Refund** | Medium | Refund Team | 94.8% |
| *"Where is my package? Tracking shows no update."* | **Shipping** | Medium | Logistics Team | 97.1% |
| *"I cannot log into my account."* | **Account** | High | Account Support Team | 93.5% |
| *"I want to cancel my subscription."* | **Cancellation** | Medium | Retention Team | 95.0% |

---

## 📷 System Screenshots
*(Run web app on `http://127.0.0.1:5000` to interact with Customer Submission & Admin Analytics UI)*

---

## 💡 Limitations & Future Scope
- **Current Limitations**: Synthetic data distribution; keyword reliance for sparse short texts.
- **Future Scope**:
  - Deep Learning / Transformer integration (BERT / RoBERTa / DistilBERT).
  - Multi-language support using multilingual NLP embeddings.
  - Integration with Zendesk/Jira APIs for production ticket sync.
