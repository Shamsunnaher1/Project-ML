# 🦟 DenguePredict AI

A professional Streamlit web application that predicts **Dengue Fever** risk using a
Random Forest classifier trained on clinically-informed indicators.

---

## Features

- **Haematological inputs** — Temperature, Platelet Count, WBC, Hematocrit
- **Lab result** — NS1 Antigen (Positive / Negative)
- **Symptom checklist** — Rash, Headache, Myalgia, Retro-orbital Pain
- **Fever duration** slider
- **Instant prediction** — Dengue Positive / Negative with a probability score
- **Feature importance panel** — shows the top contributing clinical factors
- **Input summary table** — exportable view of all entered values

---

## Setup Instructions

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/dengue-predictor.git
cd dengue-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## Project Structure

```
dengue-predictor/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Model Details

| Property       | Value                          |
|----------------|--------------------------------|
| Algorithm      | Random Forest Classifier       |
| Trees          | 300                            |
| Max Depth      | 8                              |
| Training Data  | 2,000 synthetic samples        |
| Preprocessing  | StandardScaler normalization   |

The model is trained on **synthetic but clinically-informed data** generated using
WHO dengue diagnostic criteria (platelet thresholds, NS1 significance, etc.).

---

## Input Features

| Feature               | Range / Type       | Clinical Significance          |
|-----------------------|--------------------|--------------------------------|
| Body Temperature      | 37.0 – 42.0 °C     | Fever > 38.5 °C is a key flag |
| Platelet Count        | 10 – 400 ×10³/µL   | < 100 suggests thrombocytopenia|
| WBC Count             | 1.0 – 12.0 ×10³/µL | Leukopenia common in dengue    |
| Hematocrit            | 25 – 65 %          | Elevated = plasma leakage      |
| NS1 Antigen           | Positive / Negative| Early dengue biomarker         |
| Skin Rash             | Yes / No           | Classic dengue symptom         |
| Severe Headache       | Yes / No           | Common symptom                 |
| Muscle / Joint Pain   | Yes / No           | "Break-bone fever" hallmark    |
| Retro-orbital Pain    | Yes / No           | Highly specific to dengue      |
| Fever Duration        | 1 – 14 days        | Dengue fever typically 5–7 days|

---

## Disclaimer

> ⚠️ **This application is for educational and screening purposes only.**
> It is NOT a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare professional.

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI framework
- [scikit-learn](https://scikit-learn.org) — Machine learning
- [pandas](https://pandas.pydata.org) — Data handling
- [NumPy](https://numpy.org) — Numerical computing
