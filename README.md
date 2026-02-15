<div align="center">

# ⚖️ A.C.E. (Advanced Criminal Evaluation)
### *Predictive Risk Engine & Intelligence Dashboard for Urban Safety*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-black?style=for-the-badge&logo=xgboost)](https://xgboost.ai)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)

**A.C.E.** transforms reactive data into proactive urban safety insights using advanced Machine Learning and Business Intelligence.
</div>

---

## 📖 1. Project Vision
Developed as a high-performance evaluation framework, **A.C.E.** bridges the gap between raw historical crime records and real-time decision-making. By synthesizing crime categories with socio-economic indicators and temporal patterns, the system generates a **Predictive Risk Score** ($0.0$ to $1.0$) to help city planners and law enforcement optimize resource allocation.

---

## 🛠️ 2. The Tech Ecosystem
<div align="center">

| **Data Engineering** | **Machine Learning** | **BI & Deployment** |
| :---: | :---: | :---: |
| <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas" /> | <img src="https://img.shields.io/badge/XGBoost-black?style=flat-square&logo=XGBoost" /> | <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit" /> |
| <img src="https://img.shields.io/badge/Faker-DataGen-grey?style=flat-square" /> | <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn" /> | <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly" /> |
| <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy" /> | <img src="https://img.shields.io/badge/SHAP-Explainability-blue?style=flat-square" /> | <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=Docker" /> |

</div>

---

## 🚀 3. Core Architecture

### 📂 Phase 1: Data Enrichment & Engineering
A.C.E. creates a multidimensional feature set:
* **Temporal Signals:** Extraction of cyclical trends (Hour, Day of Week, Month).
* **Social Context:** Mapping employment status and social background to crime categories.
* **Target Synthesis:** Generation of severity indices based on crime type.

### 🧠 Phase 2: Predictive Risk Engine
The AI backend utilizes gradient-boosted decision trees to evaluate risk:
* **SMOTE Balancing:** Handles the scarcity of high-severity crime incidents in training data.
* **SHAP Interpretability:** Provides "Local Explanations" to show which factors (e.g., age, social situation, or time) most influenced the risk score.

### 💻 Phase 3: The Intelligence Dashboard
A high-performance BI interface built with **Streamlit** and **Folium**:
* **Risk Heatmaps:** Dynamic geospatial visualization of predicted incident zones.
* **Situation Gauges:** Real-time tracking of average crime severity across the dataset.

---

## 📋 4. Functional Matrix

> [!IMPORTANT]
> A.C.E. is designed for low-latency inference, enabling the evaluation of thousands of records per second for city-wide reporting.

| Feature | Category | Tech Used |
| :--- | :---: | :--- |
| **Synthetic Ingestion** | Data | Python Faker, Pandas |
| **Predictive Modeling** | AI | XGBoost, Scikit-Learn |
| **Explainable AI** | ML Ops | SHAP |
| **Interactive BI** | Frontend | Streamlit, Plotly |
| **Containerization** | DevOps | Docker |

---

## 👩‍💻 5. Work Architecture


The project follows a modular pipeline:
1.  **Ingestion:** Cleaning and processing synthetic CSV data.
2.  **Logic:** AI models assign probability and severity scores.
3.  **Visuals:** The dashboard queries the logic layer to render actionable charts.

---

👤 Project Leads

<div align="center">
  <table style="border: none; border-collapse: collapse;">
    <tr>
      <td align="center" width="50%" style="border: none;">
        <img src="raed.jpg" width="180" height="180" style="border-radius:50%; object-fit: cover; border: 4px solid #2ecc71; box-shadow: 0px 4px 10px rgba(0,0,0,0.3);" alt="Mohamed Raed Bouhali"/>
        <br />
        <h3 style="margin-top: 10px;">Mohamed Raed Bouhali</h3>
        <p><b>Data Engineer & Backend Architect</b></p>
        <a href="https://github.com/mohamedraedbouhali"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" height="22"></a>
        <a href="https://www.linkedin.com/in/bouhali-mohamed-raed/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" height="22"></a>
      </td>
      <td align="center" width="50%" style="border: none;">
        <img src="ilef.jpg" width="180" height="180" style="border-radius:50%; object-fit: cover; border: 4px solid #3498db; box-shadow: 0px 4px 10px rgba(0,0,0,0.3);" alt="Ilef Ben Hassen"/>
        <br />
        <h3 style="margin-top: 10px;">Ilef Ben Hassen</h3>
        <p><b>ML Specialist & Frontend Lead</b></p>
        <a href="https://github.com/BenHassenIlef"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" height="22"></a>
        <a href="https://www.linkedin.com/in/ben-hassen-ilef-924859304/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" height="22"></a>
      </td>
    </tr>
  </table>
</div>

<br />
