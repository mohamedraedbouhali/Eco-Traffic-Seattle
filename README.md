<div align="center">

# 🌿 Eco-Traffic Seattle
### *Smart City Traffic Prediction via Multi-Source Data Enrichment*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
  [![](https://img.shields.io/badge/Status-In--Development-yellow?style=for-the-badge)]()
  [![](https://img.shields.io/badge/Location-Seattle-00629B?style=for-the-badge&logo=seattle)]()
  [![](https://img.shields.io/badge/Framework-FastAPI%20%2B%20React-green?style=for-the-badge)]()

</div>

---

## 📖 1. Project Vision
Developed for the **"Python for Data Science 2"** module, this project bridges the gap between historical data and real-time context. By scraping weather patterns and city news, we've built a classifier that understands *why* traffic happens, not just *when*.



---

## 🛠️ 2. The Tech Ecosystem
<div align="center">

| **Ingestion & Scraping** | **Machine Learning** | **DevOps & UI** |
| :---: | :---: | :---: |
| <img src="https://img.shields.io/badge/Selenium-43B02A?style=flat-square&logo=Selenium&logoColor=white" /> | <img src="https://img.shields.io/badge/XGBoost-black?style=flat-square&logo=XGBoost" /> | <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=Docker" /> |
| <img src="https://img.shields.io/badge/BeautifulSoup-grey?style=flat-square" /> | <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn" /> | <img src="https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi" /> |
| <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas" /> | <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow" /> | <img src="https://img.shields.io/badge/React--Vite-646CFF?style=flat-square&logo=vite" /> |

</div>

---

## 🚀 3. Core Architecture

### 📂 Phase 1: Data Enrichment
We don't just rely on `trafficFlow.csv`. Our engine crawls:
* **Weather Logs:** Historical and forecast data to identify rain/snow impact.
* **News Scraping:** Real-time event extraction (Sports, Accidents, Protests).

### 🧠 Phase 2: Predictive Engine

* **SMOTE Sampling:** We solved the "Minority Class" problem where critical congestion is rare but vital to predict.
* **Model Duel:** Systematic comparison between **Random Forest** and **XGBoost** using `GridSearchCV`.

### 💻 Phase 3: The Dashboard
A high-performance **React** application using **Leaflet.js** to render Seattle's coordinates. 
* **Green:** Fluid Traffic 🟢
* **Orange:** Moderate Congestion 🟠
* **Red:** Critical Traffic 🔴

---

## 📋 4. Functional Matrix

> [!IMPORTANT]  
> This system is designed for 1-second inference latency, allowing for batch processing of entire city districts.

| Feature | Category | Tech Used |
| :--- | :---: | :--- |
| **Real-time Scraping** | Ingestion | Selenium, BeautifulSoup |
| **Feature Engineering** | Data | Pandas, Seasonality Masks |
| **REST API** | Backend | FastAPI, Pydantic |
| **Containerization** | Infrastructure | Docker, Docker-Compose |

## 👩‍💻 5. Work Architecture
<img src="Architecture.png" /> 
## 👤 Project Leads

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


---
---

---

