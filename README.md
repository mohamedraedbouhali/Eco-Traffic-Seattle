<h1 align="center">Eco-Traffic Seattle </h1>
<h1 align="center">Prédiction des Niveaux de Congestion par Enrichissement de Données via Web Scraping</h1>

<h2>This project is created by </h2>
<h3>Mohamed Raed Bouhali & Ilef Ben Hassen </h3>
<h4>1. Présentation du Projet</h4>
<h4>Titre du Projet : SmartTraffic Seattle : Système de Prédiction de Congestion Urbaine par Enrichissement Multisources.</h4>

<h4>Contexte : Dans le cadre du module "Python for Data Science 2", ce projet vise à transformer des données statiques de comptage de véhicules en un outil prédictif dynamique.</h4>

<h4>Objectif Principal : Prédire le niveau de trafic (Fluide, Modéré, Critique) sur les axes routiers de Seattle en combinant des données historiques et des données contextuelles scrapées (Météo/News).</h4>
<4>## 2. Spécifications Fonctionnelles (Le "Quoi")</4>
<4>Le système est conçu pour répondre aux besoins suivants :</4>
<4></4>
<4>* [Data_Ingestion] : Collecte automatisée des données météo 2022 et extraction d'incidents via Web Scraping.</4>
<4>* [Predictive_Core] : Classification du niveau de trafic basée sur les caractéristiques géospatiales et temporelles.</4>
<4>* [User_Interface] : Visualisation interactive sur un Dashboard React pour consulter l'état futur du trafic sur une carte.</4>
<4>* [Service_Access] : Exposition des prédictions via une API REST FastAPI pour une intégration tierce.</4>
<4></section></4>
<4></4>
<4>---</4>
<4></4>
<4><section id="technical-stack"></4>
<4>## 3. Spécifications Techniques (Le "Comment")</4>
<4></4>
<4>### 🛠 A. Data Pipeline & ML (Phase 1 & 2)</4>
<4></4>
<4>* Sources : Fichier trafficFlow.csv (SDOT) + Scraping (BeautifulSoup/Selenium) pour la météo et les news.</4>
<4>* Prétraitement : Nettoyage, Feature Engineering (saisonnalité, heures de pointe, jours fériés).</4>
<4>* Équilibrage : Application de l'algorithme SMOTE pour gérer les classes de "Congestion Critique" minoritaires.</4>
<4>* Modélisation : Comparaison de modèles (Random Forest vs XGBoost) avec optimisation via GridSearchCV.</4>
<4>* Gouvernance : Suivi des métriques et versioning des modèles via MLflow.</4>
<4></4>
<4>### 🌐 B. Architecture logicielle & Déploiement (Phase 3)</4>
<4></4>
<4>* Backend (API) : Framework FastAPI avec endpoints de prédiction unitaire et batch.</4>
<4>* Frontend (Interface) : Framework React (Vite) avec intégration de cartes dynamiques (Leaflet).</4>
<4>* DevOps (Déploiement) : Conteneurisation avec Docker et orchestration via Docker-Compose.</4>
<4></section></4>
<4></4>
<4><footer></4>
<4>### 📌 Livrables Attendus</4>
<4>1. Dépôt GitHub avec code source documenté.</4>
<4>2. Environnement virtualisé via Docker.</4>
<4>3. Dashboard interactif fonctionnel.</4>
