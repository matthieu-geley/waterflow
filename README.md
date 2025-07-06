# Waterflow : Projet MLOps pour la Prédiction de la Qualité de l'Eau

## Contexte du Projet

Ce projet implémente une solution **MLOps complète** pour prédire la potabilité de l'eau à partir de mesures physico-chimiques. Il suit une approche structurée en **6 étapes** conformément au sujet d'évaluation, intégrant les meilleures pratiques du Machine Learning Operations.

### Objectif Principal

Développer un système de **classification binaire** permettant de déterminer si un échantillon d'eau est **potable** ou **non potable** à partir de 9 mesures de qualité, en utilisant une architecture MLOps robuste avec **MLflow** pour le tracking et **Flask** pour le déploiement.

## Données et Analyse

### Dataset Utilisé

- **Source** : Dataset de qualité de l'eau (3276 échantillons)
- **Variables prédictives** : 9 mesures physico-chimiques
- **Variable cible** : Potabilité binaire (0 = non potable, 1 = potable)

### Variables du Dataset

| Variable                  | Description                     | Unité        | Plage Optimale |
| ------------------------- | ------------------------------- | ------------- | -------------- |
| **pH**              | Niveau d'acidité/basicité     | Échelle 0-14 | 6.5-8.5        |
| **Hardness**        | Dureté de l'eau                | mg/L          | -              |
| **Solids**          | Total des solides dissous (TDS) | ppm           | <1000          |
| **Chloramines**     | Concentration en chloramines    | ppm           | <4             |
| **Sulfate**         | Concentration en sulfates       | mg/L          | <250           |
| **Conductivity**    | Conductivité électrique       | μS/cm        | -              |
| **Organic_carbon**  | Carbone organique total         | ppm           | <2             |
| **Trihalomethanes** | Trihalométhanes                | μg/L         | <80            |
| **Turbidity**       | Turbidité                      | NTU           | <4             |

### Analyse Exploratoire Réalisée

1. **Distribution des données** : Analyse de la répartition des classes (déséquilibre observé)
2. **Corrélations** : Étude des relations entre variables physico-chimiques
3. **Valeurs manquantes** : Identification et traitement par imputation médiane
4. **Outliers** : Détection via boxplots pour chaque variable
5. **Normalisation** : StandardScaler appliqué pour homogénéiser les échelles

## Algorithmes et Modélisation

### Approche MLOps Adoptée

Le projet utilise **MLflow** pour orchestrer le cycle de vie complet des modèles selon les principes MLOps :

- **Tracking** : Suivi automatique des expériences et métriques
- **Versioning** : Gestion des versions de modèles et datasets
- **Reproductibilité** : Environnements et paramètres versionnés
- **Comparaison** : Évaluation objective des performances
- **Déploiement** : Pipeline automatisé vers la production

### Modèles Implémentés

1. **Random Forest** (Modèle principal)

   - Ensemble de arbres de décision
   - Robuste aux outliers et overfitting
   - Interprétabilité via feature importance
2. **XGBoost** (Modèle alternatif)

   - Gradient boosting optimisé
   - Performance élevée sur données tabulaires
   - Régularisation avancée
3. **Perceptron Multicouches** (Deep Learning)

   - Réseau de neurones feed-forward
   - Capacité d'apprentissage non-linéaire
   - Architecture adaptative

### Métriques d'Évaluation

- **Accuracy** : Proportion de prédictions correctes
- **Precision** : Proportion de vrais positifs parmi les prédictions positives
- **Recall** : Proportion de vrais positifs détectés
- **F1-Score** : Moyenne harmonique precision/recall
- **Matrice de confusion** : Analyse détaillée des erreurs

## Veille Technologique Réalisée

### MLOps : Machine Learning Operations

**Définition** : MLOps = ML + DevOps + Data Engineering

Le MLOps est un ensemble de pratiques qui vise à **industrialiser** le cycle de vie des modèles ML :

#### Composants Clés

- **CI/CD pour ML** : Intégration et déploiement continus
- **Versioning** : Code, données, modèles
- **Monitoring** : Performance et dérive des données
- **Automatisation** : Pipelines d'entraînement et déploiement
- **Collaboration** : Entre data scientists et ingénieurs

#### Bénéfices Observés

- **Reproductibilité** des expériences
- **Scalabilité** des solutions
- **Réduction du time-to-market**
- **Qualité** et **gouvernance** améliorées

### MLflow : Plateforme MLOps

**MLflow** est l'outil central choisi pour ce projet :

#### Fonctionnalités Utilisées

1. **MLflow Tracking** : Logging automatique des runs
2. **MLflow Models** : Packaging et versioning
3. **MLflow UI** : Interface de comparaison
4. **Model Registry** : Gestion centralisée des modèles

#### Architecture Implémentée

```
Data → Preprocessing → MLflow Experiments → Model Registry → Flask API → Production
```

## Architecture du Projet

### Structure des Fichiers

```
waterflow/
├── main.ipynb              # Notebook d'analyse exploratoire
├── experiment.py           # Script d'expérimentation MLflow
├── app.py                  # API Flask pour le déploiement
├── tests/                  # Tests unitaires et fonctionnels
│   ├── test_unit.py        # Tests unitaires
│   ├── test_functional.py  # Tests d'intégration
│   └── test_regression.py  # Tests de non-régression
├── data/                   # Données préparées
│   ├── X_train.pkl         # Features d'entraînement
│   ├── X_val.pkl           # Features de validation
│   ├── y_train.pkl         # Labels d'entraînement
│   ├── y_val.pkl           # Labels de validation
│   ├── scaler.pkl          # Objet de normalisation
│   └── metadata.json       # Métadonnées du dataset
├── requirements.txt        # Dépendances Python
└── README.md              # Documentation complète
```

### Flux de Données

1. **Ingestion** : Chargement du dataset eau depuis URL
2. **Preprocessing** : Nettoyage, imputation, normalisation
3. **Expérimentation** : MLflow tracking des modèles
4. **Sélection** : Comparaison et promotion du meilleur modèle
5. **Déploiement** : API Flask avec chargement automatique
6. **Monitoring** : Tests continus et validation

## Installation et Utilisation

### Prérequis

- Python 3.8+
- pip ou conda
- Git

### Installation

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/waterflow
cd waterflow

# 2. Installer les dépendances
pip install -r requirements.txt
```

### Utilisation

#### 1. Analyse Exploratoire

```bash
# Ouvrir le notebook principal
jupyter notebook main.ipynb
```

#### 2. Expérimentation MLflow

```bash
# Entraîner et comparer les modèles
python experiment.py
```

#### 3. Déploiement API

```bash
# Lancer l'API Flask (MLflow sera démarré automatiquement)
python app.py
```

#### 4. Tests

```bash
# Exécuter tous les tests
python -m pytest tests/ -v

# Tests spécifiques
python -m pytest tests/test_unit.py -v
python -m pytest tests/test_functional.py -v
```

### Accès aux Interfaces

- **MLflow UI** : http://localhost:5000
- **API Flask** : http://localhost:5001
- **Documentation API** : http://localhost:5001/docs

## Résultats et Performance

### Métriques Obtenues

Les modèles sont évalués et comparés automatiquement via MLflow :

| Modèle       | Accuracy           | Precision       | Recall          | F1-Score        |
| ------------- | ------------------ | --------------- | --------------- | --------------- |
| Random Forest | **Meilleur** | tracking MLflow | tracking MLflow | tracking MLflow |
| XGBoost       | tracking MLflow    | tracking MLflow | tracking MLflow | tracking MLflow |
| MLP           | tracking MLflow    | tracking MLflow | tracking MLflow | tracking MLflow |

*Les métriques exactes sont visibles dans l'interface MLflow après exécution.*

### Sélection du Modèle

Le **meilleur modèle** est automatiquement :

1. **Identifié** par comparaison des métriques MLflow
2. **Promu** vers le Model Registry
3. **Déployé** dans l'API Flask
4. **Testé** via la suite de tests automatisés

## Tests et Qualité

### Suite de Tests Implémentée

1. **Tests Unitaires** (`test_unit.py`)

   - Fonctions de preprocessing
   - Composants MLflow
   - Logique métier
2. **Tests Fonctionnels** (`test_functional.py`)

   - API Flask endpoints
   - Intégration MLflow
   - Flux end-to-end
3. **Tests de Non-Régression** (`test_regression.py`)

   - Stabilité des prédictions
   - Performance des modèles
   - Compatibilité versions

### Couverture de Tests

```bash
# Exécuter avec couverture
python -m pytest tests/ --cov=. --cov-report=html
```

## Monitoring et Maintenance

### Suivi des Performances

- **MLflow Tracking** : Métriques automatiques
- **Model Registry** : Versioning et lifecycle
- **API Logs** : Monitoring des prédictions
- **Tests Continus** : Validation automatisée

### Mise à Jour des Modèles

1. Nouveau dataset → `experiment.py`
2. Comparaison automatique MLflow
3. Promotion si amélioration
4. Redéploiement API automatique

## Conclusion et Perspectives

### Réalisations du Projet

- **MLOps Complet** : Pipeline industriel bout-en-bout
- **Reproducibilité** : Tracking et versioning intégraux
- **Qualité** : Suite de tests complète
- **Déploiement** : API prête pour production
- **Documentation** : Complète et détaillée

### Impact et Valeur Ajoutée

Ce projet démontre une **implémentation concrète** des principes MLOps appliqués à un problème de **santé publique**. La prédiction automatisée de la qualité de l'eau peut contribuer à :

- **Prévention sanitaire** : Détection précoce d'eau non potable
- **Optimisation ressources** : Automatisation des contrôles qualité
- **Aide à la décision** : Support pour autorités sanitaires
- **Scalabilité** : Déploiement sur multiple sites

### Perspectives d'Évolution

1. **Modèles Avancés**

   - Ensemble methods (stacking, blending)
   - Deep Learning avec architectures spécialisées
   - Auto-ML pour optimisation automatique
2. **Données Enrichies**

   - Intégration données géographiques
   - Série temporelle pour tendances
   - Sources multiples (IoT, satellites)
3. **Déploiement Cloud**

   - Containerisation Docker
   - Orchestration Kubernetes
   - CI/CD GitHub Actions
4. **Interface Utilisateur**

   - Dashboard interactif (Streamlit/Dash)
   - Application mobile
   - Alertes temps réel

### Technologies Exploitées

| Catégorie       | Technologies                       |
| ---------------- | ---------------------------------- |
| **MLOps**  | MLflow, Model Registry             |
| **ML/DL**  | Scikit-learn, XGBoost, TensorFlow  |
| **Data**   | Pandas, NumPy, Matplotlib, Seaborn |
| **API**    | Flask, REST, JSON                  |
| **Tests**  | Pytest, Coverage                   |
| **DevOps** | Git, Python, Requirements.txt      |

---

### Équipe et Contribution

**Projet académique** réalisé dans le cadre d'une évaluation MLOps, démontrant la maîtrise des outils et pratiques modernes du Machine Learning en production.

---

*Documentation générée automatiquement - Dernière mise à jour : Juillet 2025*
├── run_tests.py           # Suite de tests
├── tests/                 # Tests complets
│   ├── test_unit.py       # Tests unitaires
│   ├── test_functional.py # Tests fonctionnels
│   └── test_regression.py # Tests non-régression
├── data/                  # Données préparées (généré par notebook)
├── requirements.txt       # Dépendances
└── README.md             # Documentation

```

## Installation et Configuration

### 1. Clonage et Installation

```bash
git clone <repository-url>
cd waterflow
pip install -r requirements.txt
```

### 2. Démarrage du Serveur MLflow

```bash
mlflow server --host 127.0.0.1 --port 5000
```

L'interface MLflow sera accessible sur `http://localhost:5000`

## Utilisation Étape par Étape

### Étapes 1-3 : EDA et Préparation (Notebook)

1. Ouvrez `main.ipynb` dans Jupyter/VS Code
2. Exécutez toutes les cellules pour :
   - Analyser le dataset (9 variables, 3276 échantillons)
   - Réaliser l'analyse exploratoire
   - Préprocesser et sauvegarder les données

### Étapes 4-9 : Entraînement avec MLflow

```bash
python mlflow_training.py
```

Cette étape :

- Configure MLflow et gère les expériences
- Entraîne 3 modèles (Random Forest, XGBoost, MLP)
- Log les paramètres, métriques et modèles
- Compare les performances

### Étapes 10-12 : Déploiement API

```bash
python flask_app.py
```

L'API sera accessible sur `http://127.0.0.1:5001` avec :

- Interface web de test
- Endpoint JSON `/predict`
- Documentation API

### Tests Logiciels

```bash
# Tous les tests
python run_tests.py all

# Tests spécifiques
python run_tests.py unit
python run_tests.py functional  
python run_tests.py regression
```

## Modèles Disponibles

### Random Forest

- **Avantages** : Robuste, interprétable, gestion native des features numériques
- **Paramètres** : 100 arbres, profondeur max 10, split min 5

### XGBoost

- **Avantages** : Performance élevée, gestion relations complexes, régularisation
- **Paramètres** : 100 estimateurs, learning rate 0.1, profondeur max 6

### Perceptron Multicouches (MLP)

- **Avantages** : Apprentissage relations non-linéaires, flexibilité
- **Architecture** : 128 → 64 → 32 neurones, dropout 0.3, Adam optimizer

## API Endpoints

### `GET /`

Interface web avec formulaire de test

### `POST /predict`

Prédiction de qualité d'eau

**Entrée JSON :**

```json
{
  "ph": 7.0,
  "Hardness": 200,
  "Solids": 20000,
  "Chloramines": 7,
  "Sulfate": 350,
  "Conductivity": 400,
  "Organic_carbon": 14,
  "Trihalomethanes": 80,
  "Turbidity": 4
}
```

**Sortie JSON :**

```json
{
  "prediction": 1,
  "prediction_label": "Potable",
  "probability_potable": 0.75,
  "probability_non_potable": 0.25,
  "model_info": {"name": "Random Forest", "version": "latest"}
}
```

### `GET /health`

Vérification santé de l'API

### `GET /models`

Informations sur le modèle chargé

## Types de Tests

### Tests Unitaires

- Tests composants isolés
- Validation fonctions individuelles
- Mock des dépendances externes

### Tests Fonctionnels

- Tests d'intégration
- Flux complets end-to-end
- Validation API et MLflow

### Tests de Non-Régression

- Seuils performance minimaux
- Cohérence prédictions
- Benchmarks temporels

## MLflow Interface

Accédez à `http://localhost:5000` pour :

1. **Explorer les expériences** et comparer les runs
2. **Analyser les métriques** et paramètres
3. **Visualiser les artefacts** (matrices de confusion)
4. **Gérer le Model Registry**
5. **Déployer les modèles** sélectionnés
