#!/usr/bin/env python3
"""
Application Flask pour le déploiement du modèle de prédiction de qualité d'eau
Projet Drink Safe - Waterflow

Cette application charge le meilleur modèle depuis MLflow et fournit une API REST
pour effectuer des prédictions de qualité d'eau en temps réel.

Endpoints:
- GET /              : Interface web interactive
- POST /predict      : Prédiction de qualité d'eau (JSON)
- GET /health        : Santé de l'application
- GET /models        : Informations sur le modèle chargé

Utilisation:
    python app.py
"""

import os
import json
import logging
import subprocess
import time
import threading
import signal
import atexit
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template_string
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variable globale pour le processus MLflow
mlflow_process = None

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

def start_mlflow_server():
    """Démarre le serveur MLflow si nécessaire"""
    global mlflow_process
    
    try:
        # Vérifier si le serveur est déjà en cours
        import requests
        response = requests.get("http://127.0.0.1:5000", timeout=2)
        logger.info("Serveur MLflow déjà en cours d'exécution")
        return True
    except:
        pass
    
    try:
        logger.info("Démarrage du serveur MLflow...")
        mlflow_process = subprocess.Popen(
            ['mlflow', 'server', '--host', '127.0.0.1', '--port', '5000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Attendre que le serveur soit prêt
        for i in range(30):  # Attendre jusqu'à 30 secondes
            try:
                import requests
                response = requests.get("http://127.0.0.1:5000", timeout=1)
                logger.info("Serveur MLflow démarré avec succès")
                return True
            except:
                time.sleep(1)
        
        logger.error("Impossible de vérifier le démarrage du serveur MLflow")
        return False
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de MLflow: {e}")
        return False

def stop_mlflow_server():
    """Arrête le serveur MLflow"""
    global mlflow_process
    if mlflow_process:
        try:
            if os.name == 'nt':
                # Windows
                mlflow_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Unix/Linux/Mac
                mlflow_process.terminate()
            mlflow_process.wait(timeout=5)
            logger.info("Serveur MLflow arrêté")
        except:
            if mlflow_process:
                mlflow_process.kill()
            logger.info("Serveur MLflow forcé à s'arrêter")

# Enregistrer la fonction d'arrêt
atexit.register(stop_mlflow_server)

class WaterQualityPredictor:
    """Prédicteur de qualité d'eau utilisant MLflow"""
    
    def __init__(self, mlflow_uri="http://127.0.0.1:5000"):
        """
        Initialise le prédicteur
        
        Args:
            mlflow_uri (str): URI du serveur MLflow
        """
        self.mlflow_uri = mlflow_uri
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_info = {}
        self.available_models = []
        
        self._setup_mlflow()
        self._load_preprocessing()
        self._load_available_models()
        self._load_best_model()
    
    def _setup_mlflow(self):
        """Configure la connexion MLflow"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            logger.info(f"MLflow configuré : {self.mlflow_uri}")
        except Exception as e:
            logger.error(f"Erreur MLflow : {e}")
            raise
    
    def _load_preprocessing(self):
        """Charge le scaler et les métadonnées depuis le dossier data/"""
        try:
            # Chargement du scaler
            scaler_path = 'data/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler chargé depuis data/")
            
            # Chargement des métadonnées
            metadata_path = 'data/metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                logger.info("Métadonnées chargées depuis data/")
            else:
                # Features par défaut si pas de métadonnées
                self.feature_names = [
                    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
                ]
                logger.warning("Utilisation des features par défaut")
                
        except Exception as e:
            logger.error(f"Erreur chargement preprocessing : {e}")
            # Configuration par défaut
            self.feature_names = [
                'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]
    
    def _load_available_models(self):
        """Charge la liste des modèles disponibles"""
        try:
            client = mlflow.MlflowClient()
            model_candidates = [
                "water_quality_random_forest",
                "water_quality_xgboost", 
                "water_quality_mlp"
            ]
            
            self.available_models = []
            for model_name in model_candidates:
                try:
                    versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
                    if versions:
                        latest_version = versions[0]
                        run = client.get_run(latest_version.run_id)
                        accuracy = run.data.metrics.get('accuracy', 0)
                        
                        self.available_models.append({
                            'name': model_name,
                            'version': latest_version.version,
                            'accuracy': accuracy,
                            'run_id': latest_version.run_id
                        })
                        logger.info(f"Modèle disponible : {model_name} v{latest_version.version} (Accuracy: {accuracy:.3f})")
                except Exception as e:
                    logger.warning(f"Modèle {model_name} non disponible : {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles : {e}")
            self.available_models = []
    
    def _load_best_model(self):
        """Charge le meilleur modèle depuis le Model Registry MLflow"""
        try:
            # Liste des modèles à tester par ordre de préférence
            model_candidates = [
                "water_quality_random_forest",
                "water_quality_xgboost", 
                "water_quality_mlp"
            ]
            
            best_model = None
            best_accuracy = 0
            best_model_name = None
            
            client = mlflow.MlflowClient()
            
            for model_name in model_candidates:
                try:
                    # Récupération des versions du modèle
                    versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
                    
                    if versions:
                        latest_version = versions[0]
                        model_uri = f"models:/{model_name}/{latest_version.version}"
                        
                        # Chargement du modèle
                        model = mlflow.pyfunc.load_model(model_uri)
                        
                        # Récupération des métriques
                        run = client.get_run(latest_version.run_id)
                        accuracy = run.data.metrics.get('accuracy', 0)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model
                            best_model_name = model_name
                            self.model_info = {
                                'name': model_name,
                                'version': latest_version.version,
                                'accuracy': accuracy,
                                'run_id': latest_version.run_id
                            }
                        
                        logger.info(f"Modèle trouvé : {model_name} v{latest_version.version} (Accuracy: {accuracy:.3f})")
                
                except Exception as e:
                    logger.warning(f"Modèle {model_name} non disponible : {e}")
                    continue
            
            if best_model is None:
                raise Exception("Aucun modèle disponible dans le registry MLflow")
            
            self.model = best_model
            logger.info(f"Meilleur modèle chargé : {best_model_name} (Accuracy: {best_accuracy:.3f})")
            
        except Exception as e:
            logger.error(f"Impossible de charger le modèle : {e}")
            logger.info("Vérifiez que :")
            logger.info("   1. Le serveur MLflow est démarré (port 5000)")
            logger.info("   2. Des modèles ont été entraînés avec experiment.py")
            raise
    
    def predict(self, water_data):
        """
        Effectue une prédiction de qualité d'eau
        
        Args:
            water_data (dict): Données avec les 9 paramètres physico-chimiques
            
        Returns:
            dict: Résultat de la prédiction
        """
        try:
            # Validation des features
            missing_features = [f for f in self.feature_names if f not in water_data]
            if missing_features:
                raise ValueError(f"Features manquantes : {missing_features}")
            
            # Conversion en array
            feature_values = [float(water_data[feature]) for feature in self.feature_names]
            feature_array = np.array([feature_values])
            
            # Normalisation si scaler disponible
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)
            
            # Prédiction
            prediction = self.model.predict(feature_array)[0]
            
            # Probabilité si supportée
            try:
                probabilities = self.model.predict_proba(feature_array)[0]
                probability_potable = probabilities[1]
                confidence = max(probabilities)
            except:
                probability_potable = float(prediction)
                confidence = 0.5 + abs(prediction - 0.5)
            
            return {
                'prediction': int(prediction),
                'potable': bool(prediction == 1),
                'probability_potable': float(probability_potable),
                'confidence': float(confidence),
                'prediction_text': 'Eau Potable' if prediction == 1 else 'Eau Non Potable',
                'timestamp': datetime.now().isoformat(),
                'model_info': self.model_info,
                'input_data': dict(zip(self.feature_names, feature_values))
            }
            
        except Exception as e:
            logger.error(f"Erreur prédiction : {e}")
            raise
    
    def switch_model(self, model_name):
        """
        Change le modèle actuel
        
        Args:
            model_name (str): Nom du modèle à charger
            
        Returns:
            bool: True si le changement a réussi
        """
        try:
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            
            if not versions:
                raise Exception(f"Modèle {model_name} non trouvé")
            
            latest_version = versions[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"
            
            # Chargement du nouveau modèle
            new_model = mlflow.pyfunc.load_model(model_uri)
            
            # Récupération des métriques
            run = client.get_run(latest_version.run_id)
            accuracy = run.data.metrics.get('accuracy', 0)
            
            # Mise à jour
            self.model = new_model
            self.model_info = {
                'name': model_name,
                'version': latest_version.version,
                'accuracy': accuracy,
                'run_id': latest_version.run_id
            }
            
            logger.info(f"Modèle changé vers : {model_name} v{latest_version.version}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur changement de modèle : {e}")
            return False

# Interface web HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drink Safe - Prédicteur de Qualité d'Eau</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container { 
            max-width: 1000px; margin: 0 auto; 
            background: white; border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 30px;
        }
        h1 { 
            color: #2c3e50; text-align: center; 
            margin-bottom: 10px; font-size: 2.5em;
        }
        .subtitle {
            text-align: center; color: #7f8c8d;
            margin-bottom: 30px; font-size: 1.1em;
        }
        .model-info {
            background: #ecf0f1; padding: 15px; 
            border-radius: 8px; margin-bottom: 30px;
        }
        .form-grid {
            display: grid; grid-template-columns: 1fr 1fr 1fr;
            gap: 20px; margin-bottom: 20px;
        }
        .form-group {
            display: flex; flex-direction: column;
        }
        label {
            font-weight: bold; margin-bottom: 5px;
            color: #2c3e50;
        }
        input[type="number"] {
            padding: 10px; border: 2px solid #bdc3c7;
            border-radius: 5px; font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="number"]:focus {
            outline: none; border-color: #3498db;
        }
        .predict-btn {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white; padding: 15px 30px;
            border: none; border-radius: 8px;
            font-size: 18px; cursor: pointer;
            transition: transform 0.2s;
            margin: 20px auto; display: block;
        }
        .predict-btn:hover {
            transform: translateY(-2px);
        }
        .result {
            margin-top: 30px; padding: 20px;
            border-radius: 10px; display: none;
            text-align: center; font-size: 18px;
        }
        .potable {
            background: #d5f4e6; border: 2px solid #27ae60;
            color: #1e8449;
        }
        .non-potable {
            background: #fadbd8; border: 2px solid #e74c3c;
            color: #c0392b;
        }
        .loading {
            display: none; text-align: center;
            color: #3498db; margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Drink Safe</h1>
        <p class="subtitle">Système de Prédiction de Qualité d'Eau par IA</p>
        
        <div class="model-info">
            <h3>Modèle Actuel</h3>
            <p><strong>Nom :</strong> {{ model_info.get('name', 'Non disponible') }}</p>
            <p><strong>Précision :</strong> {{ "%.1f%%" | format((model_info.get('accuracy', 0) * 100)) }}</p>
            <p><strong>Version :</strong> {{ model_info.get('version', 'N/A') }}</p>
            
            {% if available_models %}
            <div style="margin-top: 15px;">
                <strong>Changer de modèle :</strong>
                <select id="modelSelect" style="margin-left: 10px; padding: 5px;">
                    {% for model in available_models %}
                    <option value="{{ model.name }}" {% if model.name == model_info.get('name') %}selected{% endif %}>
                        {{ model.name }} ({{ "%.1f%%" | format(model.accuracy * 100) }})
                    </option>
                    {% endfor %}
                </select>
                <button onclick="changeModel()" style="margin-left: 10px; padding: 5px 10px;">Changer</button>
            </div>
            {% endif %}
        </div>
        
        <form id="predictionForm">
            <h3>Paramètres Physico-Chimiques</h3>
            <div class="form-grid">
                <div class="form-group">
                    <label for="ph">pH (6.5-8.5 optimal)</label>
                    <input type="number" id="ph" name="ph" step="0.01" value="7.0" required>
                </div>
                <div class="form-group">
                    <label for="Hardness">Dureté (mg/L)</label>
                    <input type="number" id="Hardness" name="Hardness" step="0.01" value="200" required>
                </div>
                <div class="form-group">
                    <label for="Solids">Solides dissous (ppm)</label>
                    <input type="number" id="Solids" name="Solids" step="0.01" value="20000" required>
                </div>
                <div class="form-group">
                    <label for="Chloramines">Chloramines (ppm)</label>
                    <input type="number" id="Chloramines" name="Chloramines" step="0.01" value="7" required>
                </div>
                <div class="form-group">
                    <label for="Sulfate">Sulfates (mg/L)</label>
                    <input type="number" id="Sulfate" name="Sulfate" step="0.01" value="350" required>
                </div>
                <div class="form-group">
                    <label for="Conductivity">Conductivité (μS/cm)</label>
                    <input type="number" id="Conductivity" name="Conductivity" step="0.01" value="400" required>
                </div>
                <div class="form-group">
                    <label for="Organic_carbon">Carbone organique (ppm)</label>
                    <input type="number" id="Organic_carbon" name="Organic_carbon" step="0.01" value="14" required>
                </div>
                <div class="form-group">
                    <label for="Trihalomethanes">Trihalométhanes (μg/L)</label>
                    <input type="number" id="Trihalomethanes" name="Trihalomethanes" step="0.01" value="80" required>
                </div>
                <div class="form-group">
                    <label for="Turbidity">Turbidité (NTU)</label>
                    <input type="number" id="Turbidity" name="Turbidity" step="0.01" value="4" required>
                </div>
            </div>
            
            <button type="submit" class="predict-btn">Analyser la Qualité de l'Eau</button>
        </form>
        
        <div class="loading" id="loading">
            <p>Analyse en cours...</p>
        </div>
        
        <div id="result" class="result"></div>
    </div>
    
    <script>
        function changeModel() {
            const selectedModel = document.getElementById('modelSelect').value;
            
            fetch('/switch_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({model_name: selectedModel})
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    alert('Modèle changé avec succès !');
                    location.reload();
                } else {
                    alert('Erreur lors du changement de modèle : ' + result.error);
                }
            })
            .catch(error => {
                alert('Erreur de communication : ' + error);
            });
        }
        
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Afficher le loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // Récupérer les données du formulaire
            const formData = new FormData(e.target);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = parseFloat(value);
            }
            
            // Envoyer la requête
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                document.getElementById('loading').style.display = 'none';
                
                const resultDiv = document.getElementById('result');
                
                if (result.error) {
                    resultDiv.innerHTML = `<h3>Erreur</h3><p>${result.error}</p>`;
                    resultDiv.className = 'result non-potable';
                } else {
                    const className = result.potable ? 'potable' : 'non-potable';
                    const icon = result.potable ? 'POTABLE' : 'NON POTABLE';
                    
                    resultDiv.innerHTML = `
                        <h3>${icon}</h3>
                        <p><strong>Probabilité de potabilité :</strong> ${(result.probability_potable * 100).toFixed(1)}%</p>
                        <p><strong>Confiance du modèle :</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Analysé le :</strong> ${new Date(result.timestamp).toLocaleString('fr-FR')}</p>
                    `;
                    resultDiv.className = `result ${className}`;
                }
                
                resultDiv.style.display = 'block';
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').innerHTML = `<h3>Erreur de communication</h3><p>${error}</p>`;
                document.getElementById('result').className = 'result non-potable';
                document.getElementById('result').style.display = 'block';
            });
        });
    </script>
</body>
</html>
"""

# Initialisation du prédicteur
try:
    # Démarrer le serveur MLflow
    start_mlflow_server()
    time.sleep(2)  # Attendre un peu pour que le serveur soit prêt
    
    predictor = WaterQualityPredictor()
    logger.info("Application initialisée avec succès")
except Exception as e:
    logger.error(f"Échec initialisation : {e}")
    predictor = None

# Routes Flask
@app.route('/')
def home():
    """Page d'accueil avec interface de test"""
    model_info = predictor.model_info if predictor else {}
    available_models = predictor.available_models if predictor else []
    return render_template_string(HTML_TEMPLATE, model_info=model_info, available_models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    if not predictor:
        return jsonify({'error': 'Prédicteur non initialisé'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Données JSON requises'}), 400
        
        result = predictor.predict(data)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': f'Données invalides : {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/health')
def health():
    """Endpoint de santé"""
    status = {
        'status': 'healthy' if predictor else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None,
        'model_info': predictor.model_info if predictor else None
    }
    return jsonify(status)

@app.route('/models')
def models_info():
    """Informations sur les modèles"""
    if not predictor:
        return jsonify({'error': 'Prédicteur non initialisé'}), 500
    
    return jsonify({
        'current_model': predictor.model_info,
        'available_models': predictor.available_models,
        'features': predictor.feature_names,
        'mlflow_uri': predictor.mlflow_uri
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Endpoint pour changer de modèle"""
    if not predictor:
        return jsonify({'success': False, 'error': 'Prédicteur non initialisé'}), 500
    
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'success': False, 'error': 'Nom du modèle requis'}), 400
        
        success = predictor.switch_model(model_name)
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'Modèle changé vers {model_name}',
                'new_model_info': predictor.model_info
            })
        else:
            return jsonify({'success': False, 'error': 'Échec du changement de modèle'}), 500
            
    except Exception as e:
        logger.error(f"Erreur changement modèle : {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    if predictor:
        print("DRINK SAFE - APPLICATION FLASK")
        print("=" * 35)
        print(f"Modèle chargé : {predictor.model_info.get('name', 'Inconnu')}")
        print(f"Précision : {predictor.model_info.get('accuracy', 0)*100:.1f}%")
        print(f"Interface web : http://127.0.0.1:5001")
        print(f"API REST : http://127.0.0.1:5001/predict")
        print("=" * 35)
        
        app.run(host='127.0.0.1', port=5001, debug=False)
    else:
        print("ERREUR : Impossible de démarrer l'application")
        print("Vérifiez que :")
        print("   1. Le serveur MLflow est démarré : mlflow server --host 127.0.0.1 --port 5000")
        print("   2. Les modèles sont entraînés : python experiment.py")
        print("   3. Les données sont préparées : exécution du notebook main.ipynb")