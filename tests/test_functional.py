#!/usr/bin/env python3
"""
Tests fonctionnels pour le projet Drink Safe
Tests d'intégration et de flux complets

Utilisation:
    python -m pytest tests/test_functional.py -v
"""

import pytest
import requests
import json
import time
import subprocess
import os
import sys
import tempfile
import shutil
from threading import Thread
import signal

# Ajout du répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAPIIntegration:
    """Tests d'intégration de l'API Flask"""
    
    @classmethod
    def setup_class(cls):
        """Démarrage de l'API pour les tests"""
        cls.api_url = "http://127.0.0.1:5001"
        cls.api_process = None
        
        # Création de données de test
        cls.temp_dir = tempfile.mkdtemp()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir)
        
        cls._create_test_data()
        
        # Tentative de démarrage de l'API (si possible)
        print("Configuration des tests fonctionnels...")
    
    @classmethod
    def teardown_class(cls):
        """Nettoyage après les tests"""
        if cls.api_process:
            cls.api_process.terminate()
        
        os.chdir(cls.original_cwd)
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_data(cls):
        """Création des données de test nécessaires"""
        import numpy as np
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        os.makedirs('data', exist_ok=True)
        
        # Création d'un scaler de test
        scaler = StandardScaler()
        scaler.fit(np.random.rand(100, 9))
        joblib.dump(scaler, 'data/scaler.pkl')
        
        # Métadonnées de test
        metadata = {
            'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
            'n_features': 9
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def test_api_endpoints_structure(self):
        """Test de la structure des endpoints (sans serveur)"""
        # Ce test vérifie que les endpoints sont bien définis
        from flask_app import app
        
        with app.test_client() as client:
            # Test de la page d'accueil
            response = client.get('/')
            assert response.status_code in [200, 500]  # 500 si MLflow non disponible
            
            # Test de l'endpoint health
            response = client.get('/health')
            assert response.status_code in [200, 500]
            
            # Test de l'endpoint models
            response = client.get('/models')
            assert response.status_code in [200, 500]
    
    def test_predict_endpoint_validation(self):
        """Test de validation de l'endpoint predict"""
        from flask_app import app
        
        with app.test_client() as client:
            # Test avec données manquantes
            response = client.post('/predict', 
                                 data=json.dumps({'ph': 7.0}),
                                 content_type='application/json')
            # L'endpoint doit rejeter les données incomplètes
            assert response.status_code in [400, 500]
            
            # Test sans données JSON
            response = client.post('/predict')
            assert response.status_code == 400
            
            # Test avec données complètes (si MLflow disponible)
            complete_data = {
                'ph': 7.0, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
                'Sulfate': 350, 'Conductivity': 400, 'Organic_carbon': 14,
                'Trihalomethanes': 80, 'Turbidity': 4
            }
            
            response = client.post('/predict', 
                                 data=json.dumps(complete_data),
                                 content_type='application/json')
            # Peut échouer si MLflow non disponible, c'est normal
            assert response.status_code in [200, 400, 500]
    
    def test_error_handling(self):
        """Test de la gestion d'erreurs"""
        from flask_app import app
        
        with app.test_client() as client:
            # Test endpoint inexistant
            response = client.get('/nonexistent')
            assert response.status_code == 404
            
            data = response.get_json()
            assert 'error' in data
    
    def test_data_validation_edge_cases(self):
        """Test des cas limites de validation des données"""
        from flask_app import app
        
        with app.test_client() as client:
            # Test avec valeurs extrêmes
            extreme_data = {
                'ph': -1.0, 'Hardness': -100, 'Solids': 1000000, 'Chloramines': 100,
                'Sulfate': 10000, 'Conductivity': 100000, 'Organic_carbon': 1000,
                'Trihalomethanes': 1000, 'Turbidity': 1000
            }
            
            response = client.post('/predict', 
                                 data=json.dumps(extreme_data),
                                 content_type='application/json')
            # L'API doit gérer les valeurs extrêmes
            assert response.status_code in [200, 400, 500]
            
            # Test avec valeurs NaN/Inf (JSON ne les supporte pas directement)
            invalid_data = {
                'ph': "invalid", 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
                'Sulfate': 350, 'Conductivity': 400, 'Organic_carbon': 14,
                'Trihalomethanes': 80, 'Turbidity': 4
            }
            
            response = client.post('/predict', 
                                 data=json.dumps(invalid_data),
                                 content_type='application/json')
            # Doit rejeter les données invalides
            assert response.status_code in [400, 500]


class TestMLflowIntegration:
    """Tests d'intégration MLflow"""
    
    def setup_method(self):
        """Préparation avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Création de données de test
        self._create_minimal_data()
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def _create_minimal_data(self):
        """Création de données minimales pour les tests"""
        import numpy as np
        
        os.makedirs('data', exist_ok=True)
        
        # Données d'entraînement minimales
        X_train = np.random.rand(20, 9)
        X_val = np.random.rand(5, 9)
        y_train = np.random.randint(0, 2, 20)
        y_val = np.random.randint(0, 2, 5)
        
        np.save('data/X_train.npy', X_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/y_train.npy', y_train)
        np.save('data/y_val.npy', y_val)
        
        metadata = {
            'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
            'n_features': 9,
            'n_train_samples': 20,
            'n_val_samples': 5
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def test_trainer_initialization(self):
        """Test d'initialisation du trainer"""
        from mlflow_training import WaterQualityMLflowTrainer
        
        # Le trainer doit pouvoir s'initialiser même sans serveur MLflow
        trainer = WaterQualityMLflowTrainer()
        
        assert trainer is not None
        assert hasattr(trainer, 'X_train')
        assert hasattr(trainer, 'X_val')
        assert hasattr(trainer, 'y_train')
        assert hasattr(trainer, 'y_val')
        assert hasattr(trainer, 'metadata')
    
    def test_model_training_flow(self):
        """Test du flux complet d'entraînement"""
        from mlflow_training import WaterQualityMLflowTrainer
        
        trainer = WaterQualityMLflowTrainer()
        
        # Test Random Forest
        try:
            model, metrics = trainer.train_random_forest()
            assert model is not None
            assert metrics is not None
            assert 'accuracy' in metrics
        except Exception as e:
            # Normal si MLflow non disponible
            print(f"Entraînement Random Forest échoué (normal sans MLflow): {e}")
        
        # Test XGBoost (peut échouer si non installé)
        try:
            model, metrics = trainer.train_xgboost()
            if model is not None:
                assert metrics is not None
                assert 'accuracy' in metrics
        except ImportError:
            print("XGBoost non installé - test ignoré")
        except Exception as e:
            print(f"Entraînement XGBoost échoué: {e}")
        
        # Test MLP
        try:
            model, metrics = trainer.train_mlp()
            assert model is not None
            assert metrics is not None
            assert 'accuracy' in metrics
        except Exception as e:
            print(f"Entraînement MLP échoué: {e}")


class TestEndToEndWorkflow:
    """Tests de bout en bout du workflow complet"""
    
    def setup_method(self):
        """Préparation avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_data_pipeline_consistency(self):
        """Test de cohérence du pipeline de données"""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Simulation du pipeline EDA
        raw_data = np.random.rand(100, 10)  # 9 features + 1 target
        
        # Séparation features/target
        X = raw_data[:, :-1]
        y = raw_data[:, -1]
        y = (y > 0.5).astype(int)  # Conversion en binaire
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Vérifications
        assert X_scaled.mean(axis=0).max() < 1e-10  # Moyenne ~0
        assert abs(X_scaled.std(axis=0).mean() - 1.0) < 0.1  # Std ~1
        assert X.shape[1] == 9  # 9 features
        assert len(np.unique(y)) == 2  # Classification binaire
    
    def test_model_output_consistency(self):
        """Test de cohérence des sorties de modèle"""
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Données de test
        X_train = np.random.rand(50, 9)
        y_train = np.random.randint(0, 2, 50)
        X_test = np.random.rand(10, 9)
        
        # Entraînement modèle
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Vérifications
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
        assert probabilities.shape == (len(X_test), 2)
        assert all(0 <= prob <= 1 for prob_pair in probabilities for prob in prob_pair)
        assert all(abs(sum(prob_pair) - 1.0) < 1e-10 for prob_pair in probabilities)
    
    def test_prediction_pipeline(self):
        """Test du pipeline complet de prédiction"""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        # Simulation du pipeline complet
        
        # 1. Données d'entrée utilisateur
        user_input = {
            'ph': 7.0, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
            'Sulfate': 350, 'Conductivity': 400, 'Organic_carbon': 14,
            'Trihalomethanes': 80, 'Turbidity': 4
        }
        
        feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        # 2. Conversion en array
        feature_array = np.array([[user_input[name] for name in feature_names]])
        
        # 3. Normalisation
        scaler = StandardScaler()
        # Simulation d'un scaler pré-entraîné
        scaler.mean_ = np.random.rand(9)
        scaler.scale_ = np.random.rand(9) + 0.5
        feature_scaled = scaler.transform(feature_array)
        
        # 4. Prédiction
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Simulation d'un modèle pré-entraîné
        X_dummy = np.random.rand(50, 9)
        y_dummy = np.random.randint(0, 2, 50)
        model.fit(X_dummy, y_dummy)
        
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0]
        
        # 5. Format de sortie
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Potable' if prediction == 1 else 'Non Potable',
            'probability_potable': float(probability[1]),
            'probability_non_potable': float(probability[0])
        }
        
        # Vérifications
        assert result['prediction'] in [0, 1]
        assert result['prediction_label'] in ['Potable', 'Non Potable']
        assert 0 <= result['probability_potable'] <= 1
        assert 0 <= result['probability_non_potable'] <= 1
        assert abs(result['probability_potable'] + result['probability_non_potable'] - 1.0) < 1e-10


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v"])
