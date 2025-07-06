#!/usr/bin/env python3
"""
Tests unitaires pour le projet Drink Safe
Tests des fonctions individuelles et composants isolés

Utilisation:
    python -m pytest tests/test_unit.py -v
"""

import pytest
import numpy as np
import json
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Ajout du répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment import WaterQualityMLflowTrainer
from app import WaterQualityPredictor

class TestDataPreparation:
    """Tests pour la préparation des données"""
    
    def setup_method(self):
        """Préparation avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Création de données de test
        os.makedirs('data', exist_ok=True)
        
        # Données d'exemple
        self.X_train = np.random.rand(100, 9)
        self.X_val = np.random.rand(20, 9)
        self.y_train = np.random.randint(0, 2, 100)
        self.y_val = np.random.randint(0, 2, 20)
        
        # Sauvegarde des données de test
        np.save('data/X_train.npy', self.X_train)
        np.save('data/X_val.npy', self.X_val)
        np.save('data/y_train.npy', self.y_train)
        np.save('data/y_val.npy', self.y_val)
        
        # Métadonnées de test
        self.metadata = {
            'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
            'n_features': 9,
            'n_train_samples': 100,
            'n_val_samples': 20,
            'train_potable_ratio': 0.5,
            'val_potable_ratio': 0.4
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(self.metadata, f)
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_data_loading_success(self):
        """Test le chargement réussi des données"""
        with patch('mlflow_training.mlflow') as mock_mlflow:
            mock_mlflow.set_tracking_uri.return_value = None
            mock_mlflow.set_experiment.return_value = None
            
            trainer = WaterQualityMLflowTrainer()
            
            assert trainer.X_train.shape == (100, 9)
            assert trainer.X_val.shape == (20, 9)
            assert trainer.y_train.shape == (100,)
            assert trainer.y_val.shape == (20,)
            assert trainer.metadata['n_features'] == 9
    
    def test_data_loading_missing_files(self):
        """Test la gestion des fichiers manquants"""
        # Suppression d'un fichier essentiel
        os.remove('data/X_train.npy')
        
        with patch('mlflow_training.mlflow') as mock_mlflow:
            mock_mlflow.set_tracking_uri.return_value = None
            mock_mlflow.set_experiment.return_value = None
            
            with pytest.raises(FileNotFoundError):
                WaterQualityMLflowTrainer()
    
    def test_metadata_validation(self):
        """Test la validation des métadonnées"""
        with patch('mlflow_training.mlflow') as mock_mlflow:
            mock_mlflow.set_tracking_uri.return_value = None
            mock_mlflow.set_experiment.return_value = None
            
            trainer = WaterQualityMLflowTrainer()
            
            assert len(trainer.metadata['feature_names']) == 9
            assert 'ph' in trainer.metadata['feature_names']
            assert 'Potability' not in trainer.metadata['feature_names']


class TestMLflowTrainer:
    """Tests pour le trainer MLflow"""
    
    def setup_method(self):
        """Préparation avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Création de données de test minimales
        os.makedirs('data', exist_ok=True)
        
        X_train = np.random.rand(50, 9)
        X_val = np.random.rand(10, 9)
        y_train = np.random.randint(0, 2, 50)
        y_val = np.random.randint(0, 2, 10)
        
        np.save('data/X_train.npy', X_train)
        np.save('data/X_val.npy', X_val)
        np.save('data/y_train.npy', y_train)
        np.save('data/y_val.npy', y_val)
        
        metadata = {
            'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
            'n_features': 9,
            'n_train_samples': 50,
            'n_val_samples': 10,
            'train_potable_ratio': 0.5,
            'val_potable_ratio': 0.4
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch('mlflow_training.mlflow')
    def test_metrics_calculation(self, mock_mlflow):
        """Test le calcul des métriques"""
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.set_experiment.return_value = None
        
        trainer = WaterQualityMLflowTrainer()
        
        # Données de test pour les métriques
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = trainer._calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    @patch('mlflow_training.mlflow')
    def test_mlflow_connection_failure(self, mock_mlflow):
        """Test la gestion d'échec de connexion MLflow"""
        mock_mlflow.set_tracking_uri.side_effect = Exception("Connexion échouée")
        
        trainer = WaterQualityMLflowTrainer()
        
        assert trainer.mlflow_available == False
    
    @patch('mlflow_training.mlflow')
    @patch('mlflow_training.RandomForestClassifier')
    def test_random_forest_training(self, mock_rf, mock_mlflow):
        """Test l'entraînement Random Forest"""
        # Configuration des mocks
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.set_experiment.return_value = None
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 1])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
        mock_rf.return_value = mock_model
        
        trainer = WaterQualityMLflowTrainer()
        
        model, metrics = trainer.train_random_forest()
        
        assert model is not None
        assert metrics is not None
        assert 'accuracy' in metrics
        mock_model.fit.assert_called_once()
        mock_model.predict.assert_called_once()


class TestWaterQualityPredictor:
    """Tests pour le prédicteur Flask"""
    
    def setup_method(self):
        """Préparation avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_feature_validation(self):
        """Test la validation des features d'entrée"""
        # Création de données de test minimales
        os.makedirs('data', exist_ok=True)
        
        metadata = {
            'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Mock du scaler
        with patch('flask_app.joblib.load') as mock_joblib, \
             patch('flask_app.mlflow') as mock_mlflow:
            
            mock_scaler = Mock()
            mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
            mock_joblib.return_value = mock_scaler
            
            mock_model = Mock()
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_mlflow.sklearn.load_model.return_value = mock_model
            
            predictor = WaterQualityPredictor()
            
            # Test avec toutes les features
            complete_features = {
                'ph': 7.0, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
                'Sulfate': 350, 'Conductivity': 400, 'Organic_carbon': 14,
                'Trihalomethanes': 80, 'Turbidity': 4
            }
            
            result = predictor.predict(complete_features)
            assert 'prediction' in result
            assert 'probability_potable' in result
            
            # Test avec features manquantes
            incomplete_features = {'ph': 7.0, 'Hardness': 200}
            
            with pytest.raises(ValueError) as excinfo:
                predictor.predict(incomplete_features)
            assert "Features manquantes" in str(excinfo.value)
    
    def test_prediction_output_format(self):
        """Test le format de sortie des prédictions"""
        os.makedirs('data', exist_ok=True)
        
        metadata = {
            'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        with patch('flask_app.joblib.load') as mock_joblib, \
             patch('flask_app.mlflow') as mock_mlflow:
            
            mock_scaler = Mock()
            mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
            mock_joblib.return_value = mock_scaler
            
            mock_model = Mock()
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_mlflow.sklearn.load_model.return_value = mock_model
            
            predictor = WaterQualityPredictor()
            
            features = {
                'ph': 7.0, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
                'Sulfate': 350, 'Conductivity': 400, 'Organic_carbon': 14,
                'Trihalomethanes': 80, 'Turbidity': 4
            }
            
            result = predictor.predict(features)
            
            # Vérification de la structure de sortie
            required_keys = ['prediction', 'prediction_label', 'probability_potable', 
                           'probability_non_potable', 'model_info']
            
            for key in required_keys:
                assert key in result
            
            assert result['prediction'] in [0, 1]
            assert result['prediction_label'] in ['Potable', 'Non Potable']
            assert 0 <= result['probability_potable'] <= 1
            assert 0 <= result['probability_non_potable'] <= 1
            assert abs(result['probability_potable'] + result['probability_non_potable'] - 1.0) < 1e-6


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires"""
    
    def test_feature_names_consistency(self):
        """Test la cohérence des noms de features"""
        expected_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                           'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        assert len(expected_features) == 9
        assert all(isinstance(feature, str) for feature in expected_features)
    
    def test_data_types_validation(self):
        """Test la validation des types de données"""
        # Test avec des données valides
        valid_sample = {
            'ph': 7.0, 'Hardness': 200.5, 'Solids': 20000, 'Chloramines': 7.2,
            'Sulfate': 350.1, 'Conductivity': 400, 'Organic_carbon': 14.5,
            'Trihalomethanes': 80.3, 'Turbidity': 4.1
        }
        
        for key, value in valid_sample.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert np.isfinite(value)


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v"])
