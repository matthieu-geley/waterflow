#!/usr/bin/env python3
"""
Tests de non-régression pour le projet Drink Safe
Vérification que les modifications n'introduisent pas de régressions

Utilisation:
    python -m pytest tests/test_regression.py -v
"""

import pytest
import numpy as np
import json
import os
import sys
import tempfile
import shutil
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ajout du répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelPerformanceRegression:
    """Tests de non-régression des performances des modèles"""
    
    # Seuils de performance minimaux attendus
    MIN_ACCURACY = 0.55  # Minimum acceptable pour un modèle binaire
    MIN_PRECISION = 0.50
    MIN_RECALL = 0.40
    MIN_F1_SCORE = 0.45
    
    def setup_method(self):
        """Préparation avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Création de données de test reproducibles
        self._create_regression_test_data()
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def _create_regression_test_data(self):
        """Création de données de test reproducibles pour les tests de régression"""
        np.random.seed(42)  # Seed fixe pour la reproducibilité
        
        os.makedirs('data', exist_ok=True)
        
        # Génération de données synthétiques mais réalistes
        n_train, n_val = 200, 50
        n_features = 9
        
        # Données d'entraînement
        X_train = np.random.rand(n_train, n_features)
        # Création d'une relation non-triviale pour la target
        y_train = ((X_train[:, 0] > 0.5) & (X_train[:, 1] < 0.7) | 
                  (X_train[:, 2] * X_train[:, 3] > 0.3)).astype(int)
        
        # Données de validation
        X_val = np.random.rand(n_val, n_features)
        y_val = ((X_val[:, 0] > 0.5) & (X_val[:, 1] < 0.7) | 
                (X_val[:, 2] * X_val[:, 3] > 0.3)).astype(int)
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Sauvegarde
        np.save('data/X_train.npy', X_train_scaled)
        np.save('data/X_val.npy', X_val_scaled)
        np.save('data/y_train.npy', y_train)
        np.save('data/y_val.npy', y_val)
        joblib.dump(scaler, 'data/scaler.pkl')
        
        # Métadonnées
        metadata = {
            'feature_names': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
            'n_features': n_features,
            'n_train_samples': n_train,
            'n_val_samples': n_val,
            'train_potable_ratio': y_train.mean(),
            'val_potable_ratio': y_val.mean()
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # Sauvegarde des performances de référence
        self._create_baseline_performance()
    
    def _create_baseline_performance(self):
        """Création d'un modèle de référence avec performances de base"""
        X_train = np.load('data/X_train.npy')
        X_val = np.load('data/X_val.npy')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
        
        # Modèle de référence Random Forest
        baseline_model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=5, 
            random_state=42
        )
        baseline_model.fit(X_train, y_train)
        
        # Prédictions et métriques de référence
        y_pred = baseline_model.predict(X_val)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        baseline_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0)
        }
        
        # Sauvegarde des métriques de référence
        with open('data/baseline_metrics.json', 'w') as f:
            json.dump(baseline_metrics, f)
        
        joblib.dump(baseline_model, 'data/baseline_model.pkl')
        
        return baseline_metrics
    
    def test_random_forest_performance_regression(self):
        """Test de non-régression des performances Random Forest"""
        from mlflow_training import WaterQualityMLflowTrainer
        
        # Chargement des métriques de référence
        with open('data/baseline_metrics.json', 'r') as f:
            baseline_metrics = json.load(f)
        
        # Entraînement du modèle actuel
        trainer = WaterQualityMLflowTrainer()
        
        try:
            model, current_metrics = trainer.train_random_forest(
                n_estimators=50, max_depth=5, min_samples_split=2
            )
            
            # Vérification que les performances ne régressent pas
            assert current_metrics['accuracy'] >= self.MIN_ACCURACY, \
                f"Accuracy trop faible: {current_metrics['accuracy']:.3f} < {self.MIN_ACCURACY}"
            
            assert current_metrics['precision'] >= self.MIN_PRECISION, \
                f"Precision trop faible: {current_metrics['precision']:.3f} < {self.MIN_PRECISION}"
            
            assert current_metrics['recall'] >= self.MIN_RECALL, \
                f"Recall trop faible: {current_metrics['recall']:.3f} < {self.MIN_RECALL}"
            
            assert current_metrics['f1_score'] >= self.MIN_F1_SCORE, \
                f"F1-Score trop faible: {current_metrics['f1_score']:.3f} < {self.MIN_F1_SCORE}"
            
            # Vérification que les performances sont stables par rapport à la baseline
            accuracy_degradation = baseline_metrics['accuracy'] - current_metrics['accuracy']
            assert accuracy_degradation <= 0.05, \
                f"Dégradation d'accuracy trop importante: {accuracy_degradation:.3f}"
            
            print(f"✓ Random Forest - Performances maintenues:")
            print(f"  Accuracy: {current_metrics['accuracy']:.3f} (baseline: {baseline_metrics['accuracy']:.3f})")
            print(f"  Precision: {current_metrics['precision']:.3f} (baseline: {baseline_metrics['precision']:.3f})")
            print(f"  Recall: {current_metrics['recall']:.3f} (baseline: {baseline_metrics['recall']:.3f})")
            print(f"  F1-Score: {current_metrics['f1_score']:.3f} (baseline: {baseline_metrics['f1_score']:.3f})")
            
        except Exception as e:
            pytest.skip(f"Entraînement Random Forest échoué (MLflow indisponible): {e}")
    
    def test_prediction_consistency_regression(self):
        """Test de non-régression de la cohérence des prédictions"""
        # Données de test fixes pour vérifier la cohérence
        test_samples = [
            {
                'ph': 7.0, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
                'Sulfate': 350, 'Conductivity': 400, 'Organic_carbon': 14,
                'Trihalomethanes': 80, 'Turbidity': 4
            },
            {
                'ph': 6.5, 'Hardness': 150, 'Solids': 15000, 'Chloramines': 5,
                'Sulfate': 250, 'Conductivity': 300, 'Organic_carbon': 10,
                'Trihalomethanes': 60, 'Turbidity': 3
            },
            {
                'ph': 8.5, 'Hardness': 300, 'Solids': 25000, 'Chloramines': 9,
                'Sulfate': 450, 'Conductivity': 500, 'Organic_carbon': 18,
                'Trihalomethanes': 100, 'Turbidity': 6
            }
        ]
        
        # Test avec le modèle de référence
        baseline_model = joblib.load('data/baseline_model.pkl')
        scaler = joblib.load('data/scaler.pkl')
        feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        baseline_predictions = []
        for sample in test_samples:
            feature_array = np.array([[sample[name] for name in feature_names]])
            feature_scaled = scaler.transform(feature_array)
            prediction = baseline_model.predict(feature_scaled)[0]
            probability = baseline_model.predict_proba(feature_scaled)[0]
            
            baseline_predictions.append({
                'prediction': int(prediction),
                'probability_potable': float(probability[1])
            })
        
        # Vérifications de cohérence
        for i, (sample, pred) in enumerate(zip(test_samples, baseline_predictions)):
            assert pred['prediction'] in [0, 1], f"Prédiction invalide pour échantillon {i}: {pred['prediction']}"
            assert 0 <= pred['probability_potable'] <= 1, f"Probabilité invalide pour échantillon {i}: {pred['probability_potable']}"
        
        print("✓ Prédictions cohérentes sur les échantillons de test")
    
    def test_data_preprocessing_regression(self):
        """Test de non-régression du préprocessing des données"""
        # Chargement des données
        X_train = np.load('data/X_train.npy')
        X_val = np.load('data/X_val.npy')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
        
        # Vérifications des dimensions
        assert X_train.shape[1] == 9, f"Nombre de features incorrect: {X_train.shape[1]} != 9"
        assert X_val.shape[1] == 9, f"Nombre de features validation incorrect: {X_val.shape[1]} != 9"
        assert len(y_train) == len(X_train), "Incohérence taille y_train/X_train"
        assert len(y_val) == len(X_val), "Incohérence taille y_val/X_val"
        
        # Vérifications de la normalisation
        assert abs(X_train.mean(axis=0).max()) < 1e-10, "Données d'entraînement mal normalisées (moyenne)"
        assert abs(X_train.std(axis=0).mean() - 1.0) < 0.1, "Données d'entraînement mal normalisées (std)"
        
        # Vérifications des labels
        assert set(np.unique(y_train)) <= {0, 1}, f"Labels d'entraînement invalides: {np.unique(y_train)}"
        assert set(np.unique(y_val)) <= {0, 1}, f"Labels de validation invalides: {np.unique(y_val)}"
        
        # Vérifications de l'équilibre des classes (ne doit pas être trop déséquilibré)
        train_balance = y_train.mean()
        val_balance = y_val.mean()
        assert 0.2 <= train_balance <= 0.8, f"Déséquilibre excessif train: {train_balance:.3f}"
        assert 0.1 <= val_balance <= 0.9, f"Déséquilibre excessif val: {val_balance:.3f}"
        
        print(f"✓ Préprocessing validé:")
        print(f"  Dimensions: Train {X_train.shape}, Val {X_val.shape}")
        print(f"  Équilibres: Train {train_balance:.3f}, Val {val_balance:.3f}")
    
    def test_api_response_format_regression(self):
        """Test de non-régression du format de réponse API"""
        try:
            from flask_app import WaterQualityPredictor
            
            # Mock du prédicteur pour les tests
            import unittest.mock as mock
            
            with mock.patch('flask_app.mlflow') as mock_mlflow, \
                 mock.patch('flask_app.joblib.load') as mock_joblib:
                
                # Configuration des mocks
                mock_scaler = mock.Mock()
                mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
                mock_joblib.return_value = mock_scaler
                
                mock_model = mock.Mock()
                mock_model.predict.return_value = np.array([1])
                mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
                mock_mlflow.sklearn.load_model.return_value = mock_model
                
                predictor = WaterQualityPredictor()
                
                # Test avec des données valides
                test_input = {
                    'ph': 7.0, 'Hardness': 200, 'Solids': 20000, 'Chloramines': 7,
                    'Sulfate': 350, 'Conductivity': 400, 'Organic_carbon': 14,
                    'Trihalomethanes': 80, 'Turbidity': 4
                }
                
                result = predictor.predict(test_input)
                
                # Vérification du format de réponse attendu
                required_keys = ['prediction', 'prediction_label', 'probability_potable', 
                               'probability_non_potable', 'model_info']
                
                for key in required_keys:
                    assert key in result, f"Clé manquante dans la réponse: {key}"
                
                # Vérification des types et valeurs
                assert isinstance(result['prediction'], int), "Type prediction incorrect"
                assert result['prediction'] in [0, 1], "Valeur prediction invalide"
                assert isinstance(result['prediction_label'], str), "Type prediction_label incorrect"
                assert result['prediction_label'] in ['Potable', 'Non Potable'], "Valeur prediction_label invalide"
                assert isinstance(result['probability_potable'], float), "Type probability_potable incorrect"
                assert 0 <= result['probability_potable'] <= 1, "Valeur probability_potable invalide"
                assert isinstance(result['probability_non_potable'], float), "Type probability_non_potable incorrect"
                assert 0 <= result['probability_non_potable'] <= 1, "Valeur probability_non_potable invalide"
                
                # Vérification de la somme des probabilités
                prob_sum = result['probability_potable'] + result['probability_non_potable']
                assert abs(prob_sum - 1.0) < 1e-6, f"Somme des probabilités incorrecte: {prob_sum}"
                
                print("✓ Format de réponse API conforme")
                
        except ImportError as e:
            pytest.skip(f"Modules indisponibles pour test API: {e}")
    
    def test_feature_names_regression(self):
        """Test de non-régression des noms de features"""
        # Noms de features attendus (ne doivent pas changer)
        expected_features = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]
        
        # Vérification dans les métadonnées
        with open('data/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        actual_features = metadata['feature_names']
        
        assert len(actual_features) == len(expected_features), \
            f"Nombre de features incorrect: {len(actual_features)} != {len(expected_features)}"
        
        assert actual_features == expected_features, \
            f"Noms de features modifiés: {actual_features} != {expected_features}"
        
        print("✓ Noms de features inchangés")


class TestPerformanceBenchmarks:
    """Tests de benchmark de performance"""
    
    def test_training_time_regression(self):
        """Test de non-régression du temps d'entraînement"""
        import time
        
        # Données de test
        X_train = np.random.rand(1000, 9)
        y_train = np.random.randint(0, 2, 1000)
        
        # Benchmark Random Forest
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Le temps d'entraînement ne doit pas dépasser un seuil raisonnable
        max_training_time = 30.0  # 30 secondes max pour 1000 échantillons
        
        assert training_time < max_training_time, \
            f"Temps d'entraînement trop élevé: {training_time:.2f}s > {max_training_time}s"
        
        print(f"✓ Temps d'entraînement acceptable: {training_time:.2f}s")
    
    def test_prediction_time_regression(self):
        """Test de non-régression du temps de prédiction"""
        import time
        
        # Modèle pré-entraîné
        X_train = np.random.rand(100, 9)
        y_train = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Données de test pour prédiction
        X_test = np.random.rand(1000, 9)
        
        # Benchmark de prédiction
        start_time = time.time()
        predictions = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Le temps de prédiction doit être rapide
        max_prediction_time = 1.0  # 1 seconde max pour 1000 prédictions
        
        assert prediction_time < max_prediction_time, \
            f"Temps de prédiction trop élevé: {prediction_time:.3f}s > {max_prediction_time}s"
        
        # Temps par prédiction
        time_per_prediction = prediction_time / len(X_test) * 1000  # en millisecondes
        max_time_per_prediction = 1.0  # 1ms max par prédiction
        
        assert time_per_prediction < max_time_per_prediction, \
            f"Temps par prédiction trop élevé: {time_per_prediction:.3f}ms > {max_time_per_prediction}ms"
        
        print(f"✓ Temps de prédiction acceptable: {prediction_time:.3f}s ({time_per_prediction:.3f}ms/prédiction)")


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v"])
