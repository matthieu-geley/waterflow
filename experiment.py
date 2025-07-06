#!/usr/bin/env python3
"""
Script d'expérimentation MLflow pour le projet Drink Safe
Projet Waterflow - Prédiction de Qualité d'Eau

Ce script implémente toutes les étapes d'utilisation de MLflow :
1. Configuration du serveur de tracking
2. Création/récupération d'expériences
3. Logging des paramètres et métriques
4. Entraînement de modèles multiples
5. Comparaison des performances
6. Enregistrement dans le Model Registry
7. Transition des modèles vers la production

Utilisation:
    # 1. Démarrer le serveur MLflow
    mlflow server --host 127.0.0.1 --port 5000
    
    # 2. Exécuter les expériences
    python experiment.py

Auteur: Équipe Drink Safe
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Suppression des warnings pour un output plus propre
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DrinkSafeMLflowExperiment:
    """
    Classe principale pour gérer les expériences MLflow du projet Drink Safe
    
    Cette classe encapsule toute la logique d'expérimentation MLflow :
    - Configuration et connexion au serveur
    - Gestion des expériences
    - Entraînement et évaluation des modèles
    - Logging des artefacts et métriques
    - Gestion du Model Registry
    """
    
    def __init__(self, 
                 tracking_uri: str = "http://127.0.0.1:5000",
                 experiment_name: str = "experiment_water_quality"):
        """
        Initialise l'expérimentateur MLflow
        
        Args:
            tracking_uri (str): URI du serveur MLflow
            experiment_name (str): Nom de l'expérience MLflow
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
        self.models_results = {}
        
        # Données d'entraînement (chargées depuis le preprocessing)
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.scaler = None
        self.feature_names = None
        
        # Configuration initiale
        self._setup_mlflow()
        self._load_preprocessed_data()
    
    def _setup_mlflow(self) -> None:
        """
        Configure la connexion MLflow et initialise l'expérience
        
        Étapes:
        1. Configuration de l'URI de tracking
        2. Test de connexion au serveur
        3. Création/récupération de l'expérience
        4. Initialisation du client MLflow
        """
        logger.info("Configuration de MLflow...")
        
        try:
            # Configuration de l'URI de tracking
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"URI de tracking configuré : {self.tracking_uri}")
            
            # Test de connexion
            experiments = mlflow.search_experiments()
            logger.info(f"Connexion réussie - {len(experiments)} expériences trouvées")
            
            # Initialisation du client
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            
            # Configuration de l'expérience
            self._setup_experiment()
            
        except Exception as e:
            logger.error(f"Erreur de configuration MLflow : {e}")
            logger.error("Vérifiez que le serveur MLflow est démarré :")
            logger.error("   mlflow server --host 127.0.0.1 --port 5000")
            raise
    
    def _setup_experiment(self) -> None:
        """
        Configure l'expérience MLflow
        
        Crée l'expérience si elle n'existe pas, sinon la récupère
        """
        try:
            # Tentative de récupération de l'expérience existante
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                # Création d'une nouvelle expérience
                self.experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    tags={
                        "project": "drink_safe",
                        "version": "1.0",
                        "description": "Expérience de prédiction de qualité d'eau",
                        "created_by": "waterflow_team"
                    }
                )
                logger.info(f"🆕 Expérience créée : {self.experiment_name} (ID: {self.experiment_id})")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Expérience récupérée : {self.experiment_name} (ID: {self.experiment_id})")
            
            # Activation de l'expérience
            mlflow.set_experiment(experiment_id=self.experiment_id)
            
        except Exception as e:
            logger.error(f"Erreur configuration expérience : {e}")
            raise
    
    def _load_preprocessed_data(self) -> None:
        """
        Charge les données prétraitées depuis le dossier data/
        
        Charge :
        - Ensembles d'entraînement et de validation
        - Scaler de normalisation
        - Métadonnées (noms des features, etc.)
        """
        logger.info("📂 Chargement des données prétraitées...")
        
        data_dir = "data"
        
        try:
            # Vérification de l'existence du dossier
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    f"Dossier '{data_dir}' non trouvé. "
                    "Exécutez d'abord le notebook main.ipynb pour préparer les données."
                )
            
            # Chargement des données d'entraînement et de validation
            self.X_train = joblib.load(f"{data_dir}/X_train.pkl")
            self.X_val = joblib.load(f"{data_dir}/X_val.pkl")
            self.y_train = joblib.load(f"{data_dir}/y_train.pkl")
            self.y_val = joblib.load(f"{data_dir}/y_val.pkl")
            self.scaler = joblib.load(f"{data_dir}/scaler.pkl")
            
            # Chargement des métadonnées
            with open(f"{data_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
            
            # Application de la normalisation
            self.X_train_scaled = self.scaler.transform(self.X_train)
            self.X_val_scaled = self.scaler.transform(self.X_val)
            
            logger.info(f"Données chargées :")
            logger.info(f"   - Entraînement : {self.X_train.shape}")
            logger.info(f"   - Validation : {self.X_val.shape}")
            logger.info(f"   - Features : {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Erreur chargement données : {e}")
            raise
    
    def _log_common_parameters(self, run_name: str) -> None:
        """
        Log des paramètres communs à tous les modèles
        
        Args:
            run_name (str): Nom du run MLflow
        """
        mlflow.log_param("experiment_name", self.experiment_name)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("experiment_date", datetime.now().isoformat())
        mlflow.log_param("train_samples", len(self.X_train))
        mlflow.log_param("val_samples", len(self.X_val))
        mlflow.log_param("n_features", len(self.feature_names))
        mlflow.log_param("features", self.feature_names)
        mlflow.log_param("preprocessing", "StandardScaler + median_imputation")
        
        # Tags pour l'organisation
        mlflow.set_tag("project", "drink_safe")
        mlflow.set_tag("stage", "experimentation")
        mlflow.set_tag("model_category", "water_quality_classification")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calcule les métriques d'évaluation
        
        Args:
            y_true (np.ndarray): Vraies étiquettes
            y_pred (np.ndarray): Prédictions
            y_pred_proba (np.ndarray, optional): Probabilités de prédiction
            
        Returns:
            Dict[str, float]: Dictionnaire des métriques
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # ROC-AUC si les probabilités sont disponibles
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                # En cas de problème avec ROC-AUC (ex: une seule classe)
                metrics['roc_auc'] = 0.5
        
        return metrics
    
    def experiment_random_forest(self) -> Tuple[Any, Dict[str, float]]:
        """
        Expérience avec Random Forest et Grid Search
        
        Returns:
            Tuple[Any, Dict[str, float]]: Modèle entraîné et métriques
        """
        logger.info("Expérience Random Forest avec Grid Search...")
        
        with mlflow.start_run(run_name="RandomForest_WaterQuality_GridSearch") as run:
            # Grille de paramètres pour le grid search
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Logging des paramètres
            self._log_common_parameters("RandomForest_WaterQuality_GridSearch")
            mlflow.log_param("algorithm", "Random Forest")
            mlflow.log_param("model_type", "Ensemble Learning")
            mlflow.log_param("grid_search", True)
            mlflow.log_param("param_grid_size", sum(len(v) if isinstance(v, list) else 1 for v in param_grid.values()))
            
            # Grid Search avec validation croisée
            start_time = time.time()
            rf_base = RandomForestClassifier(n_jobs=-1)
            grid_search = GridSearchCV(
                rf_base, 
                param_grid, 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            training_time = time.time() - start_time
            
            # Meilleur modèle
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Logging des meilleurs paramètres
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Prédictions avec le meilleur modèle
            y_pred = best_model.predict(self.X_val_scaled)
            y_pred_proba = best_model.predict_proba(self.X_val_scaled)[:, 1]
            
            # Calcul des métriques
            metrics = self._calculate_metrics(self.y_val, y_pred, y_pred_proba)
            metrics['training_time'] = training_time
            
            # Logging des métriques
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Importance des features
            feature_importance = dict(zip(self.feature_names, best_model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Classification report
            class_report = classification_report(self.y_val, y_pred, output_dict=True)
            mlflow.log_dict(class_report, "classification_report.json")
            
            # Enregistrement du modèle
            mlflow.sklearn.log_model(
                best_model, 
                "model",
                registered_model_name="water_quality_random_forest"
            )
            
            # Stockage des résultats
            results = {
                'model': best_model,
                'metrics': metrics,
                'run_id': run.info.run_id,
                'model_name': 'Random Forest',
                'best_params': best_params
            }
            
            logger.info(f"Random Forest Grid Search terminé:")
            logger.info(f"  - Meilleurs paramètres: {best_params}")
            logger.info(f"  - CV Score: {grid_search.best_score_:.4f}")
            logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  - Temps d'entraînement: {training_time:.2f}s")
            
            return best_model, metrics
            self.models_results['RandomForest'] = {
                'model': model,
                'metrics': metrics,
                'run_id': run.info.run_id,
                'model_uri': f"runs:/{run.info.run_id}/model"
            }
            
            logger.info(f"✅ Random Forest - Accuracy: {metrics['accuracy']:.4f}")
            return model, metrics
    
    def experiment_xgboost(self) -> Optional[Tuple[Any, Dict[str, float]]]:
        """
        Expérience avec XGBoost et Grid Search
        
        Returns:
            Optional[Tuple[Any, Dict[str, float]]]: Modèle entraîné et métriques (si XGBoost disponible)
        """
        try:
            import xgboost as xgb
            logger.info("Expérience XGBoost avec Grid Search...")
            
            with mlflow.start_run(run_name="XGBoost_WaterQuality_GridSearch") as run:
                # Grille de paramètres pour le grid search
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
                
                # Logging des paramètres
                self._log_common_parameters("XGBoost_WaterQuality_GridSearch")
                mlflow.log_param("algorithm", "XGBoost")
                mlflow.log_param("model_type", "Gradient Boosting")
                mlflow.log_param("grid_search", True)
                mlflow.log_param("param_grid_size", sum(len(v) if isinstance(v, list) else 1 for v in param_grid.values()))
                
                # Grid Search avec validation croisée
                start_time = time.time()
                xgb_base = xgb.XGBClassifier(
                    eval_metric='logloss',
                    use_label_encoder=False,
                    n_jobs=-1
                )
                
                grid_search = GridSearchCV(
                    xgb_base, 
                    param_grid, 
                    cv=5, 
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(self.X_train_scaled, self.y_train)
                training_time = time.time() - start_time
                
                # Meilleur modèle
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                # Logging des meilleurs paramètres
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                
                # Prédictions avec le meilleur modèle
                y_pred = best_model.predict(self.X_val_scaled)
                y_pred_proba = best_model.predict_proba(self.X_val_scaled)[:, 1]
                
                # Calcul des métriques
                metrics = self._calculate_metrics(self.y_val, y_pred, y_pred_proba)
                metrics['training_time'] = training_time
                
                # Logging des métriques
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Importance des features
                feature_importance = dict(zip(self.feature_names, best_model.feature_importances_))
                mlflow.log_dict(feature_importance, "feature_importance.json")
                
                # Classification report
                class_report = classification_report(self.y_val, y_pred, output_dict=True)
                mlflow.log_dict(class_report, "classification_report.json")
                
                # Enregistrement du modèle
                mlflow.xgboost.log_model(
                    best_model, 
                    "model",
                    registered_model_name="water_quality_xgboost"
                )
                
                # Stockage des résultats
                self.models_results['XGBoost'] = {
                    'model': best_model,
                    'metrics': metrics,
                    'run_id': run.info.run_id,
                    'model_uri': f"runs:/{run.info.run_id}/model",
                    'best_params': best_params
                }
                
                logger.info(f"XGBoost Grid Search terminé:")
                logger.info(f"  - Meilleurs paramètres: {best_params}")
                logger.info(f"  - CV Score: {grid_search.best_score_:.4f}")
                logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  - Temps d'entraînement: {training_time:.2f}s")
                
                return best_model, metrics
                
        except ImportError:
            logger.warning("XGBoost non installé - expérience ignorée")
            return None
        except Exception as e:
            logger.error(f"Erreur XGBoost : {e}")
            return None
    
    def experiment_mlp(self) -> Tuple[Any, Dict[str, float]]:
        """
        Expérience avec Perceptron Multicouches (MLP) et Grid Search
        
        Returns:
            Tuple[Any, Dict[str, float]]: Modèle entraîné et métriques
        """
        logger.info("Expérience Perceptron Multicouches avec Grid Search...")
        
        with mlflow.start_run(run_name="MLP_WaterQuality_GridSearch") as run:
            # Grille de paramètres pour le grid search
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50), (128, 64, 32)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
            
            # Logging des paramètres
            self._log_common_parameters("MLP_WaterQuality_GridSearch")
            mlflow.log_param("algorithm", "Multi-Layer Perceptron")
            mlflow.log_param("model_type", "Neural Network")
            mlflow.log_param("grid_search", True)
            mlflow.log_param("param_grid_size", sum(len(v) if isinstance(v, list) else 1 for v in param_grid.values()))
            
            # Grid Search avec validation croisée
            start_time = time.time()
            mlp_base = MLPClassifier(
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20
            )
            
            grid_search = GridSearchCV(
                mlp_base, 
                param_grid, 
                cv=3,  # Réduction du CV pour les réseaux de neurones (plus lent)
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            training_time = time.time() - start_time
            
            # Meilleur modèle
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Logging des meilleurs paramètres
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Prédictions avec le meilleur modèle
            y_pred = best_model.predict(self.X_val_scaled)
            y_pred_proba = best_model.predict_proba(self.X_val_scaled)[:, 1]
            
            # Calcul des métriques
            metrics = self._calculate_metrics(self.y_val, y_pred, y_pred_proba)
            metrics['training_time'] = training_time
            metrics['n_iterations'] = best_model.n_iter_
            
            # Logging des métriques
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Classification report
            class_report = classification_report(self.y_val, y_pred, output_dict=True)
            mlflow.log_dict(class_report, "classification_report.json")
            
            # Enregistrement du modèle
            mlflow.sklearn.log_model(
                best_model, 
                "model",
                registered_model_name="water_quality_mlp"
            )
            
            # Stockage des résultats
            self.models_results['MLP'] = {
                'model': best_model,
                'metrics': metrics,
                'run_id': run.info.run_id,
                'model_uri': f"runs:/{run.info.run_id}/model",
                'best_params': best_params
            }
            
            logger.info(f"MLP Grid Search terminé:")
            logger.info(f"  - Meilleurs paramètres: {best_params}")
            logger.info(f"  - CV Score: {grid_search.best_score_:.4f}")
            logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  - Temps d'entraînement: {training_time:.2f}s")
            logger.info(f"  - Itérations: {best_model.n_iter_}")
            
            return best_model, metrics
    
    def compare_models(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Compare les performances des modèles et identifie le meilleur
        
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: Nom et données du meilleur modèle
        """
        if not self.models_results:
            logger.warning("Aucun modèle à comparer")
            return None
        
        logger.info("\nCOMPARAISON DES MODÈLES")
        logger.info("=" * 50)
        
        # Création du tableau de comparaison
        comparison_data = []
        for model_name, results in self.models_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Modèle': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', 'N/A'),
                'Temps (s)': metrics['training_time']
            })
        
        # Tri par accuracy décroissante
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Affichage du tableau
        print("\n" + comparison_df.round(4).to_string(index=False))
        
        # Identification du meilleur modèle
        best_model_name = comparison_df.iloc[0]['Modèle']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        
        logger.info(f"\nMEILLEUR MODÈLE : {best_model_name}")
        logger.info(f"Accuracy : {best_accuracy:.4f}")
        
        # Logging de la comparaison dans MLflow
        with mlflow.start_run(run_name="Models_Comparison_Summary"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_accuracy", best_accuracy)
            mlflow.log_metric("models_compared", len(self.models_results))
            
            # Sauvegarde du tableau de comparaison
            comparison_df.to_csv("models_comparison.csv", index=False)
            mlflow.log_artifact("models_comparison.csv")
            os.remove("models_comparison.csv")  # Nettoyage
            
            mlflow.set_tag("analysis", "model_comparison")
            mlflow.set_tag("best_model", best_model_name)
        
        return best_model_name, self.models_results[best_model_name]
    
    def promote_best_model_to_production(self, best_model_name: str) -> None:
        """
        Promeut le meilleur modèle vers la production dans le Model Registry
        
        Args:
            best_model_name (str): Nom du meilleur modèle
        """
        try:
            logger.info(f"Promotion du modèle {best_model_name} vers la production...")
            
            # Nom du modèle dans le registry
            registry_model_name = f"water_quality_{best_model_name.lower()}"
            
            # Récupération de la dernière version
            latest_versions = self.client.get_latest_versions(
                registry_model_name, 
                stages=["None", "Staging"]
            )
            
            if latest_versions:
                latest_version = latest_versions[0]
                
                # Transition vers Production
                self.client.transition_model_version_stage(
                    name=registry_model_name,
                    version=latest_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                
                logger.info(f"Modèle {registry_model_name} v{latest_version.version} promu en Production")
            else:
                logger.warning(f"Aucune version trouvée pour {registry_model_name}")
                
        except Exception as e:
            logger.error(f"Erreur promotion modèle : {e}")
    
    def run_full_experiment(self) -> None:
        """
        Exécute l'ensemble des expériences MLflow
        
        Séquence complète :
        1. Expérience Random Forest
        2. Expérience XGBoost (si disponible)
        3. Expérience MLP
        4. Comparaison des modèles
        5. Promotion du meilleur modèle
        """
        logger.info("DÉMARRAGE DES EXPÉRIENCES MLFLOW")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Expérience 1 : Random Forest
            self.experiment_random_forest()
            
            # Expérience 2 : XGBoost (si disponible)
            xgb_result = self.experiment_xgboost()
            if xgb_result is None:
                logger.info("XGBoost ignoré (non disponible)")
            
            # Expérience 3 : MLP
            self.experiment_mlp()
            
            # Comparaison des modèles
            comparison_result = self.compare_models()
            
            if comparison_result:
                best_model_name, best_model_data = comparison_result
                
                # Promotion du meilleur modèle
                self.promote_best_model_to_production(best_model_name)
                
                total_time = time.time() - start_time
                
                logger.info(f"\nEXPÉRIENCES TERMINÉES AVEC SUCCÈS !")
                logger.info(f"Temps total : {total_time:.2f} secondes")
                logger.info(f"Meilleur modèle : {best_model_name}")
                logger.info(f"Accuracy : {best_model_data['metrics']['accuracy']:.4f}")
                logger.info(f"Interface MLflow : {self.tracking_uri}")
                logger.info(f"Prêt pour déploiement avec app.py")
                
            else:
                logger.error("Aucun modèle valide produit")
                
        except Exception as e:
            logger.error(f"Erreur lors des expériences : {e}")
            raise

def main():
    """
    Fonction principale pour exécuter les expériences MLflow
    """
    print("DRINK SAFE - EXPÉRIENCES MLFLOW")
    print("=" * 40)
    print("Projet Waterflow - Prédiction de Qualité d'Eau")
    print()
    
    try:
        # Vérification des prérequis
        if not os.path.exists('data'):
            print("❌ Erreur : Dossier 'data' non trouvé")
            print("Exécutez d'abord le notebook main.ipynb pour préparer les données")
            return 1
        
        # Initialisation de l'expérimentateur
        experimenter = DrinkSafeMLflowExperiment()
        
        # Exécution des expériences
        experimenter.run_full_experiment()
        
        print("\n✅ SUCCESS : Toutes les expériences ont été exécutées avec succès !")
        print("Consultez l'interface MLflow : http://127.0.0.1:5000")
        print("Lancez l'application : python app.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        print("\nVérifications à effectuer :")
        print("1. Serveur MLflow démarré : mlflow server --host 127.0.0.1 --port 5000")
        print("2. Données préparées : exécution du notebook main.ipynb")
        print("3. Dépendances installées : pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)