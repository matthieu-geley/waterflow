#!/usr/bin/env python3
"""
Script pour convertir les fichiers .npy en .pkl pour la compatibilité avec experiment.py
"""

import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler

def convert_data():
    """Convertit les données .npy en .pkl et crée le scaler"""
    
    # Chargement des données .npy
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    
    print(f"Données chargées :")
    print(f"- X_train: {X_train.shape}")
    print(f"- X_val: {X_val.shape}")
    print(f"- y_train: {y_train.shape}")
    print(f"- y_val: {y_val.shape}")
    
    # Création et ajustement du scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Sauvegarde en format .pkl
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(X_val, 'X_val.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(y_val, 'y_val.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Création des métadonnées
    feature_names = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'classes': [0, 1],
        'target_names': ['Non Potable', 'Potable']
    }
    
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Conversion terminée :")
    print("- Fichiers .pkl créés")
    print("- Scaler créé et ajusté")
    print("- Métadonnées sauvegardées")
    
    return True

if __name__ == "__main__":
    convert_data()
