#!/usr/bin/env python3
"""
Génération de données d'exemple pour les tests
Ce script crée des données synthétiques pour permettre l'exécution des tests
"""

import numpy as np
import pandas as pd
import os

def generate_sample_data():
    """Génère des données d'exemple pour les tests"""
    
    # Paramètres
    n_samples = 1000
    n_features = 9
    
    # Génération des features
    np.random.seed(42)  # Pour la reproductibilité des tests
    
    # Simulation de paramètres de qualité d'eau
    ph = np.random.normal(7.0, 1.0, n_samples)
    hardness = np.random.normal(180, 50, n_samples)
    solids = np.random.normal(15000, 5000, n_samples)
    chloramines = np.random.normal(7, 2, n_samples)
    sulfate = np.random.normal(250, 100, n_samples)
    conductivity = np.random.normal(400, 100, n_samples)
    organic_carbon = np.random.normal(12, 4, n_samples)
    trihalomethanes = np.random.normal(70, 30, n_samples)
    turbidity = np.random.normal(4, 2, n_samples)
    
    # Combinaison des features
    X = np.column_stack([
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    ])
    
    # Génération de labels basés sur des règles simples
    # Eau potable si la plupart des paramètres sont dans les bonnes plages
    potable_conditions = (
        (ph >= 6.5) & (ph <= 8.5) &
        (hardness <= 300) &
        (solids <= 20000) &
        (chloramines <= 10) &
        (sulfate <= 400) &
        (turbidity <= 5)
    )
    
    y = potable_conditions.astype(int)
    
    # Ajout de bruit pour rendre la tâche plus réaliste
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Division train/validation
    split_idx = int(0.8 * n_samples)
    
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    return X_train, X_val, y_train, y_val

def save_data():
    """Sauvegarde les données générées"""
    X_train, X_val, y_train, y_val = generate_sample_data()
    
    # Création du dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    
    # Sauvegarde
    np.save('data/X_train.npy', X_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_val.npy', y_val)
    
    # Création d'un DataFrame pour visualisation
    feature_names = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    
    # Dataset complet
    X_complete = np.vstack([X_train, X_val])
    y_complete = np.hstack([y_train, y_val])
    
    df = pd.DataFrame(X_complete, columns=feature_names)
    df['Potability'] = y_complete
    df.to_csv('data/water_potability.csv', index=False)
    
    print(f"Données générées avec succès:")
    print(f"- Training set: {X_train.shape[0]} échantillons")
    print(f"- Validation set: {X_val.shape[0]} échantillons")
    print(f"- Features: {X_train.shape[1]}")
    print(f"- Distribution des classes (train): {np.bincount(y_train)}")
    print(f"- Distribution des classes (val): {np.bincount(y_val)}")

if __name__ == "__main__":
    save_data()
