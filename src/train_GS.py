import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import ParameterGrid # Importar ParameterGrid

# Scripts necessários para treinar o modelo
from data_loader import load_datasets
from model_architecture import create_model

# ===================================================================
# DEFINIÇÃO DA CONFIGURAÇÃO BASE E DA GRELHA DE HIPERPARÂMETROS
# ===================================================================
# Configuração base que não será alterada pela pesquisa em grelha
base_config = {
    "base_data_dir": "C:\\Users\\Programmer\\processed",
    "experiments_base_dir": "C:\\Users\\Miguel António\\Desktop\\PORTFOLIO\\image_classifier\\experiments",
    "img_size": (224, 224),
    "epochs": 20, # Aumentar o número de épocas pode ser útil, EarlyStopping irá parar se não houver melhoria
    "early_stopping_patience": 5,
    "model_architecture_name": "Minha_CNN_Base" 
}

# --- A GRELHA DE PARÂMETROS PARA A PESQUISA ---
# Defina aqui os hiperparâmetros que deseja testar.
# O script irá iterar sobre todas as combinações possíveis.
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [16, 32],
    'optimizer': ['adam', 'rmsprop'] 
    # Pode adicionar outros parâmetros aqui, como 'dropout_rate', se o seu create_model() o suportar.
}

# ===================================================================
# CONFIGURAÇÃO DO EXPERIMENTO DE GRID SEARCH
# ===================================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
grid_search_dir = os.path.join(base_config["experiments_base_dir"], f"grid_search_{timestamp}")
os.makedirs(grid_search_dir, exist_ok=True)
print(f"Iniciando Grid Search - Resultados serão salvos em: {grid_search_dir}\n")

# Variáveis para acompanhar o melhor desempenho
best_accuracy = 0.0
best_params = None
best_run_dir = ""

# Gerar todas as combinações de parâmetros
grid = ParameterGrid(param_grid)

# ===================================================================
# LOOP PRINCIPAL DA PESQUISA EM GRELHA
# ===================================================================
for i, params in enumerate(grid):
    run_name = f"run_{i+1}"
    experiment_dir = os.path.join(grid_search_dir, run_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # --- Combina a configuração base com os parâmetros atuais da grelha ---
    config = {**base_config, **params}
    
    print("="*50)
    print(f"Iniciando {run_name} com os parâmetros:")
    print(json.dumps(config, indent=4))
    print("="*50)

    # Salva os hiperparâmetros desta execução
    params_path = os.path.join(experiment_dir, "params.json")
    with open(params_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # ===================================================================
    # CARREGAMENTO DOS DADOS (dentro do loop para usar o batch_size correto)
    # ===================================================================
    train_ds, val_ds, test_ds, class_names = load_datasets(
        base_dir=config["base_data_dir"],
        img_size=config["img_size"],
        batch_size=config["batch_size"]
    )
    
    # ===================================================================
    # CONSTRUÇÃO E COMPILAÇÃO DO MODELO
    # ===================================================================
    input_shape = (*config["img_size"], 3)
    num_classes = len(class_names)

    # Recria o modelo do zero para cada execução
    model = create_model(input_shape=input_shape, num_classes=num_classes)

    # Seleciona o otimizador com base no parâmetro da grelha
    if config['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    elif config['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config["learning_rate"])
    else:
        # Adicione outros otimizadores se necessário
        raise ValueError(f"Otimizador '{config['optimizer']}' não é suportado.")

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # model.summary() # Descomente se quiser ver o resumo do modelo em cada execução

    # ===================================================================
    # CONFIGURAÇÃO DE CALLBACKS E TREINO
    # ===================================================================
    model_checkpoint_path = os.path.join(experiment_dir, "best_model.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=0) # Verbose=0 para um log mais limpo
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=config["early_stopping_patience"], verbose=1, restore_best_weights=True)

    print(f"\nIniciando treino para {run_name} por {config['epochs']} épocas...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=config["epochs"], 
        callbacks=[checkpoint_callback, early_stopping_callback],
        verbose=2 # Verbose=2 mostra uma linha por época
    )
    
    # ===================================================================
    # AVALIAÇÃO E VISUALIZAÇÃO
    # ===================================================================
    print(f"\nRealizando avaliação final para {run_name} no dataset de teste...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Resultado de {run_name} -> Perda: {loss:.4f}, Acurácia: {accuracy:.4f}\n")

    # Guardar a melhor acurácia para encontrar o melhor modelo
    if accuracy > best_accuracy:
        print(f"!!! Nova melhor acurácia encontrada: {accuracy:.4f} (anterior: {best_accuracy:.4f}) !!!")
        best_accuracy = accuracy
        best_params = config
        best_run_dir = experiment_dir
        
    # Salvar gráficos de treino
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss per epoch')
    plt.legend()
    
    plot_path = os.path.join(experiment_dir, "training_plots.png")
    plt.savefig(plot_path)
    plt.close() # Fecha a figura para não ser exibida no final

    # Salvar métricas de avaliação
    metrics = {"test_loss": loss, "test_accuracy": accuracy}
    metrics_path = os.path.join(experiment_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

# ===================================================================
# RESUMO FINAL DO GRID SEARCH
# ===================================================================
print("\n" + "="*50)
print("PESQUISA EM GRELHA CONCLUÍDA")
print("="*50)
print(f"A melhor acurácia no teste foi: {best_accuracy:.4f}")
print("Obtida com os seguintes hiperparâmetros:")
print(json.dumps(best_params, indent=4))
print(f"O melhor modelo e os logs estão salvos em: {best_run_dir}")

# Salvar o resumo do melhor resultado na pasta principal do Grid Search
summary_path = os.path.join(grid_search_dir, "best_params_summary.json")
with open(summary_path, 'w') as f:
    summary_data = {
        "best_test_accuracy": best_accuracy,
        "best_hyperparameters": best_params,
        "best_model_path": os.path.join(best_run_dir, "best_model.keras")
    }
    json.dump(summary_data, f, indent=4)