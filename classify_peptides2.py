import pandas as pd
import ollama
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from datetime import datetime
import json


def load_data():
    df = pd.read_excel('data/CPP_Classif_data.xlsx')
    return df 


def ensure_model(model_name='llama3.2:1b'):
    try:
        ollama.show(model_name)
    except Exception as e:
        print(f'Error: Model "{model_name}" no alojado en Ollama. ')


def get_balanced_few_shot_examples(df, max_per_class):
    """
    Selecciona balanceadamente max_per_class ejemplos por clase (0 y 1) de forma aleatoria.
    """
    examples = []
    for cls in df['y'].unique():
        samples = df[df['y'] == cls].sample(n=min(max_per_class, len(df[df['y'] == cls])), random_state=42)
        examples.extend(zip(samples['peptide'], samples['y']))
    return examples


def generate_few_shot_prompt(sequence, examples):
    example_prompt = ""
    for seq, label in examples:
        example_prompt += f"Sequence: {seq}\nIs it a cell-penetrating peptide? {label}\n\n"

    final_prompt = (
        "You are an expert in peptide classification. "
        "Below are classified examples:\n\n"
        f"{example_prompt}"
        f"Now analyze the following sequence:\nSequence: {sequence}\nIs it a cell-penetrating peptide? Reply with 1 for yes, 0 for no.\n"
        "Return only the numeric value without any explanation."
    )

    return final_prompt


def classify_peptide_with_context(sequence, examples, model_name='gemma3:1b'):
    prompt = generate_few_shot_prompt(sequence, examples)
    print("Modelo :", model_name)
    
    try:
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}])
        content = response['message']['content'].strip().replace('.', '')
        return int(content)
    except Exception as e:
        print(f'Error classifying {sequence}: {str(e)}')
        return np.nan


def evaluate_model(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    y_true_clean = y_true[mask].astype(int)
    y_pred_clean = y_pred[mask].astype(int)

    accuracy = accuracy_score(y_true_clean, y_pred_clean)
    precision = precision_score(y_true_clean, y_pred_clean)
    recall = recall_score(y_true_clean, y_pred_clean)
    f1 = f1_score(y_true_clean, y_pred_clean)
    conf_matrix = confusion_matrix(y_true_clean, y_pred_clean).tolist()
    class_report = classification_report(y_true_clean, y_pred_clean, output_dict=True)

    print("\n--- Métricas de Evaluación ---")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    print("\nReporte de Clasificación:")
    print(classification_report(y_true_clean, y_pred_clean))
    print(f"\nExactitud (accuracy): {accuracy:.4f}")

    metrics = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }
    json_path = f"data/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Métricas guardadas en {json_path}")

if __name__ == '__main__':
    df = load_data()
    ensure_model()

    # Dividir en entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=42)

    # Calcular número máximo de ejemplos por clase
    class_counts = train_df['y'].value_counts()
    max_per_class = class_counts.min()
    
    # Obtener ejemplos balanceados para few-shot
    few_shot_examples = get_balanced_few_shot_examples(train_df, max_per_class=max_per_class)

    print('Ejemplos usados para contexto few-shot:')
    for seq, label in few_shot_examples:
        print(f' - Clase {label}: {seq}')

    print('\nClasificando péptidos del conjunto de prueba...')

    # Clasificar con contexto few-shot
    test_df['predicted_class'] = test_df['peptide'].apply(
        lambda seq: classify_peptide_with_context(seq, few_shot_examples, model_name='qwen2.5:3b')
    )

    evaluate_model(test_df['y'], test_df['predicted_class'])

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.concat([train_df.assign(predicted_class=np.nan), test_df])
    output_path = f'data/classification_results_{timestamp}.xlsx'
    results_df.to_excel(output_path, index=False)
    print(f'Clasificación terminada. Resultados guardados en {output_path}')
