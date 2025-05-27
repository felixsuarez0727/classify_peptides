import pandas as pd
import ollama
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score


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
        example_prompt += f"Secuencia: {seq}\n¿Es un péptido de penetración celular? {label}\n\n"

    final_prompt = (
        "Eres un experto en clasificación de péptidos. "
        "A continuación, se presentan ejemplos clasificados:\n\n"
        f"{example_prompt}"
        f"Ahora analice la siguiente secuencia:\nSecuencia: {sequence}\n¿Es un péptido de penetración celular? Responda con 1 para sí, 0 para no.\n"
        "Devuelva solo el valor numérico sin ninguna explicación."
    )

    return final_prompt


def classify_peptide_with_context(sequence, examples, model_name='gemma3:1b'):
    prompt = generate_few_shot_prompt(sequence, examples)
    
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

    print("\n--- Métricas de Evaluación ---")
    print("Accuracy: ", accuracy_score(y_true_clean, y_pred_clean))
    print("Precision: ", precision_score(y_true_clean, y_pred_clean))
    print("Recall: ", recall_score(y_true_clean, y_pred_clean))
    print("F1-score: ", f1_score(y_true_clean, y_pred_clean))
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_true_clean, y_pred_clean))
    print("\nReporte de Clasificación:")
    print(classification_report(y_true_clean, y_pred_clean))
    print(f"\nExactitud (accuracy): {accuracy_score(y_true_clean, y_pred_clean):.4f}")
   

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
        lambda seq: classify_peptide_with_context(seq, few_shot_examples)
    )

    evaluate_model(test_df['y'], test_df['predicted_class'])

    # Guardar resultados
    results_df = pd.concat([train_df.assign(predicted_class=np.nan), test_df])
    results_df.to_excel('data/classification_results.xlsx', index=False)
    print('Clasificación terminada. Resultados guardados en data/classification_results.xlsx')
