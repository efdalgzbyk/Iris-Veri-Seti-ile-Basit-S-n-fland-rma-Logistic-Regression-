import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix'):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    # 1. Veri setini yükle
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 2. DataFrame oluştur
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = pd.Categorical.from_codes(y, target_names)

    print("İlk 5 kayıt:")
    print(df.head())

    # 3. Veri dağılımını görselleştir
    sns.pairplot(df, hue='species')
    plt.suptitle("Iris Veri Seti Özellik Dağılımı", y=1.02)
    plt.savefig("iris_pairplot.png")
    plt.show()

    # 4. Eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. GridSearchCV ile hiperparametre arama
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print(f"\nEn iyi parametreler: {grid.best_params_}")
    print(f"En iyi doğruluk (CV): {grid.best_score_:.4f}")

    # 6. En iyi modeli kullanarak test setinde tahmin yap
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # 7. Modeli değerlendir
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Seti Doğruluk Oranı: {acc:.4f}\n")
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 8. Confusion matrix çiz ve kaydet
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, target_names)

    # 9. Modeli kaydet
    joblib.dump(best_model, 'iris_logistic_model_optimized.pkl')
    print("\nOptimizasyon sonrası model 'iris_logistic_model_optimized.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    main()
