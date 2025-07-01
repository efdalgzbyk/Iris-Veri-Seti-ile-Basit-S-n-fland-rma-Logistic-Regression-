
# Logistic Regression Classification with Iris Dataset

This project aims to classify flower species using machine learning techniques on the famous Iris dataset. It is implemented in Python using popular libraries such as scikit-learn, pandas, seaborn, and matplotlib.

### Project Overview:

- **Data Loading and Preprocessing:**  
  The Iris dataset is loaded from scikit-learn and converted into a pandas DataFrame. Features and target classes are properly labeled.

- **Data Visualization:**  
  Using seaborn's pairplot, the distribution of features across different species is visualized for exploratory data analysis.

- **Train-Test Split:**  
  The dataset is randomly split into 80% training and 20% testing sets. This helps in evaluating the model's real-world performance.

- **Model Training and Hyperparameter Optimization:**  
  A logistic regression model is created, and hyperparameters are tuned using GridSearchCV with cross-validation to find the best model settings.

- **Model Evaluation:**  
  Accuracy on the test set is calculated and a classification report provides detailed metrics like precision, recall, and f1-score.

- **Confusion Matrix:**  
  A confusion matrix is plotted to visualize how the model performs in classifying each class correctly or incorrectly.

- **Model Saving:**  
  The trained and optimized model is saved to a `.pkl` file for later use, enabling easy loading and prediction without retraining.

### Project Purpose:

This project offers a practical introduction to machine learning workflows, covering essential concepts such as data analysis, model building, hyperparameter tuning, evaluation, and visualization. The Iris dataset is a classic and widely used example for beginners in machine learning.

### Requirements:

- Python 3.x  
- scikit-learn  
- pandas  
- seaborn  
- matplotlib  
- joblib  

### Usage:

Run the project by executing the following command in the terminal:

```bash
python iris_classification.py
```



# Iris Veri Seti ile Lojistik Regresyon Sınıflandırması

Bu proje, ünlü Iris çiçek veri seti üzerinde makine öğrenmesi tekniklerini kullanarak çiçek türlerini sınıflandırmayı amaçlamaktadır. Proje, Python programlama dili ve popüler kütüphaneler olan scikit-learn, pandas, seaborn ve matplotlib kullanılarak geliştirilmiştir.

### Proje İçeriği:

- **Veri Yükleme ve Ön İşleme:**  
  `scikit-learn` kütüphanesinden Iris veri seti yüklenir ve pandas DataFrame yapısına dönüştürülür. Veri setindeki özellikler ve hedef sınıflar etiketlenir.

- **Veri Görselleştirme:**  
  Seaborn kütüphanesi ile veri setindeki özelliklerin sınıflara göre dağılımı `pairplot` grafiği ile görselleştirilir ve analiz edilir.

- **Eğitim ve Test Setine Bölme:**  
  Veri, %80 eğitim ve %20 test olacak şekilde rastgele ikiye ayrılır. Bu, modelin gerçek dünya performansını değerlendirmek için önemlidir.

- **Model Eğitimi ve Hiperparametre Optimizasyonu:**  
  Lojistik regresyon modeli oluşturulur ve hiperparametreler `GridSearchCV` kullanılarak çapraz doğrulama ile optimize edilir. Bu sayede modelin en iyi ayarları bulunur.

- **Model Değerlendirme:**  
  Test seti üzerinde modelin doğruluğu hesaplanır, sınıflandırma raporu ile detaylı performans metrikleri (precision, recall, f1-score) sunulur.

- **Confusion Matrix (Karışıklık Matrisi):**  
  Modelin hangi sınıfları nasıl doğru ya da yanlış tahmin ettiğini gösteren confusion matrix görselleştirilir ve kaydedilir.

- **Model Kaydetme:**  
  Eğitilen ve optimize edilen model `.pkl` dosyası olarak diske kaydedilir. Böylece daha sonra kolayca yüklenip kullanılabilir.

### Projenin Amacı:

Bu proje, makine öğrenmesi süreçlerine pratik bir giriş yapmayı sağlar. Veri analizi, model oluşturma, hiperparametre ayarlama, model değerlendirme ve sonuçların görselleştirilmesi gibi temel kavramları öğrenmek için idealdir. Iris veri seti ise makine öğrenmesi alanında en çok kullanılan ve yeni başlayanlar için klasik bir örnektir.

### Gereksinimler:

- Python 3.x  
- scikit-learn  
- pandas  
- seaborn  
- matplotlib  
- joblib  

### Kullanım:

Projeyi çalıştırmak için terminalde:

```bash
python iris_classification.py

