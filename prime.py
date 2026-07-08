import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Globalna konfiguracja biblioteki Matplotlib w celu zapewnienia spójności wizualnej
plt.rcParams['font.family'] = 'DejaVu Sans'  # Ustawienie czcionki obsługującej szeroki zakres znaków
plt.rcParams['axes.unicode_minus'] = False    # Poprawne renderowanie znaku minus na osiach wykresów (zapobiega błędom fontu)

print("\n" + "="*80)
print("PENGUIN CLASSIFICATION - EXPLORATORY DATA ANALYSIS AND MODEL TRAINING")
print("="*80)

# =========================================================================
# 1. WCZYTANIE I WSTĘPNA ANALIZA STRUKTURY DANYCH (EDA)
# =========================================================================

# Wczytanie zbioru danych z pliku CSV do obiektu DataFrame biblioteki Pandas
penguins = pd.read_csv('penguins.csv')

# Standaryzacja nazw kolumn na format snake_case wraz z jawnym określeniem jednostek miar
penguins = penguins.rename(columns={
    'CulmenLength': 'culmen_length_mm',  # Długość grzbietu dzioba
    'CulmenDepth': 'culmen_depth_mm',    # Wysokość/głębokość dzioba
    'FlipperLength': 'flipper_length_mm',# Długość skrzydła
    'BodyMass': 'body_mass_g',           # Masa ciała
    'Species': 'species'                 # Zmienna docelowa (gatunek)
})

print("\n1. INFORMACJE O ZBIORZE DANYCH:")
print(f"Wymiary: {penguins.shape}")      # Zwraca krotkę (liczba wierszy, liczba kolumn)
print(f"\nPierwsze wiersze:\n{penguins.head()}") # Wyświetla pierwsze 5 rekordów w celu weryfikacji struktury

print(f"\nInformacje o kolumnach:")
penguins.info()                          # Wyświetla typy danych kolumn oraz liczbę niepustych wartości (non-null)

print(f"\nStatystyki opisowe:\n{penguins.describe()}") # Generuje statystyki (średnia, odchylenie std, min, max, kwartyle) dla cech numerycznych
print(f"\nBrakujące wartości:\n{penguins.isnull().sum()}") # Zlicza wartości NaN (brakujące) indywidualnie dla każdej kolumny

# =========================================================================
# 2. EKSPLORACJA I WIZUALIZACJA ZMIENNYCH (EKSPLORACYJNA ANALIZA DANYCH)
# =========================================================================
print("\n2. ROZKŁAD GATUNKÓW PINGWINÓW:")
print(penguins['species'].value_counts()) # Wyświetla liczebność klas w zmiennej docelowej (sprawdzenie zrównoważenia zbioru)

# Inicjalizacja siatki wykresów o wymiarach 2x2 o zdefiniowanym rozmiarze w calach
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Wykres 1 (lewy górny): Rozkład liczebności zmiennej kategorycznej (gatunków) za pomocą wykresu słupkowego
penguins['species'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'orange', 'green'])
axes[0, 0].set_title('Rozkład Gatunków Pingwinów', fontweight='bold')
axes[0, 0].set_ylabel('Liczba obserwacji')
axes[0, 0].set_xlabel('Gatunek')

# Wykres 2 (prawy górny): Wykres rozrzutu (scatter plot) badający zależność liniową cech ciągłych z podziałem na klasy
for species in penguins['species'].unique():
    # Odfiltrowanie danych dla konkretnego gatunku w celu nałożenia osobnej serii punktów
    data = penguins[penguins['species'] == species]
    axes[0, 1].scatter(data['culmen_length_mm'], data['body_mass_g'], label=species, alpha=0.7)
axes[0, 1].set_xlabel('Długość dzioba (mm)')
axes[0, 1].set_ylabel('Masa ciała (g)')
axes[0, 1].set_title('Długość dzioba vs Masa ciała', fontweight='bold')
axes[0, 1].legend()                    # Dodanie legendy identyfikującej kolory klas
axes[0, 1].grid(True, alpha=0.3)       # Włączenie delikatnej siatki pomocniczej

# Wykres 3 (lewy dolny): Wykres pudełkowy (boxplot) obrazujący rozkład statystyczny masy ciała w zależności od gatunku
penguins.boxplot(column='body_mass_g', by='species', ax=axes[1, 0])
axes[1, 0].set_title('Rozkład Masy Ciała po Gatunkach', fontweight='bold')
axes[1, 0].set_xlabel('Gatunek')
axes[1, 0].set_ylabel('Masa ciała (g)')
axes[1, 0].get_figure().suptitle('')   # Czyszczenie automatycznego, domyślnego tytułu generowanego przez Pandas boxplot
axes[1, 0].tick_params(axis='x', rotation=0) # Resetowanie ewentualnego obrotu etykiet osi X

# Wykres 4 (prawy dolny): Macierz korelacji cech numerycznych (współczynnik korelacji Pearsona)
numeric_data = penguins.select_dtypes(include=[np.number]) # Izolacja wyłącznie kolumn numerycznych
correlation_matrix = numeric_data.corr()                   # Obliczenie macierzy korelacji
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1], fmt='.2f') # Wizualizacja jako mapa ciepła
axes[1, 1].set_title('Macierz Korelacji Cech', fontweight='bold')

# Optymalizacja światła między wykresami, eksport do pliku graficznego i zwolnienie pamięci ram RAM
plt.tight_layout()
plt.savefig('penguins_exploration.png', dpi=150, bbox_inches='tight')
print("\n-> Zapisano wykres eksploracji: penguins_exploration.png")
plt.close()

# =========================================================================
# 3. PRZYGOTOWANIE DANYCH (PREPROCESSING) DO PROCESU MODELOWANIA
# =========================================================================
print("\n3. PRZYGOTOWANIE DANYCH DO MODELOWANIA:")

# Strategia radzenia sobie z brakami: usuwanie wierszy zawierających puste wartości (listwise deletion)
penguins_clean = penguins.dropna()
print(f"Liczba obserwacji po usunięciu brakujących wartości: {len(penguins_clean)}")

# Separacja macierzy cech wejściowych (X) od wektora zmiennej docelowej (y)
feature_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X_penguins = penguins_clean[feature_columns]
y_penguins = penguins_clean['species']

# Konwersja etykiet tekstowych (string) klasy docelowej na postać numeryczną (0, 1, 2) wymaganą przez algorytmy Scikit-Learn
le = LabelEncoder()
y_penguins_encoded = le.fit_transform(y_penguins)

print(f"\nKlasy: {le.classes_}")
print(f"Liczba cech: {X_penguins.shape[1]}")
print(f"Liczba obserwacji: {X_penguins.shape[0]}")

# Podział zbioru danych na podzbiór treningowy (75%) i testowy (25%)
# stratify=y_penguins_encoded zapewnia identyczny rozkład klas w obu podzbiorach (kluczowe przy braku idealnego zrównoważenia)
# random_state=42 gwarantuje determinizm i powtarzalność podziału przy każdym uruchomieniu kodu
X_train_peng, X_test_peng, y_train_peng, y_test_peng = train_test_split(
    X_penguins, y_penguins_encoded, test_size=0.25, random_state=42, stratify=y_penguins_encoded
)

print(f"\nObserwacje treningowe: {X_train_peng.shape[0]}")
print(f"Obserwacje testowe: {X_test_peng.shape[0]}")

# Standaryzacja cech (Z-score normalization): przekształcenie danych tak, aby średnia wynosiła 0, a odchylenie standardowe 1.
# Zapobiega to dominacji modeli przez cechy o szerokich przedziałach wartości (np. body_mass_g).
scaler_peng = StandardScaler()
X_train_peng_scaled = scaler_peng.fit_transform(X_train_peng) # Dopasowanie skalera do danych treningowych i ich transformacja
X_test_peng_scaled = scaler_peng.transform(X_test_peng)     # Transformacja danych testowych na bazie parametrów zbioru treningowego (brak wycieku danych)

# =========================================================================
# 4. TRENOWANIE I EWALUACJA MODELI PREDYKCYJNYCH
# =========================================================================
print("\n4. TRENOWANIE MODELI:")
penguins_results = {} # Słownik do agregacji wyników metryki F1 dla poszczególnych algorytmów

# 4a. Regresja Logistyczna (model parametryczny, wymaga standaryzacji cech)
print("\nTrenowanie Logistic Regression...")
lr_peng = LogisticRegression(max_iter=1000, random_state=42) # Zwiększony limit iteracji gwarantuje zbieżność algorytmu optymalizacyjnego
lr_peng.fit(X_train_peng_scaled, y_train_peng)
y_pred_lr_peng = lr_peng.predict(X_test_peng_scaled)
# Obliczenie F1-score w wariancie 'macro' (średnia arytmetyczna F1-score dla każdej klasy traktowanej równorzędnie)
f1_lr_peng = f1_score(y_test_peng, y_pred_lr_peng, average='macro')
penguins_results['Logistic Regression'] = f1_lr_peng
print(f"   F1 Score (macro): {f1_lr_peng:.4f}")

# 4b. Drzewo Decyzyjne (model nieparametryczny, niewrażliwy na skalowanie danych)
print("Trenowanie Decision Tree...")
dt_peng = DecisionTreeClassifier(max_depth=5, random_state=42) # max_depth ogranicza złożoność drzewa, pełniąc rolę regularyzacji (zapobiega overfittingowi)
dt_peng.fit(X_train_peng, y_train_peng)
y_pred_dt_peng = dt_peng.predict(X_test_peng)
f1_dt_peng = f1_score(y_test_peng, y_pred_dt_peng, average='macro')
penguins_results['Decision Tree'] = f1_dt_peng
print(f"   F1 Score (macro): {f1_dt_peng:.4f}")

# 4c. Las Losowy (algorytm zespołowy typu bagging, redukuje wariancję pojedynczych drzew)
print("Trenowanie Random Forest...")
rf_peng = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1) # n_jobs=-1 alokuje wszystkie dostępne rdzenie procesora do obliczeń równoległych
rf_peng.fit(X_train_peng, y_train_peng)
y_pred_rf_peng = rf_peng.predict(X_test_peng)
f1_rf_peng = f1_score(y_test_peng, y_pred_rf_peng, average='macro')
penguins_results['Random Forest'] = f1_rf_peng
print(f"   F1 Score (macro): {f1_rf_peng:.4f}")

# 4d. K-Najbliższych Sąsiadów (algorytm leniwy oparty na metryce odległości, bezwzględnie wymaga standaryzacji)
print("Trenowanie K-Nearest Neighbors...")
knn_peng = KNeighborsClassifier(n_neighbors=5) # Klasyfikacja na podstawie głosowania większościowego 5 najbliższych obiektów w przestrzeni cech
knn_peng.fit(X_train_peng_scaled, y_train_peng)
y_pred_knn_peng = knn_peng.predict(X_test_peng_scaled)
f1_knn_peng = f1_score(y_test_peng, y_pred_knn_peng, average='macro')
penguins_results['K-Nearest Neighbors'] = f1_knn_peng
print(f"   F1 Score (macro): {f1_knn_peng:.4f}")

# 4e. AdaBoost (algorytm zespołowy typu boosting, sekwencyjnie wzmacniający słabe modele)
print("Trenowanie AdaBoost...")
base_est = DecisionTreeClassifier(max_depth=1, random_state=42) # Słaby klasyfikator bazowy (tzw. pień decyzyjny - decision stump)
ada_peng = AdaBoostClassifier(estimator=base_est, n_estimators=50, random_state=42)
ada_peng.fit(X_train_peng, y_train_peng)
y_pred_ada_peng = ada_peng.predict(X_test_peng)
f1_ada_peng = f1_score(y_test_peng, y_pred_ada_peng, average='macro')
penguins_results['AdaBoost'] = f1_ada_peng
print(f"   F1 Score (macro): {f1_ada_peng:.4f}")

# =========================================================================
# 5. AGREGACJA, RANKING I WIZUALIZACJA PORÓWNAWCZA WYNIKÓW
# =========================================================================
print("\n5. PODSUMOWANIE WYNIKÓW MODELI:")
print("\nRanking modeli:")
# Sortowanie słownika wyników malejąco według wartości metryki F1-score
sorted_results = sorted(penguins_results.items(), key=lambda x: x[1], reverse=True)
for idx, (model_name, score) in enumerate(sorted_results, 1):
    print(f"  {idx}. {model_name}: {score:.4f}")

# Generowanie wykresu słupkowego porównującego efektywność algorytmów
fig, ax = plt.subplots(figsize=(10, 6))
models_names = list(penguins_results.keys())
models_scores = list(penguins_results.values())
# Przypisanie koloru zielonego dla najwyższego wyniku predykcyjnego i niebieskiego dla pozostałych modeli
colors = ['green' if score == max(models_scores) else 'skyblue' for score in models_scores]
ax.bar(models_names, models_scores, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('F1 Score (macro)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Porównanie Wyników Modeli na Zbiorze Pingwinów', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.0]) # Sztywne ustawienie zakresu osi Y od 0 do 100% dla obiektywnej prezentacji różnic
ax.grid(True, alpha=0.3, axis='y')
# Nakładanie etykiet tekstowych z precyzyjną wartością numeryczną bezpośrednio nad każdym ze słupków
for i, (name, score) in enumerate(zip(models_names, models_scores)):
    ax.text(i, score + 0.02, f'{score:.4f}', ha='center', fontsize=10, fontweight='bold')
plt.xticks(rotation=45, ha='right') # Obrót etykiet osi X o 45 stopni zapobiega ich nakładaniu się
plt.tight_layout()
plt.savefig('penguins_models_comparison.png', dpi=150, bbox_inches='tight')
print("\n-> Zapisano wykres porównania modeli: penguins_models_comparison.png")
plt.close()

# =========================================================================
# 6. SZCZEGÓŁOWA ANALIZA DIAGNOSTYCZNA NAJLEPSZEGO MODELU KLASYFIKACJI
# =========================================================================
best_model_name = sorted_results[0][0] # Wyznaczenie nazwy algorytmu, który wygrał ranking
print(f"\n6. ANALIZA NAJLEPSZEGO MODELU: {best_model_name}")
# Warunkowa ekstrakcja odpowiednich predykcji oraz poprawnego zbioru testowego (skalowanego bądź nieskalowanego) dla zwycięzcy
if best_model_name == 'Logistic Regression':
    best_model = lr_peng
    y_pred_best = y_pred_lr_peng
    X_test_best = X_test_peng_scaled
elif best_model_name == 'Decision Tree':
    best_model = dt_peng
    y_pred_best = y_pred_dt_peng
    X_test_best = X_test_peng
elif best_model_name == 'Random Forest':
    best_model = rf_peng
    y_pred_best = y_pred_rf_peng
    X_test_best = X_test_peng
elif best_model_name == 'K-Nearest Neighbors':
    best_model = knn_peng
    y_pred_best = y_pred_knn_peng
    X_test_best = X_test_peng_scaled
else: # Przypadek dla AdaBoost
    best_model = ada_peng
    y_pred_best = y_pred_ada_peng
    X_test_best = X_test_peng

print(f"\nRaport klasyfikacji dla {best_model_name}:")
# Wyświetlenie szczegółowych metryk: Precision (Precyzja), Recall (Czułość) oraz F1-score indywidualnie dla każdej podklasy
# Rzutowanie nazw klas na typ string zabezpiecza przed błędami parsowania obiektów klasyfikatora
print(classification_report(y_test_peng, y_pred_best, target_names=[str(c) for c in le.classes_]))

# Obliczenie macierzy konfuzji/błędów (Confusion Matrix) w celu identyfikacji charakterystyki błędnych predykcji
cm_best = confusion_matrix(y_test_peng, y_pred_best)
plt.figure(figsize=(8, 6))
# Rysowanie macierzy błędów w postaci mapy ciepła z tekstowym naniesieniem wartości liczbowych (fmt='d')
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(c) for c in le.classes_],
            yticklabels=[str(c) for c in le.classes_])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidywana klasa')
plt.tight_layout()
# Dynamiczne generowanie nazwy pliku wyjściowego na bazie nazwy wybranego modelu ze spacji na podłogę
plt.savefig(f'penguins_confusion_matrix_{best_model_name.replace(" ", "")}.png', dpi=150, bbox_inches='tight')
print(f"\n-> Zapisano macierz błędów: penguins_confusion_matrix_{best_model_name.replace(' ', '_')}.png")
plt.close()

print("\n" + "="*80)
print("ANALIZA ZBORU PINGWINÓW ZAKOŃCZONA")
print("="*80)