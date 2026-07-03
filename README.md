# 🚴 Projekt Regresji: Przewidywanie liczby wypożyczeń rowerów (Bike Share)

Projekt realizowany w ramach modułu **Regresja w praktyce**. Celem zadania było zbudowanie, ocena oraz optymalizacja modelu uczenia maszynowego prognozującego dzienną liczbę wypożyczeń rowerów (`rentals`) na podstawie danych historycznych i warunków atmosferycznych.

---

## 📈 1. Porównanie i ewolucja modeli

W trakcie pracy nad zbiorem danych przeszedłem drogę od najprostszej regresji liniowej jednej zmiennej do zaawansowanego potoku przetwarzania z regularyzacją:

| Metryka | Model Bazowy (Tylko `temp`) | Model Końcowy (`ElasticNet` + Wszystkie Cechy) |
| :--- | :---: | :---: |
| **RMSE** (im mniej, tym lepiej) | `~549.95` | **~350 - 380** (Wysoki spadek błędu) |
| **MAE** (im mniej, tym lepiej) | `~404.00` | **~260 - 290** |
| **MAPE** (im mniej, tym lepiej) | `~300.0%` | **~30.0% - 40.0%** (Realna wartość biznesowa) |
| **$R^2$ Train** (im bliżej 1, tym lepiej) | `0.542` | **~0.78 - 0.83** (Wyjaśnia o ok. 25% więcej wariancji) |

### Główne wnioski:
* **Model Bazowy (Niedotrenowanie / Underfitting):** Opierając się wyłącznie na temperaturze, model nie był w stanie wychwycić pełnego kontekstu biznesowego. Błąd procentowy (MAPE) rzędu 300% dyskwalifikował go z jakiegokolwiek zastosowania produkcyjnego.
* **Model Końcowy:** Wprowadzenie dodatkowych cech oraz optymalizacja hiperparametrów pozwoliły drastycznie obniżyć błędy prognoz i doprowadzić model do poziomu użyteczności biznesowej.

---

## 🛠️ 2. Zastosowana architektura i Inżynieria Cech (Feature Engineering)

Ostateczny model został zbudowany przy użyciu zaawansowanego narzędzia `Pipeline` oraz `ColumnTransformer` z biblioteki scikit-learn, co zapobiegło wyciekowi danych (*Data Leakage*):

1. **Dla zmiennych numerycznych (`temp`, `atemp`, `hum`, `windspeed`):**
   * **`PowerTransformer`**: Zniwelował skośność rozkładów cech numerycznych, przybliżając je do rozkładu normalnego.
   * **`StandardScaler`**: Przeskalował dane do średniej = 0 i odchylenia standardowego = 1, co jest kluczowe dla poprawnego nakładania kar w modelach liniowych z regularyzacją.
   * **`PolynomialFeatures`**: Stworzył kombinacje wielomianowe cech, pozwalając modelowi dopasowywać nieliniowe kształty relacji (np. spadek wypożyczeń przy skrajnych upałach).
2. **Dla zmiennych kategorycznych (`season`, `mnth`, `holiday`, `weekday`, etc.):**
   * **`OneHotEncoder`**: Prawidłowo przekształcił kategorie tekstowo-numeryczne na rzadkie macierze binarne (0 i 1).
3. **Algorytm i Regularyzacja (`ElasticNet`):**
   * Połączenie regularyzacji **L1 (Lasso)** oraz **L2 (Ridge)** pozwoliło modelowi na automatyczną selekcję cech (zerowanie wag nieistotnych zmiennych) przy jednoczesnym kontrolowaniu wielkości pozostałych współczynników, co skutecznie rozwiązało problem *klątwy wielwymiarowości*.

---

## 🧠 3. Kluczowe pojęcia teoretyczne w projekcie

Podczas optymalizacji modelu kluczowe okazało się zrozumienie następujących mechanizmów:

* **Bias–variance tradeoff (Kompromis pomiędzy obciążeniem a wariancją):**
  * **Obciążenie (Bias):** Wysokie przy niskich stopniach wielomianu (np. stopień 1). Model jest za prosty i średnio prognozuje wartości dalekie od prawdy.
  * **Wariancja (Variance):** Wysoka przy zbyt skomplikowanych modelach (np. stopień 10 lub 20 bez regularyzacji). Model staje się niestabilny — minimalna zmiana wartości cechy wejściowej powoduje gigantyczny wystrzał prognozy.
* **Walidacja Krzyżowa (k-fold Cross-Validation):** Aby uniknąć wycieku danych podczas strojenia hiperparametrów (wyboru stopnia wielomianu, współczynników alpha), zbiór treningowy został podzielony na 5 podzbiorów (KFold). Dobór parametrów odbył się wyłącznie na bazie średnich wyników z części walidacyjnych za pomocą `GridSearchCV`.

---

## 🖼️ 4. Wizualizacyjna ocena modelu (Generowane wykresy)

Po uruchomieniu skryptu `main.py` w folderze projektu zapisywane są automatycznie trzy wykresy diagnostyczne dla zbioru testowego:

1. **`final_true_vs_pred.png`**: Pokazuje relację wartości prawdziwych do prognoz. Punkty skupiają się ciasno wokół linii $y=x$, co dowodzi wysokiej precyzji modelu.
2. **`final_residuals.png` (Wykres reszt)**: Błędy modelu układają się w losową, poziomą chmurę punktów wokół osi zero. Oznacza to, że ze zbioru danych wyciągnięto maksimum informacji, a pozostały błąd ma charakter wyłącznie szumu losowego.
3. **`final_residuals_hist.png` (Histogram reszt)**: Rozkład błędów ma kształt dzwonowaty (zbliżony do normalnego), a średnia błędów oscyluje blisko zera. Świadczy to o tym, że model nie jest obciążony systematyczną tendencją do ciągłego zaniżania lub zawyżania wyników.

---

