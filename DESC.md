# Projekt: K*-Means - Algorytm Klasteryzacji Bez Parametrów

## 1. Postawienie problemu

Celem projektu jest implementacja i analiza algorytmu **K*-Means** zaproponowanego przez Louis Mahona i Mirellę Lapatę (2025).  
K*-Means jest wariantem klasycznego K-Means, który automatycznie dobiera liczbę klastrów, eliminując konieczność podawania jej z góry.  

Klasyczny K-Means wymaga określenia liczby klastrów **k**, co jest trudne w praktyce, zwłaszcza gdy struktura danych nie jest znana.  
K*-Means rozwiązuje ten problem dzięki mechanizmowi **split/merge**, który dynamicznie dzieli lub scala klastry w trakcie iteracji, na podstawie wariancji punktów w klastrze.

---

## 2. Dotychczasowe rozwiązania

- **K-Means**: szybki i prosty, wymaga podania k, wrażliwy na inicjalizację centroidów.
- **Hierarchical Clustering**: nie wymaga k, ale jest kosztowny obliczeniowo.
- **DBSCAN**: wykrywa klastry o dowolnym kształcie, ale wymaga podania parametrów epsilon i min_samples.

K*-Means łączy prostotę K-Means z automatycznym wyborem liczby klastrów, oferując kompromis między dokładnością a wygodą stosowania.

---

## 3. Nowe podejście (K*-Means)

1. Inicjalizacja centroidów losowo (minimum `k_min`).
2. Przypisanie punktów do najbliższego centroidu.
3. Dla każdego centroidu:
   - Obliczenie wariancji punktów w klastrze.
   - Jeśli wariancja jest wysoka i nie przekroczono `k_max`, klaster jest dzielony (split).
   - Jeśli wariancja jest niska i są podobne klastry, mogą zostać scalone (merge).
4. Iteracja do momentu zbieżności centroidów lub osiągnięcia `max_iter`.

Parametry algorytmu:
- `k_min` – minimalna liczba klastrów.
- `k_max` – maksymalna liczba klastrów (opcjonalnie).
- `max_iter` – maksymalna liczba iteracji.
- `tol` – tolerancja dla zbieżności centroidów.
- `random_state` – ustawienie ziarna generatora losowego.

---

## 4. Wykorzystane dane

- **Dane syntetyczne**: generowane funkcją `make_blobs` w 2D, umożliwiające wizualizację klastrów.
- **Dane rzeczywiste**: zbiór Iris (150 próbek, 4 cechy, 3 klasy).

---

## 5. Wyniki i analiza

- K*-Means samodzielnie określa liczbę klastrów (`k*`).
- Dla danych Iris:
  - True k = 3
  - Estimated k* ≈ 3–8 (zależnie od losowej inicjalizacji i parametrów)
- Metryki jakości klasteryzacji:
  - **ARI (Adjusted Rand Index)** – zgodność z prawdziwymi klasami.
  - **NMI (Normalized Mutual Information)** – informacja wzajemna między klasami.
  - **Silhouette Score** – spójność i separacja klastrów.

---

## 6. Podsumowanie

Projekt dostarcza:
- Pełną implementację K*-Means w Pythonie.
- Eksperymenty na danych syntetycznych i rzeczywistych.
- Obliczanie metryk jakości klasteryzacji.
- Wizualizacje wyników dla danych 2D.
- Repozytorium gotowe do dalszych eksperymentów i analiz.
