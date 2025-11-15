# Przykłady analizy z użyciem K*-Means

## 1. Dane rzeczywiste: Iris

Uruchomienie eksperymentu:

```
python experiments/realdata_experiment.py
```
Przykładowy wynik:

| Metryka        | Wartość |
|----------------|---------|
| True k         | 3       |
| Estimated k*   | 8       |
| ARI            | 0.612   |
| NMI            | 0.688   |
| Silhouette     | 0.467   |

Interpretacja:
- Algorytm dobrze rozdziela dane, mimo że estymowana liczba klastrów jest nieco większa niż prawdziwa.
- ARI i NMI pokazują umiarkowaną zgodność z prawdziwymi klasami.
- Silhouette > 0,4 wskazuje na wyraźne grupowanie punktów.

---

## 2. Dane syntetyczne: 2D Blobs

Uruchomienie eksperymentu:
```
python experiments/synthetic_experiment.py
```

Przykładowy wynik:

| Metryka        | Wartość |
|----------------|---------|
| True k         | 4       |
| Estimated k*   | 4       |
| ARI            | 1.0     |
| NMI            | 1.0     |
| Silhouette     | 0.65    |

Dodatkowo generowany jest wykres 2D:
- Punkty pokolorowane według przypisanego klastra.
- Centroidy zaznaczone jako czerwone X.

---

## 3. Wnioski

- Algorytm automatycznie dostosowuje liczbę klastrów do danych.
- Synthetic data: idealne odwzorowanie prawdziwych klastrów (ARI=1.0).
- Iris dataset: poprawne grupowanie, możliwe nadmiarowe podziały (k* > true k), co jest typowe dla mechanizmu split/merge.
- Wizualizacje pomagają intuicyjnie ocenić jakość klasteryzacji.