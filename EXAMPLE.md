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
| Estimated k*   | 2       |
| ARI            | 0.539   |
| NMI            | 0.656   |

Interpretacja:
- Algorytm dobrze rozdziela dane, mimo że estymowana liczba klastrów jest nieco większa niż prawdziwa.
- ARI i NMI pokazują umiarkowaną zgodność z prawdziwymi klasami.

---

## 2. Dane syntetyczne: 2D Blobs

Uruchomienie eksperymentu:
```
python experiments/synthetic_experiment.py
```

Przykładowy wynik:

| Metryka        | Wartość |
|----------------|---------|
| True k         | 7       |
| Estimated k*   | 7       |
| ARI            | 0.895   |
| NMI            | 0.915   |

Dodatkowo generowany jest wykres 2D:
- Punkty pokolorowane według przypisanego klastra.

---

## 3. Wnioski

- Algorytm automatycznie dostosowuje liczbę klastrów do danych.
- Iris dataset: poprawne grupowanie, możliwe nadmiarowe podziały (k* > true k), co jest typowe dla mechanizmu split/merge.
- Wizualizacje pomagają intuicyjnie ocenić jakość klasteryzacji.
