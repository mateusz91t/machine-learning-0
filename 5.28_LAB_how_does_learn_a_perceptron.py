# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:42:59 2021

@author: mattt
"""

# Na lekcji opowiadałem o tym, że wagi perceptronu aktualizują się w oparciu
# o uzyskiwane predykcje.
# W ostatnich minutach filmu można było zobaczyć przepis na perceptron:
#    * wylosuj wagi w
#    * ustal ilość epok
#    * dla każdej epoki
#    * wylicz y_pred
#    * wyznacz delta_w
#    * aktualizuj w

# W tym LAB zrealizujemy ten przepis!
# Zaczynamy od kodu, który udało nam się stworzyć w poprzednim LAB:

import numpy as np

X = np.arange(-25, 25, 1).reshape(10, 5)
ones = np.ones((X.shape[0], 1), dtype=int)
X_1 = np.append(X, ones, axis=1)
w = np.random.rand(X_1.shape[1])


def predict(x, w):
    total_stimulation = np.dot(x, w)
    y_pred = 1 if total_stimulation > 0 else -1
    return y_pred


# 1. Zdefiniuj zmienną y i eta w następujący sposób:

y = np.array([1, -1, -1, 1, -1, 1, -1, -1, 1, -1])
eta = 0.01

# 2. Napisz pętlę for, która będzie jednocześnie iterować przez zmienną X_1 i y
# (skorzystaj z funkcji zip do połączenia X_1 i y.
#  Zmienne, którym będziemy przypisywać wartości zwracane przez zip
#  nazwij x i y_target). W tej pętli

#    * wywołuj funkcję predict dla zmiennej x i w.
#    Wynik zapisuj w zmiennej y_pred
#    (to będzie wartość odgadywana przez perceptron dla próbki x przy wagach w)


#    * wyznacz wartość delta_w, jako wynik mnożenia eta przez
#    (y_target - y_pred) i x

#    * aktualizuj w dodając do niej wartość delta_w

#    * wyświetl aktualną wartość w

# 3. Zdefiniuj zmienną epochs i przypisz jej wartość 10.
epochs = 10
# 4. Napisz pętlę for, która będzie się wykonywać epochs razy.
# W tej pętli wklej pętlę z kroku 2.
results = np.zeros(6).reshape(1, 6)

for i in range(epochs):
    # print(f'epoch {i}')
    for sample in zip(X_1, y):
        x = sample[0]
        y_target = sample[1]
        y_pred = predict(x, w)
        delta_w = eta * (y_target - y_pred) * x
        w += delta_w
        # print(w)
        # results.append(w)
        results = np.append(results, w.reshape(1, 6), axis=0)

# 5. Nie wiem, czy wiesz, ale praktycznie masz napisany perceptron
# (szczegóły w kolejnej lekcji)

results = results[1:]
print(results)

from matplotlib import pyplot as plt

plt.plot(results)
plt.plot(np.transpose(results))
