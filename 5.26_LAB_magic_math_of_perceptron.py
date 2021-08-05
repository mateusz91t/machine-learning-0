# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:34:07 2021

@author: mattt
"""
import numpy as np
# 1.  Napisz kod, który utworzy tablicę X zawierającą 10 wierszy i 5 kolumn
# o wartościach od -25 do 24.
# (skorzystaj z metody np.arange)
X = np.arange(-25, 25).reshape(10, 5)
# 2. Utwórz kod, który utworzy tablicę ones,
# zawierającą tyle wierszy co X i jedną kolumnę wypełnioną jedynkami
# (skorzystaj z metody np.ones oraz właściwości shape dla X)
ones = np.ones(X.shape[0], dtype=int)  # it does't work in next Ex!
ones = np.ones((X.shape[0], 1), dtype=int)
ones
# 3. Utwórz zmienną X_1, która powstaje przez doklejenie do zmiennej X
# kolumny pochodzącej z ones
# (skorzystaj z metody append przesyłając argument axis=1)
X_1 = np.append(X, ones, axis=1)
# 4. Utwórz wektor w, który ma tyle samo współrzędnych
# co ilość kolumn w zmiennej X_1. Wartości współrzędnych mają być wylosowane
# (skorzystaj z np.random.rand)
w = np.random.rand(X_1.shape[1])
w
# 5. Zdefiniuj funkcję predict przyjmującą jako argument x
# (będzie to jeden wiersz z tablicy X_1) i w, a w niej:

#    pomnóż przez siebie wektor x i w
#    (skorzystaj z metody np.dot).
#    Wynik zapisz w zmiennej total_stimulation

#    napisz wyrażenie if, które będzie działać jak funkcja skoku i

#        dla total_stimulation > 0 zwróci 1

#        a w przeciwnym razie zwróci -1


def predict(x, w):
    total_stimulation = np.dot(x, w)
    return 1 if total_stimulation > 0 else -1


# 6. Przetestuj działanie tej funkcji dla w i X_1[0,]
if __name__ == '__main__':
    predict(X_1[0], w)
# 7. Napisz pętlę for, która dla każdego wiersza z X_1 wywoła funkcję predict.
# Wyniki tej funkcji na razie będziemy tylko wyświetlać
if __name__ == '__main__':
    for row in X_1:
        print(predict(row, w))
