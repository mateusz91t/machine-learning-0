# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 15:07:24 2021

@author: mattt
"""

#
# 5.23

# 3==<
# Neuron składa się z:
#    * Dendrytów, z których przychodzą informacje
#    * jądra komórki
#    * Axonu, przez który przechodzi sygnał do synaps
#    * Synaps, które przekazują sygnał to następnego Dendryta innego neuronu

# Przykładowy neuron: Czy dany przedmiot to stół?
# Dendrony:
#    x1: ile ma nóg?
#    x2: czy jest twardy?
#    x3: czy żyje?
#    x4: masa?

#
# 5.25

# Wszystkie dendryty tworzą wektor oznaczany zwykle X,
# np. jeśli istnieją 4 dendryty, to:
# X = (x1, x2, x3, x4)
# Przykładowo wektory:
#    X(pies) = (4, False, True, 5)
#    X(stół) = (4, True, False, 5)

# Poszczególne dendryty mogą mieć różne wagi oznaczane zwykle W,
# np. informacja z Dendrytu nr 1 jest ważniejsza od informacji z Dendrytu nr 2.
#    W = (w1, w2, w3, w4)
#

# Całkowita wartość pobudzenia perceptronu
# Moc sygnału oznaczamy zwykle Z.
# Obliczamy ją poprzez wymnożenie wektorów i dodanie wynikóW X * W
#    Z = x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4
#
# Jednakże mnożenie wektorów Z = X * W, to mnożenie macierzy.
# Wiersz powinien być mnożony przez kolumnę.
# Trzeba więc jeden wyraz transponować:
#    Z = X^T * W
#
# Funkcja aktywacji (|) Fi:
#          / 1 for z >= -o-(Teta)
# (|) (Z) =
#          \ -1 for z < -o-
# gdzie wartość -o- to wartość graniczna pobudzenia całkowitego
#
# Funkcja (|) fi jest nazywana funkcją skoku lub Heavyside'a,
# bo wynik jest konwertowany przez nią na [0, 1]

# -o- zmieniła się w przeszłości na iloczyn w0 * x0
# Od tamtej pory wzór na Fi od Z wygląda (dla powyższego neuronu) następująco:
#    Z = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4
#    a więc
#             / 1 for z >= 0
#    (|) (Z) =
#             \ -1 for z < 0
# wzór w postaci sumy (E to Sigma):
#     m
# Z = E wj * xj
#    j=0
# dla ilości cech = m
# słownie: Z równa się Sigma, dla j od 0 do m, wj * xj
# Funkcja fi dla perceptronu jest nazywana funkcją aktywacji.

#
# 5.27
# Y to prawdziwy wynik, a Ý to przewidywany wynik.
# wzór na zmianę pojedynczej wagi dla jednej próbki:
#    ▲wj = n(y - ý) xj
# n to Eta: współczynnk uczenia, prędkości uczenia:
# jak mocno będą zmieniać się wagi.
# Szybsze uczenie = większy błąd
#
# Po każdej iteracji następuje poprawa wj:
#    wj = wj + ▲wj
#
# Epoka - jedna iteracja uczenia, czyli:
# predykcji, sprawdzenia wyniku, poprawienia wag
