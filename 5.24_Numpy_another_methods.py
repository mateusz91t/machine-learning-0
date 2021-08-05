# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:05:54 2021

@author: mattt
"""
import numpy as np
# 1. W zmiennej X zapisz tablicę dwuwymiarową z kolejnymi liczbami od 1 do 25.
# (Skorzystaj z arange() i reshape())
X = np.arange(1, 26).reshape(5, 5)
X
# 2. W zmiennej Ones zapisz tablicę o takim samym kształcie jak X,
# ale całą wypełnioną tylko jedynkami.
# (Skorzystaj z ones())
Ones = np.ones((5, 5), dtype=int)
Ones
# 3. Kiedy mnożąc przez siebie zwykłe liczby, chcesz żeby liczba pomnożona
# przez jakąś wartość dawała w wyniku tą samą liczbę,
# to tą jakąś wartością powinno być 1 (tzw. element neutralny).
# Czy tablica Ones jest taką "macierzową jedynką"?
# Sprawdź to mnożąc przez siebie X i Ones. Czy wynik jest równy X?
# (Skorzytaj z dot() )
Ones * X
X.dot(Ones)
Ones.dot(X)
np.dot(X, Ones)
# 4. W algebrze macierzy elementem neutralnym jest macierz zer,
# wypełniona jedynkami na przekątnej.

#     Do zmiennej diag zapisz macierz o takich wymiarach jak X
#     i wypełnioną samymi zerami
#     (skorzystaj z zeros())
diag = np.zeros((5, 5), dtype=int)
diag
#     Wypełnij diag jedynkami na przekątnej.
#     (Skorzystaj z fill_diagonal()
np.fill_diagonal(diag, 1)
diag
#     Pomnóż przez siebie X i diag i sprawdź czy wynik to X.
#     (Skorzystaj z dot())
X
X * diag
X.dot(diag)
# 5. Wyświetl tablicę o wymiarach takich jak X, gdzie
# (we wszystkich punktach korzystaj z where):

#     występują tylko jedynki i zera. Jedynki mają się pojawić tam,
#     gdzie w X występuje wartość > 10, a zero w pozostałych przypadkach
np.where(X > 10, 1, 0)
#     występują tylko jedynki i zera. Jedynki mają się pojawiać tam,
#     gdzie w X występuje wartość parzysta, a zero w pozostałych przypadkach
np.where(X % 2 == 0, 1, 0)
#     występują tylko liczby parzyste. Jeśli w X wartość była parzysta,
#     to należy ją przepisać. Jeśli była nieparzysta , to dodać 1
np.where(X % 2 == 0, X, X + 1)
# 6. Utwórz tablicę X_bis wyznaczoną w następujący sposób.
# Dla wartości w X większych od 10 zwrócona jest wartość 2 razy większa,
# a dla pozostałych 0. Policz ile jest elementów niezerowych w X_bis
# (skorzystaj z count_nonzero())
X_bis = np.where(X > 10, X * 2, 0)
np.count_nonzero(X_bis)
np.count_nonzero(diag)
# 7. Oto dwie tablice:

x = np.array([[10, 20, 30], [40, 50, 60]])
y = np.array([[100], [200]])

# Dodaj (a właściwie doklej) do tablicy x tablicę y jako nową kolumnę.
# (Skorzystaj z append() z parametrem axis=1)
x + y
np.append(x, y)  # works like flatten
np.append(x, y, axis=1)  # adds to end of each row
# 8. Oto dwie tablice:

x = np.array([[10, 20, 30], [40, 50, 60]])
y = np.array([[100, 200, 300]])
x
y

# Dodaj (a właściwie doklej) do tablicy x tablicę y, jako nowy wiersz
np.append(x, y, axis=0)

# ----------------------------------------------------------------------------
# Summary: to add new column use axis=1, to add new row use axis=0
# ----------------------------------------------------------------------------

# 9. Do tablicy x z poprzedniego punktu, doklej tablicę x jako nowe wiersze
# (wiersze będą zduplikowane)
np.append(np.append(x, y, axis=0), x, axis=0)
