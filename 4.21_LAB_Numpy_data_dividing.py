# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:12:57 2021

@author: mattt
"""

# 1. Zacznijmy od zaimportowania numpy korzystając ze standardowego aliasu
import numpy as np

# 2. Utwórz tablicę arr zawierającą nieparzyste liczby od 5 do 29.
# Hint: skorzystaj z funkcji arange
arr = np.arange(5, 30, 2)
arr

# 3. Utwórz tablicę boolArr zawierającą wartości True/False.
# True, gdy na odpowiedniej pozycji w arr jest liczba < 10,
# a False w przeciwnym razie.
# Hint: zacznij od wykonania operacji porównania na tablicy arr,
# a potem zapisz wynik w zmiennej
boolArr = arr < 10
boolArr
# 4. Wyświetl tylko te pozycje z tablicy arr, gdzie:
#     na odpowiedniej pozycji w boolArr znajduje się wartość True.
#     Hint: odwołując się do zawartości tablicy arr poprzez [] możesz
#     przekazywać tablicę wartości logicznych
arr[boolArr]
#     wartość pozycji jest < 20. Hint: warunek logiczny może być zapisywany
#     bezpośrednio w []
arr[arr > 20]
#     wartość pozycji jest podzielna przez 3.
#     Hint: Warunek logiczny powinien sprawdzać,
#     czy reszta z dzielenia pozycji tablicy przez 3 jest równa 0
arr[arr % 3 == 0]
#     wartość pozycji jest większa niż 10 i mniejsza niż 20.
#     Hint: Warunek logiczny w [] może być bardziej skomplikowany,
#     np. można skorzystać z operatora &
arr[(10 < arr) & (arr < 20)]
# 5. Zmień kształt arr na 4x6, a następnie wyświetl
arr = np.arange(24).reshape(4, 6)
arr
# (pamiętaj o tym, że w pythonie numerujemy od zera,
# więc tutaj, zdecydowałem się pisać po pythonowemu - też liczę od zera):
#     pierwszy wiersz.
#     Hint: korzystaj z []
arr[1]
#     z pierwszego wiersza drugą pozycję.
#     Hint: korzystaj z podwójnego [numer_wiersza][numer_kolumny]
#     lub z adresowania w postaci [numer wiersza, numer_kolumny]
arr[1, 2]
#     z pierwszego wiersza pozycje od 2 do 3.
#     Hint: Odwołując się do kolumn możesz skorzystać ze "slicing".
#     Wartość określająca koniec zakresu do niego nie wchodzi
#     (jak zwykle w Pythonie)
arr[1, 2:4]
#     z pierwszego wiersza pozycje od 2 do 4.
#     Hint: pamiętaj że ostatnia liczba z zakresu nie jest brana pod uwagę -
#     przedział jest z prawej strony otwarty
arr[1, 2:5]
#     z pierwszego wiersza wszystkie pozycje, ale budując to wyrażenie
#      skorzystaj z notacji [numer_wiersza, zakres_kolumn].
#     Kiedy tworzysz zakres dla wszystkich kolumn - po prostu napisz :
arr[1, ]
arr[1, :]
#     z drugiej kolumny wszystkie wiersze.
#     Hint: wyrażenie będzie podobne, co w poprzednim punkcie,
#     tylko zmień kolejność w []
arr[:, 2]
#     z wierszy od 0 do 2 wyświetl tylko 2 kolumnę.
#     Hint: pamiętaj że ostatnia liczba z zakresu nie jest brana pod uwagę -
#     przedział jest z prawej strony otwarty
arr[0:3, 2]
#     wiersze do drugiego, druga kolumna.
#     Hint: Istnieje odpowiednia notacja slice odpowiadająca
#     za "do n-tego elementu"
arr[:3, 2]
#     wiersze do drugiego, kolumny od 2 do 3.
#     Hint: Użyj operatora slice w odwołaniu do wiersza i do kolumny
arr[:3, 2:4]
#     wszystkie wiersze, ostatnia kolumna.
#     Hint: Za odwołanie do ostatniej kolumny odpowiada zapis -1
arr[:, -1]
#     wszystkie wiersze, wszystkie kolumny oprócz ostatniej.
#     Hint: Zbuduj slice z : i liczbą -1
arr[:, :-1]
# 6. Do zmiennej arr zapisz tablicę 50-elementową, o wymiarach 10x5.
arr = np.arange(50).reshape(10, 5)
arr
# Teraz trochę zbliżymy się do naszego celu,
# czyli jakiegoś "półautomatycznego" cięcia zbioru danych
# na dane uczące i testowe. Wykonaj polecenia:

# how much data should be "test-data' - here 20%
split_level = 0.2
num_rows = arr.shape[0]
split_border = split_level * num_rows

# Teraz zbuduj wyrażenie zwracające
#     wiersze od początku do split_border i wszystkie kolumny
arr[:round(split_border)]
#     wiersze od split_border do końca i wszystkie kolumny
#     Hint: wartość split_border, to liczba float.
#     Zaokrąglij ją do liczby całkowitej podczas budowania slice(funkcja round)
arr[round(split_border):]

# To już prawie dobry zbiór uczący i testowy, bo...
# dane testowe byłyby zawsze brane z początku zbioru...
# a to trochę mało losowe. Dlatego "wymieszaj" wiersze:

np.random.shuffle(arr)
arr
# i powtórz dzielenie tablicy arr na 20% i 80% jeszcze raz. Lepiej?
arr[:round(split_border)]
arr[round(split_border):]
# 7. Finiszujemy. Napiszemy kod, który dzieli dane na uczące
# i testowe oraz rozdziela je na zbiór feature i zbiór target.
# Ostateczne oznaczenia mają być takie:

#     X_learn - 80% danych - wszystkie kolumny oprócz ostatniej
#     X_test - 20% danych - wszystkie kolumny oprócz ostatniej
#     y_learn - 80% danych - ostatnia kolumna
#     y_test - 20% danych - ostatnia kolumna

# Oto dane:

data = np.arange(500).reshape(100, 5)

# Jeśli wiesz jak to zrobić, to jazda!
# A jak potrzebujesz dokładniejszych instrukcji, oto one:

#     przetasuj dane, żeby były ułożone losowo

#     podobnie, jak w kodzie powyżej wyznacz nową wartość split_border,
#     która określa ile wierszy będzie wziętych do danych testowych

#     Wyznacz X_test - wiersze od początku do split_border,
#     wszystkie kolumny oprócz ostatniej

#     Wyznacz X_train - wiersze od split border do końca,
#     wszystkie kolumny oprócz ostatniej

#     Wyznacz y_test - wiersze od początku do split_border,
#     tylko ostatnia kolumna

#     Wyznacz y_train - wiersze od split_border do końca,
#     tylko ostatnia kolumna


def train_test_x_y(feature, target,
                   test_size: float = .25,
                   shuffle: bool = False)\
        -> tuple:
    """
    Split data to 4 datasets: 2 to train and 2 to test.

    Parameters
    ----------
    data_x : TYPE
        DESCRIPTION.
    data_y : TYPE
        DESCRIPTION.
    test_size : float, optional
        DESCRIPTION. The default is .25.
    shuffle : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    np.ndarray
        A feature train dataset.
    np.ndarray
        A feature test dataset.
    np.ndarray
        A target train dataset.
    np.ndarray
        A target test dataset.

    """
    border = round(feature.shape[0] * test_size)
    x, y = np.array(feature), np.array(target)
    if shuffle:
        np.random.shuffle(x), np.random.shuffle(y)
    return x[:-border], x[-border:], y[:-border], y[-border:]


tts1 = train_test_x_y(data[:, :-1], data[:, -1], test_size=.2, shuffle=False)

# Podobne czynności, co w punkcie 7, są wykonywane w train_test_split,
# a teraz ta funkcja nie jest już dla Ciebie tajemnicza. Fajnie, co?
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = \
tts2 =\
    train_test_split(data[:, :-1], data[:, -1], test_size=0.2, shuffle=False)

for i in range(4):
    print(np.array_equal(tts1[i], tts2[i]))
