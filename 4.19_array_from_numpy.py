# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 20:43:16 2021

@author: mattt
"""

# 1. Załaduj moduł numpy
import numpy as np

# 2. Utwórz w zmiennej a jednowymiarową tablicę z elementami od 0 do 19.
# Hint: Służy do tego metoda np.arange(...)
a = np.arange(20)  # array range?
a
# 3. Wyświetl kształt tej tablicy. Hint: służy do tego właściwość shape
a.shape

# 4. Wyświetl zerowy element tablicy. Wyświetl czwarty element tablicy.
# Hint: Korzystaj z [] i numeruj od 0
a[0], a[3], a[4]

# 5. Zmień kształt tablicy na 2x10. Wynik przekształcenia zapisz w a.
# Potem ponownie sprawdź kształt i wyświetl zawartość tablicy.
# Hint: Skorzystaj z metody reshape(...)
a.reshape(2, 10)  # divide to 2 elements of 10 in each
a.reshape(10, 2)  # divide to 10 elements of 2 in each
a = a.reshape(2, 10)
a.shape
a

# 6.  Wyświetl element zerowy z przekształconej w poprzednim kroku tablicy.
# Czy jest to ten sam element co poprzednio?
a[0]

# 7. (Uwaga - tu będzie błąd, ale na błędach człowiek się uczy, więc to dobrze).
# Wyświetl czwarty element z tablicy a
# a[3]

# 8. A teraz zróbmy to poprawnie.
# Z tablicy a wyświetl czwarty element znajdujący się w zerowym wierszu.
# Hint: Musisz dwa razy skorzystać z [][]
a[0, 3]
a[0][3]

# 9. Zmień kształt tablicy a na 2x5x2
a = a.reshape(2, 5, 2)  # 2 lists of 5 in each of 2 in each

# 10. Potem ponownie sprawdź kształt i wyświetl zawartość tablicy.
# Hint: hahah - poszukaj odpowiedniego hinta powyżej :)
a.shape
a

# 11. Wyświetl z tablicy a:
a
#     zerowy wiersz
a[0]

#     czwartą kolumnę z zerowego wiersza
a[0, 3]
a[0][3]

#     drugi element z czwartego wiersza w zerowym wierszu
a[0, 3, 1]

# 12. Tworząc tablicę, możesz od razu definiować jej kształt.
# Utwórz tablicę b korzystając z parametrów przekazywanych do metody array
# lub wywołując odpowiednie metody dla już istniejącej tablicy,
# tak aby zawierała ona elementy parzyste od 0 do 40 o kształcie 4x5
b = np.arange(0, 40, 2).reshape(4, 5)

# 13. Polecenie
b

a_python_list = [2**x for x in range(10)]

# tworzy standardową pythonową listę dziesięciu kolejnych potęg dwójki.
# W oparciu o tą listę utwórz obiekt array c
c = np.array(a_python_list)
c
type(c)

# 14. To może się wydawać trochę dziwne,
# ale czasami w uczeniu maszynowym potrzebne są bardzo specyficzne obiekty array.
# W tym zadaniu utworzysz takie "dziwaczne" tablice

#     Utwórz tablicę zero_array składającą się z 10 zer.
#       Hint: Skorzystaj z metody zeros
zero_array = np.zeros(10)
zero_array

#     Utwórz tablicę one_array składającą się z 10 jedynek.
#       Hint: Skorzystaj z metody ones
one_array = np.ones(10)
one_array

#     Utwórz tablicę empty_array składającą się ze 100  byle-jakich-liczb.
#       Hint skorzystaj z metody empty
#     Ta metoda tworzy tablicę, która zawiera to, co akurat było w pamięci.
#       Wielokrotne uruchamianie w/w polecenia może dawać różne wyniki,
#       ale to nie jest jakiś świetny sposób na generowanie liczb losowych
empty_array = np.empty(100)
empty_array

#     Utwórz tablicę lucky_array o wymiarach 5x5 składającą się z samych trzynastek.
#       Hint: Skorzystaj z metody full
lucky_array = np.full(shape=(5, 5), fill_value=13)
lucky_array

#     Utwórz tablicę diagonal_array [przekątna] o wymiarach 5x5,
#       która na przekątnej ma jedynki, a pozostałe wartości to zera.
#       Hint: Skorzystaj z metody eye
diagonal_array = np.eye(5)
diagonal_array

#     Utwórz tablicę random_array składającą się z 10 losowych liczb.
#       Hint: Skorzystaj z metody np.random.random
random_array = np.random.random(10)
random_array

#     Utwórz tablicę linspace_array zawierającą 5 elementów,
#       która jako zerową wartość ma 100, jako ostatnią ma wartość 200,
#       i wszystkie elementy różnią się od siebie o tyle samo.
#       Hint: Skorzystaj z metody linspace
linspace_array = np.linspace(100, 200, num=5)
len(linspace_array)
linspace_array


#------------------------------------------------------------------------
# Preparing data to Machine Learning
