# GeoNewsClassify

Classify news into categories.

Installation

```
    pip3 install beautifulsoup4
    pip3 install jieba
    pip3 install googlemaps
    pip3 install flask
```

Dictionary
```
location.dict : locations in hong kong
words.txt : common words from http://kaifangcidian.com/xiazai/
```

Benchmark

```
Train acc =  0.944154925464
               precision    recall  f1-score   support

  china_world       0.94      0.96      0.95     33631
entertainment       0.92      0.98      0.95     33476
      finance       0.96      0.96      0.96     25134
    lifestyle       0.98      0.79      0.88     13060
         news       0.95      0.95      0.95     47032
        sport       0.99      0.64      0.77      1754

  avg / total       0.95      0.94      0.94    154087

[[32122   719   155     6   628     1]
 [  172 32948    62   114   178     2]
 [  108   404 24068    63   490     1]
 [  220  1284   139 10351  1061     5]
 [ 1413   120   612     7 44878     2]
 [   26   520    15    11    67  1115]]

Test acc =  0.944443002155
               precision    recall  f1-score   support

  china_world       0.94      0.95      0.95      8407
entertainment       0.92      0.98      0.95      8369
      finance       0.96      0.96      0.96      6283
    lifestyle       0.98      0.80      0.88      3265
         news       0.95      0.95      0.95     11757
        sport       0.99      0.65      0.79       438

  avg / total       0.95      0.94      0.94     38519

[[ 8014   180    38     5   169     1]
 [   54  8224    13    35    43     0]
 [   19    93  6025    15   131     0]
 [   62   304    32  2608   258     1]
 [  360    37   132     4 11222     2]
 [    4   121     6     7    14   286]]
```
