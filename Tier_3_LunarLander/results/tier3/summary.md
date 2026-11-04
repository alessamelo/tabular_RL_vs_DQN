Hardware: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 32 CPUs, Windows | runs trained in parallel: 12 (1 CPU thread each) | budget 500,000 steps | seeds [0, 1, 2, 3, 4]

| Configuration | Params | Final eval | Final 100 ep. | Solved | AUC | Best eval | Steps to threshold | Reached | Bias | Time [s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DQN (vanilla) | 69,124 | 229.0 ± 37.0 | 230.0 ± 42.3 | 4/5 | 73.3 ± 30.6 | 271.7 ± 9.3 | 210,000 | 5/5 | -11.8 ± 7.6 | 1643 ± 700 |
| Double DQN | 69,124 | 187.2 ± 102.0 | 199.0 ± 91.1 | 4/5 | 76.8 ± 17.4 | 270.0 ± 8.1 | 250,000 | 5/5 | -6.3 ± 10.3 | 1709 ± 709 |
| Dueling DQN | 68,741 | 253.8 ± 29.5 | 251.1 ± 27.5 | 4/5 | 100.5 ± 8.9 | 274.8 ± 6.6 | 220,000 | 5/5 | -1.7 ± 3.3 | 1478 ± 728 |
| Double + Dueling | 68,741 | 222.8 ± 65.3 | 206.2 ± 66.5 | 4/5 | 107.3 ± 13.5 | 272.8 ± 8.5 | 190,000 | 5/5 | -12.1 ± 5.8 | 1533 ± 752 |
| Double + Dueling + PER | 68,741 | 200.9 ± 43.6 | 207.0 ± 28.5 | 3/5 | 120.1 ± 8.8 | 273.3 ± 3.1 | 160,000 | 5/5 | -5.2 ± 6.4 | 1690 ± 864 |
