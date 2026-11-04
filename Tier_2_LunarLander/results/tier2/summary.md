Hardware: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 32 CPUs, Windows | runs trained in parallel: 12 (1 CPU thread each) | budget 500,000 steps | seeds [0, 1, 2]

| Configuration | Final eval | Final 100 ep. | Solved | Best eval | Steps to threshold | Reached | Time [s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DQN (default) | 243.6 ± 11.7 | 255.6 ± 9.3 | 3/3 | 266.9 ± 7.7 | 210,000 | 3/3 | 1501 ± 618 |
| lr = 1e-4 | 152.0 ± 110.0 | 149.3 ± 105.5 | 1/3 | 219.0 ± 53.0 | 415,000 | 2/3 | 1438 ± 610 |
| lr = 1e-3 | 226.3 ± 49.4 | 232.4 ± 55.1 | 2/3 | 281.1 ± 1.3 | 140,000 | 3/3 | 1572 ± 648 |
| C = 250 | 241.0 ± 19.8 | 227.7 ± 18.7 | 3/3 | 272.0 ± 6.2 | 140,000 | 3/3 | 1538 ± 636 |
| C = 5000 | 218.6 ± 7.5 | 234.0 ± 11.6 | 3/3 | 231.1 ± 7.9 | 410,000 | 3/3 | 1071 ± 616 |
| No target network | 227.5 ± 67.6 | 228.2 ± 65.6 | 2/3 | 272.2 ± 8.4 | 180,000 | 3/3 | 1087 ± 643 |
| No replay buffer | -397.9 ± 358.0 | -399.0 ± 382.3 | 0/3 | 125.9 ± 34.2 | not reached | 0/3 | 1012 ± 606 |
| No target, no replay | 149.8 ± 61.6 | 181.8 ± 58.5 | 1/3 | 246.3 ± 3.9 | 330,000 | 3/3 | 997 ± 595 |
