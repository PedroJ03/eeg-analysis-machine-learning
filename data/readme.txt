Dataset from publication "Estimating cognitive workload using a commercial in-ear EEG headset"

DOI: https://doi.org/10.5258/SOTON/D3057

files for each participant

variable "data" has 31 channels at 200 Hz, filtered between 0.5 and 100 Hz, Notch filter at 50 Hz

1: LSL time stamps
2: ECG signal
3: hor EOG signal
4: vert EOG signal
EEG channels (not referenced yet)
5: FP1
6: F7
7: Fz
8: F3
9: FT9
10: T7
11: C3
12: Cz
13: CP1
14: TP9
15: P7
16: P3
17: Pz
18: O1
19: FP2
20: F8
21: FT10
22: F4
23: C4
24: T8
25: CP2
26: TP10
27: P4
28: P8
29: O2
30: Oz

31: Inear signal

variable "experiment" has 2 channels

1: LSL time stamps
2: experiment code:

-99: experiment starts
for the first experiment the next four elements are repeating:
first element: trial ID (0-17), start of trial (trial lasted 2.5 seconds)
second element: nback level (0-3)
third element: displayed stimulus of trial (0-9)
fourth element: participant response (0 target, 1 non-target, -1 no response)

for the second experiment the next four elements are repeating:
first element: trial ID (0-17), start of trial (trial lasted 5 (display exercise) + 3.5 (participant response) = 8.5 seconds)
second element: first number to be added (1-875)
third element: second number to be added (1-875)
fourth element: participant response (number or -1 if no response)
