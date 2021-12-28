import numpy as np
import numpy

y_pred = [[914, 352, 160],
         [644, 86, 53],
         [1248, 348, 30],
         [50, 654, 38]]
y_true = [[1249, 344, 30], [911, 349, 160], [644, 86, 50]]
y_pred_r = [0, 0, 0, 0, 0, 0]
for i in range(0, len(y_pred)):
    list_distance = []
    for index in range(0, len(y_true)):
        distance = numpy.sqrt(
            ((y_pred[i][0] - y_true[index][0]) ** 2) + ((y_pred[i][1] - y_true[index][1]) ** 2))
        list_distance.append(distance)
    if min(list_distance) <= 10:  # d<10
        if y_pred_r[list_distance.index(min(list_distance))] == 0:
            y_pred_r[list_distance.index(min(list_distance))] = y_pred[i][2]
        else:
            smaller_r = min(y_pred_r[list_distance.index(min(list_distance))], y_pred[i][2])
            y_pred_r[list_distance.index(min(list_distance))] = smaller_r
count = 0
for m in range(0, len(y_pred_r)):
    if y_pred_r[m] == 0:
        count = count + 1
for n in range(0, count):
    y_pred_r.remove(0)
print(f"y_pred_r: {y_pred_r}")

y_true_r = []
for i in range(0, len(y_true)):
    y_true_r.append(y_true[i][2])
print(f"y_true_r: {y_true_r}")