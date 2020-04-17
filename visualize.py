import csv
import matplotlib.pyplot as plt

steps_rnn = []
losses_rnn = []
steps_bsde = []
losses_bsde = []

with open('logs/rnn_training_history.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] == 'step':
            continue
        steps_rnn.append(int(row[0]))
        losses_rnn.append(float(row[1]))

with open('logs/test_training_history.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] == 'step':
            continue
        steps_bsde.append(int(row[0]))
        losses_bsde.append(float(row[1]))


plt.semilogy(steps_rnn, losses_rnn, label='RNN')
plt.semilogy(steps_bsde, losses_bsde, label='DeepBSDE')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Comparison between RNN and DeepBSDE')
plt.legend()
plt.savefig('img/loss_comparison.png')
