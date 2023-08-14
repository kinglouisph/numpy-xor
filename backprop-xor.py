import numpy as np

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

outputs = np.array([
    [0],
    [1],
    [1],
    [0]
])

def activation(n):
    return (n > 0) * n

def activationd(n):
    return (n > 0) + 0

hiddenSize = 6

weights1 = np.random.rand(hiddenSize, 2)*2-1 #input to hidden
weights2 = np.random.rand(1, hiddenSize)*2-1 #hidden to output

alpha = 0.1

for iteration in range(300):
    error = 0
    for i in range(4):
        out1 = activation(np.matmul(weights1, inputs[i]))
        out2 = (np.matmul(weights2, out1))[0] # the nn output

        layer2Delta = (outputs[i] - out2)[0]
        layer1Delta = np.multiply(weights2 * layer2Delta, activationd(out1))
               
        error += layer2Delta ** 2
        #print(layer2Delta, layer1Delta, weights2)
        #print((out1, layer2Delta, weights2))

        weights1 += alpha * inputs[i:i+1] * layer1Delta.T
        weights2 += alpha * out1 * layer2Delta

    if iteration % 10 == 0:
        print(error)

while 1:
    a = float(input("1: "))
    b = float(input("2: "))
    out1 = activation(np.matmul(weights1, np.array([a,b])))
    out2 = (np.matmul(weights2, out1))[0] # the nn output
    print(out2)

