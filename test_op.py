from lib.network import Network


network = Network([2, 5, 1])

inputs = [
  [x, x] for x in range(10)
]

outputs = [
  [2*x] for x in range(10)
]

for _ in range(100):
  for i in range(len(inputs)):
    network.train(inputs[i], outputs[i])
    
for i in range(len(inputs)):
  print("Expected {} | Predicted {}".format(outputs[i], network.predict(inputs[i])))