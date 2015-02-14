
# NN.js

A feedforward neural network class


## Installation

```shell
npm install
```

## Usage

```js
// train an XNOR network
var nn = new NN ([2, 2, 1]); 
nn.enableGradientChecking = false;
nn.trainingSet = [
    [[0, 0], 1],
    [[0, 1], 0],
    [[1, 0], 0],
    [[1, 1], 1],
];
var Theta = nn.gradientDescent (10000, 10);
var h = nn.getH (Theta);
assert (h ([0, 0])[0] >= 0.5);
assert (h ([0, 1])[0] < 0.5);
assert (h ([1, 0])[0] < 0.5);
assert (h ([1, 1])[0] >= 0.5);
```

