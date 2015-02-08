#!/usr/bin/node

if (typeof require !== 'undefined') var math = require ('mathjs');

var NN = (function () {

/**
 * @param Array S units per layer
 */
function NN (S) {
    this.S = S; 
    this.L = this.S.length; // number of layers
    this.trainingSet = [];    
    this.Theta = []; // parameters   
    this.lambda = 0.001; // regularization term
    this.a = []; // activations
};

/**
 * Hypothesis function
 * @param Array X
 */
NN.prototype.h = function (X) {
    return this.forwardProp (X);
};

/**
 * Get regularization term of cost function
 */
NN.prototype.getRegularizationTerm = function () {
    return math.multiply (
        this.lambda / (2 * this.trainingSet.length),
        math.sum (math.square (this.unrollParams (true)))
    );
};

/**
 * Cost function
 * @param Array Theta
 */
NN.prototype.J = function () {
    var cost = 0, 
        hypothesis,
        ex,
        x,
        y;
    for (var i in this.trainingSet) {
        ex = this.trainingSet[i];
        x = ex[0];
        y = ex[1];
        hypothesis = this.h (x);
        cost = math.add (
            cost,
            math.add (
                math.multiply (
                    y,
                    math.log (
                        hypothesis
                    )
                ),
                math.multiply (
                    math.subtract (
                        1,
                        y
                    ),
                    math.log (
                        math.subtract (
                            1,
                            hypothesis
                        )
                    )
                )
            )
        );
    }
    cost = math.multiply (
        -(1 / this.trainingSet.length),
        cost
    );
    cost = math.add (
        cost,
        this.getRegularizationTerm ()
    );
    return cost;
};

/**
 * Sigmoid activation
 */
NN.prototype.g = function (X, Theta) {
    if (X instanceof Array) {
        return 1 / (1 + Math.pow (
            Math.E, 
            -math.multiply (
                Theta, 
                X
            )));
    } else {
        return 1 / (1 + Math.pow (Math.E, -X));
    }
};

/**
 * Map helper which coerces argument to array
 */
NN.prototype.map = function (x, callback) {
    x = x instanceof Array ? x : [x];
    return math.map (x, callback); 
};

/**
 * Forward propagate input vector through neural network, saving 
 * activations of each layer of neurons in the property a. Saved
 * activation values can are used by back propagation.
 * @param Array X training example
 * @return Array hypothesis
 */
NN.prototype.forwardProp = function (X, i) {
    i = typeof i === 'undefined' ? this.L - 2 : i; 
    if (i === -1) {
        this.a[i + 1] = X;
    } else {
        var that = this;
    //   console.log ('this.Theta[i]  = ');
    //   console.log (this.Theta[i] );
    //   console.log ('[1].concat (this.forwardProp (X, i - 1)) = ');
    //   console.log ([1].concat (this.forwardProp (X, i - 1)));
        this.a[i + 1] = this.map (
            math.multiply (
                this.Theta[i], 
                this.forwardProp (X, i - 1)
            ), function (elem) {
                return that.g (elem);
            });
    }
    if (i !== this.L - 2) {
        // add bias activation value
        this.a[i + 1] = [1].concat (this.a[i + 1]);
    }
    return this.a[i + 1];
};

/**
 * Helpfer function for backProp which recursively calculates error 
 * terms for each neuron
 */
NN.prototype.getErrorTerms = function (y, delta, i) {
//    console.log ('delta = ');
//    console.log (delta);
    if (typeof i === 'undefined') {
        i = this.S.length - 1;
//        console.log ('this.a[i] = ');
//        console.log (this.a[i]);
        delta = [math.subtract (
            this.a[i],
            y
        )];
    } else if (i === 0) {
        return delta;
    } else {
//        console.log ('math.transpose (this.Theta[i]) = ');
//        console.log (math.transpose (this.Theta[i]));
        delta = [math.multiply (
            math.multiply (
                math.transpose (this.Theta[i]),
                i === this.S.length - 2 ? 
                    delta[0] :
                    delta[0].slice (1) // remove bias error unit
            ),
            math.multiply (
                this.a[i],
                math.subtract (
                    1,
                    this.a[i]
                )
            )
        )].concat (delta);
    }
//    console.log ('i = ');
//    console.log (i);
    return this.getErrorTerms (y, delta, i - 1);
};

NN.prototype.backProp = function () {
    // initialize partial derivative values at 0
    var Delta = [];
    for (var i = 0; i < this.S.length - 1; i++) {
        Delta.push ([]);
        for (var j = 0; j < this.S[i + 1]; j++) {
            Delta[i].push ([]);
            for (var k = 0; k < this.S[i] + 1; k++) {
                Delta[i][j].push (0);
            }
        }
    }
    //console.log ('Delta = ');
    //console.log (Delta);

    var ex, delta;
    for (var i in this.trainingSet) {
        ex = this.trainingSet[i];
        this.forwardProp (ex[0]); // calculate activation values
//        console.log ('this.a = ');
//        console.log (this.a);
//        console.log ('ex[1] = ');
//        console.log (ex[1]);
        delta = this.getErrorTerms (ex[1]);
//        console.log ('delta = ');
//        console.log (delta);
        for (var j = 0; j < this.S.length - 1; j++) {
            Delta[j] = math.add (
                Delta[j],
                math.multiply (
                    delta[j],
                    math.transpose (this.a[j + 1])
                )
            );
        }
        //console.log ('Delta = ');
        //console.log (Delta);
    }
    //console.log ('Delta = ');
    //console.log (Delta);
    for (var i = 0; i < this.S.length - 1; i++) {
        Delta[i] = math.add (
            math.multiply (
                1 / this.trainingSet.length,
                Delta[i]
            ),
            math.multiply (
                this.lambda,
                this.Theta[i].map (function (row) { // remove bias params
                    return [0].concat (row.slice (1));
                })
            )
        );
    }
    //console.log ('Delta = ');
    //console.log (Delta);
    return Delta;
};

NN.prototype.checkGradient = function () {

};

/**
 * Initialize parameters to random values in [-epsilon, epsilon)
 * @return Array
 */
NN.prototype.initTheta = function (epsilon) {
    epsilon = typeof epsilon === 'undefined' ? 0.12 : epsilon; 

    // count number of parameters
    var count = 0;
    for (var i = 0; i < this.S.length - 1; i++) {
        count += (this.S[i] + 1) * this.S[i + 1];
    }

    // randomly initialize parameters
    var Theta = [];
    for (var i = 0; i < count; i++) {
        Theta.push (Math.random () * 2 * epsilon - epsilon);
    }
    this.Theta = this.reshapeParams (Theta);
};

NN.prototype.unrollParams = function (excludeBiasUnits) {
    excludeBiasUnits = typeof excludeBiasUnits === 'undefined' ? 
        false : excludeBiasUnits; 
    var unrolled = [];
    for (var i in this.Theta) {
        for (var j in this.Theta[i]) {
            if (excludeBiasUnits && j === 0) continue;
            unrolled.push (this.Theta[i][j]); 
        }
    }
    return unrolled;
};

/**
 * Convert vector of parameters into parameter matrices
 * @param Array unrolled parameters
 * @return Array resized parameters
 */
NN.prototype.reshapeParams = function (Theta) {
    var elements,
        elementCount,
        reshaped = [],
        Theta = math.clone (Theta);
    for (var i = 0; i < this.S.length - 1; i++) {
        elementCount = (this.S[i] + 1) * this.S[i + 1];
        elements = Theta.slice (0, elementCount); 
        Theta = Theta.slice (elementCount);
        reshaped.push (
            this.reshape (elements, [this.S[i + 1], this.S[i] + 1]));
    }
    return reshaped;
};

/**
 * Reshapes vector into matrix with specified dimensions
 * @param Array arr
 * @param Array dimensions (e.g. [3, 5])
 * @return Array
 */
NN.prototype.reshape = function (arr, dimensions) {
    var reshaped = []; 
    for (var i = 0; i < dimensions[0]; i++) {
            reshaped.push ([]);
        for (var j = 0; j < dimensions[1]; j++) {
            reshaped[i].push (arr[i * dimensions[1] + j]);
        }
    }
    return reshaped;
};

NN.prototype.gradientDescent = function () {

};


return NN;

}) ();

if (typeof module !== 'undefined') module.exports = NN;

GLOBAL.test = function () {

    // test back prop
    (function () {
        var nn = new NN ([2, 2, 1]); // XNOR network
        nn.Theta = [
            [[-30, 20, 20], [10, -20, -20]],
            [[-10, 20, 20]]
        ];
        nn.trainingSet = [
            [[0, 0], 1],
            [[0, 1], 0],
            [[1, 0], 0],
            [[1, 1], 1],
        ];
        nn.backProp ();
    }) ();
    return;

    // test cost function
    (function () {
        var nn = new NN ([2, 1]);
        nn.Theta = [[-30, 20, 20]]; // AND params
        nn.trainingSet = [
            [[0, 0], 0],
            [[0, 1], 0],
            [[1, 0], 0],
            [[1, 1], 1],
        ];
        console.log (nn.J ());
        nn.Theta = [[10, -20, -20]]; // NAND params
        console.log (nn.J ());
        nn.Theta = [[-10, 20, 20]]; // OR params
        console.log (nn.J ());
    }) ();

    return;

    // basic functionality test
    (function () {
        // test param initialization
        var nn = new NN ([2, 2, 1]);
        nn.initTheta ();
        console.log ('nn.Theta = ');
        console.log (nn.Theta);
        // test activation function
        console.log (nn.g ([1, 1, 1], nn.Theta[0][0]));
    } ());

    // AND network
    (function () {
        var nn = new NN ([2, 1]);
        nn.Theta = [[-30, 20, 20]];
        console.log (nn.forwardProp ([0, 0]));
        console.log (nn.forwardProp ([0, 1]));
        console.log (nn.forwardProp ([1, 0]));
        console.log (nn.forwardProp ([1, 1]));
    }) ();

    // NAND network
    (function () {
        var nn = new NN ([2, 1]);
        nn.Theta = [[10, -20, -20]];
        console.log (nn.forwardProp ([0, 0]));
        console.log (nn.forwardProp ([0, 1]));
        console.log (nn.forwardProp ([1, 0]));
        console.log (nn.forwardProp ([1, 1]));
    }) ();

    // OR network
    (function () {
        var nn = new NN ([2, 1]);
        nn.Theta = [[-10, 20, 20]];
        console.log (nn.forwardProp ([0, 0]));
        console.log (nn.forwardProp ([0, 1]));
        console.log (nn.forwardProp ([1, 0]));
        console.log (nn.forwardProp ([1, 1]));
    }) ();

    // XNOR network
    (function () {
        // test param initialization
        var nn = new NN ([2, 2, 1]);
        nn.Theta = [
            [[-30, 20, 20], [10, -20, -20]],
            [[-10, 20, 20]]
        ];
        console.log (nn.forwardProp ([0, 0]));
        console.log (nn.forwardProp ([0, 1]));
        console.log (nn.forwardProp ([1, 0]));
        console.log (nn.forwardProp ([1, 1]));
    }) ();
};

