#!/usr/bin/node

if (typeof require !== 'undefined') var math = require ('mathjs');

var NN = (function () {

function NN (S) {
    this.S = S; // units per layer
    this.L = this.S.length;
    this.trainingSet = [];    
    this.Theta = [];    
};

/**
 * Hypothesis function
 * @param Array X
 */
NN.prototype.h = function (X) {
    return this.forwardProp (X);
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
 * @param Array X training example
 * @return Array hypothesis
 */
NN.prototype.forwardProp = function (X, i) {
    i = typeof i === 'undefined' ? this.L - 2 : i; 
    if (i === -1) return X;
    var that = this;
//   console.log ('this.Theta[i]  = ');
//   console.log (this.Theta[i] );
//   console.log ('[1].concat (this.forwardProp (X, i - 1)) = ');
//   console.log ([1].concat (this.forwardProp (X, i - 1)));
    return this.map (
        math.multiply (
            this.Theta[i], 
            // add the bias unit
            [1].concat (this.forwardProp (X, i - 1))
        ), function (elem) {
            return that.g (elem);
        });
};

NN.prototype.backProp = function () {

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
        // test activation
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

