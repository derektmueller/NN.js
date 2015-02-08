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
    this.lambda = 0.001; // regularization term
    this.a = []; // activations
    this.enableRegularization = false;
};

/**
 * Hypothesis function getter
 * @param Array X
 */
NN.prototype.getH = function (Theta) {
    var that = this;
    return function (X) {
        return that.forwardProp (Theta, X);
    };
};

/**
 * Get regularization term of cost function
 */
NN.prototype.getRegularizationTerm = function (Theta) {
    return math.multiply (
        this.lambda / (2 * this.trainingSet.length),
        math.sum (math.square (this.unrollParams (Theta, true)))
    );
};

/**
 * Cost function
 * @param Array Theta
 */
NN.prototype.J = function (Theta) {
    var cost = 0, 
        h = this.getH (Theta),
        hVal,
        ex,
        x,
        y;
    for (var i in this.trainingSet) {
        ex = this.trainingSet[i];
        x = ex[0];
        y = ex[1];
        hVal = h (x);
        cost = math.add (
            cost,
            math.sum (math.add (
                math.multiply (
                    y,
                    math.log (
                        hVal
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
                            hVal
                        )
                    )
                )
            ))
        );
    }
    cost = math.multiply (
        -(1 / this.trainingSet.length),
        cost
    );
    if (this.enableRegularization)
        cost = math.add (
            cost,
            this.getRegularizationTerm (Theta)
        );
    return cost;
};

/**
 * Sigmoid activation
 */
NN.prototype.g = function (X, Theta) {
    if (X instanceof Array) {
        return math.divide (
            1,
            math.add (
                1, 
                math.pow (
                    Math.E, 
                    -math.multiply (
                        Theta, 
                        X
                    )
                )
            )
        );
    } else {
        return math.divide (
            1,
            math.add (
                1, 
                math.pow (
                    Math.E, 
                    -X
                )
            )
        );
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
NN.prototype.forwardProp = function (Theta, X, i) {
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
                Theta[i], 
                this.forwardProp (Theta, X, i - 1)
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
NN.prototype.getErrorTerms = function (Theta, y, delta, i) {
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
                math.transpose (Theta[i]),
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
    return this.getErrorTerms (Theta, y, delta, i - 1);
};

NN.prototype.backProp = function (Theta) {
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
        this.forwardProp (Theta, ex[0]); // calculate activation values
        console.log ('this.a = ');
        console.log (this.a);
        console.log ('ex[1] = ');
        console.log (ex[1]);
        delta = this.getErrorTerms (Theta, ex[1]);
//        console.log ('delta = ');
//        console.log (delta);
        for (var j = 0; j < this.S.length - 1; j++) {
//            console.log ('j = ');
//            console.log (j);
//            console.log ('Delta[j]');
//            console.log (Delta[j]);
//           console.log ('delta[j] = ');
//            console.log (delta[j].slice (j === this.S.length - 2 ? 0 : 1).
//                map (function (elem) {
//                    return [elem];
//                }));
            Delta[j] = math.add (
                Delta[j],
                math.multiply (
                    delta[j].slice (j === this.S.length - 2 ? 0 : 1).
                        map (function (elem) {
                            return [elem];
                        }),
                    [this.a[j]]
                )
            );
        }
        //console.log ('Delta = ');
        //console.log (Delta);
    }
    //console.log ("\n");
    //console.log ('Delta = ');
    //console.log (Delta);
    for (var i = 0; i < this.S.length - 1; i++) {
        if (this.enableRegularization) {
            Delta[i] = math.add (
                math.multiply (
                    1 / this.trainingSet.length,
                    Delta[i]
                ),
                math.multiply (
                    this.lambda,
                    Theta[i].map (function (row) { // remove bias params
                        return [0].concat (row.slice (1));
                    })
                )
            );
        } else {
            Delta[i] = math.multiply (
                1 / this.trainingSet.length,
                Delta[i]
            );
        }
    }
    //console.log ('Delta = ');
    //console.log (Delta);
    return Delta;
};

/**
 * Calculate gradient approximation
 */
NN.prototype.gradApprox = function (Theta, epsilon) {
    epsilon = typeof epsilon === 'undefined' ? 0.0001 : epsilon; 
    var unrolled = this.unrollParams (Theta),
        gradApprox = [],
        thetaPlus,
        thetaMinus;
   //console.log ('unrollParams = ');
   //console.log (unrolled);

    for (var i in unrolled) {
        thetaPlus = unrolled.slice ()
        thetaMinus = unrolled.slice ()
        thetaPlus[i] = math.add (thetaPlus[i], epsilon);
        thetaMinus[i] = math.subtract (thetaMinus[i], epsilon);
//        console.log (this.reshapeParams (thetaPlus));
//        console.log (this.reshapeParams (thetaMinus));
//        console.log (this.J (this.reshapeParams (thetaPlus)));
//        console.log (this.J (this.reshapeParams (thetaMinus)));
        gradApprox[i] = math.divide (
            math.subtract (
                this.J (this.reshapeParams (thetaPlus)),
                this.J (this.reshapeParams (thetaMinus))
            ),
            2 * epsilon
        );
    }
    //console.log ('gradApprox = ');
    //console.log (gradApprox);
    return this.reshapeParams (gradApprox);
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
    return this.reshapeParams (Theta);
};

NN.prototype.unrollParams = function (Theta, excludeBiasUnits) {
    excludeBiasUnits = typeof excludeBiasUnits === 'undefined' ? 
        false : excludeBiasUnits; 
    var unrolled = [];
    for (var i in Theta) {
        for (var j in Theta[i]) {
            for (var k in Theta[i][j]) {
                if (excludeBiasUnits && k === 0) continue;
                unrolled.push (Theta[i][j][k]); 
            }
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
        var Theta = [
            [[-30, 20, 20], [10, -20, -20]],
            [[-10, 20, 20]]
        ];
        //var Theta = nn.initTheta ();
        nn.trainingSet = [
//            [[20, 0], 0],
//            [[20, 5], 0],
//            [[25, 5], 0],
//            [[0, 0], 0],
            [[0, 0], 1],
            [[0, 1], 0],
            [[1, 0], 0],
            [[1, 1], 1],
        ];
        console.log (nn.backProp (Theta));
        console.log (nn.gradApprox (Theta));
    }) ();
    return;

//    // test cost function
//    (function () {
//        var nn = new NN ([2, 1]);
//        nn.trainingSet = [
//            [[0, 0], 0],
//            [[0, 1], 0],
//            [[1, 0], 0],
//            [[1, 1], 1],
//        ];
//        var Theta = [[[-30, 20, 20]]]; // AND params
//        console.log (nn.J (Theta));
//        Theta = [[[10, -20, -20]]]; // NAND params
//        console.log (nn.J (Theta));
//        Theta = [[[-10, 20, 20]]]; // OR params
//        console.log (nn.J (Theta));
//    }) ();
//    return;

//    // basic functionality test
//    (function () {
//        // test param initialization
//        var nn = new NN ([2, 2, 1]);
//        var Theta = nn.initTheta ();
//        console.log ('Theta = ');
//        console.log (Theta);
//        // test activation function
//        console.log (nn.g ([1, 1, 1], Theta[0][0]));
//    } ());
//    return;

    // AND network
    (function () {
        var nn = new NN ([2, 1]);
        var Theta = [[-30, 20, 20]];
        console.log (nn.forwardProp (Theta, [0, 0]));
        console.log (nn.forwardProp (Theta, [0, 1]));
        console.log (nn.forwardProp (Theta, [1, 0]));
        console.log (nn.forwardProp (Theta, [1, 1]));
    }) ();
    return;

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

