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

};

/**
 * Cost function
 * @param Array Theta
 */
NN.prototype.J = function (Theta) {

};

NN.prototype.activation = function (z) {

};

/**
 * @param Array X training example
 * @return Array hypothesis
 */
NN.prototype.forwardProp = function (X) {

};

NN.prototype.backProp = function () {

};

NN.prototype.checkGradient = function () {

};

/**
 * @return Array
 */
NN.prototype.initTheta = function (epsilon) {
    epsilon = typeof epsilon === 'undefined' ? 0.12 : epsilon; 

    // count number of parameters
    var count = 0;
    for (var i = 0; i < this.S.length - 1; i++) {
        count += this.S[i] * this.S[i + 1];
    }

    // randomly initialize parameters to values in [-epsilon, epsilon)
    var Theta = [];
    for (var i = 0; i < count; i++) {
        Theta.push (Math.random () * 2 * epsilon - epsilon);
    }
    this.Theta = this.reshapeParams (Theta);
};

/**
 * @param Array unrolled parameters
 * @return Array resized parameters
 */
NN.prototype.reshapeParams = function (Theta) {
    var elements,
        elementCount,
        reshaped = [],
        Theta = math.clone (Theta);
    for (var i = 0; i < this.S.length - 1; i++) {
        elementCount = this.S[i] * this.S[i + 1];
        elements = Theta.slice (0, elementCount); 
        Theta = Theta.slice (elementCount);
        console.log ('elements = ');
        console.log (elements);
        reshaped.push (
            this.reshape (elements, [this.S[i], this.S[i + 1]]));
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
    var nn = new NN ([2, 5, 2]);
    nn.initTheta ();
    console.log ('nn.Theta = ');
    console.log (nn.Theta);
};

