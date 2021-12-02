const tf = require('@tensorflow/tfjs')

function f(x, y){
    return x+y
}

// Build and compile model.
const mlModel = tf.sequential();
mlModel.add(tf.layers.dense({units: 1, inputShape: [1]}));
mlModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await mlModel.fit(xs, ys, {epochs: 1000}); //epochs refers to the iterations in a dataset.

// Run inference with predict().
mlModel.predict(tf.tensor2d([[5]], [1, 1])).print();