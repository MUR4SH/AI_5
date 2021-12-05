const tft = require('@tensorflow/tfjs-node')

const size = 32

// Build and compile model.
let mlModel = tf.sequential();
mlModel.add(tf.layers.dense({ units: 1, inputShape: [1], outputShape: [size] }));
mlModel.add(tf.layers.dense({ units: size, inputShape: [size], outputShape: [size] }));
mlModel.add(tf.layers.dense({ units: 1, inputShape: [size], outputShape: [1] }));
mlModel.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: 'Adam',
    metrics: [tf.metrics.MSE]
});

module.exports = {
    execute_sinus: async (k) => {
        k = Number(k)
        console.log('\nloading model')
        mlModel = await tf.loadLayersModel('file://./models/model_sinus/model.json', 'file://./models/model_sinus/weights.bin')
        console.log('\nmodel sucessfully loaded')
        let r = await mlModel.predict(tf.tensor([k])).array()
        console.log(`predicted = ${r[0]}`)
        return r[0]
    },
    execute_radian: async (k) => {
        k = Number(k)
        console.log('\nloading model')
        mlModel = await tf.loadLayersModel('file://./models/model_rad/model.json', 'file://./models/model_rad/weights.bin')
        console.log('\nmodel sucessfully loaded')
        let r = await mlModel.predict(tf.tensor([k])).array()
        console.log(`predicted = ${r[0]}`)
        return r[0]
    },
    execute_matrix: async (k) => {
        k = Number(k)
        console.log('\nloading model')
        mlModel = await tf.loadLayersModel('file://./models/model_number/model.json', 'file://./models/model_number/weights.bin')
        console.log('\nmodel sucessfully loaded')
        let r = await mlModel.predict(tf.tensor([k])).array()
        console.log(`predicted = ${r[0]}`)
        return r[0]
    }
}