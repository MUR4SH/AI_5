const tf = require('@tensorflow/tfjs')
const mtx = require('./matrix.json')

f = (arg) => Number(Math.sin(arg).toFixed(2)) //Синус вычисляется по радианам
rad = (arg) => ((arg)*Math.PI)/180  //Перевод из градусов в радианы - есть погрешность
const train_arr = [0, 0.259, 0.5, 0.707, 0.866, 0.966, 1, 0.966, 0.866, 0.707, 0.5, 0.259, 0, -0.259, -0.5, -0.707, -0.866, -0.966, -1, -0.966, -0.866, -0.707, -0.5, -0.259, 0]

// Build and compile model.
const mlModel_sinus = tf.sequential({
    layers: [tf.layers.dense({ units: 1, inputShape: 1 }), tf.layers.dense({ units: 10, inputShape: 10 }), tf.layers.dense({ units: 1, inputShape: 1 })],
  });
mlModel_sinus.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const mlModel_rad = tf.sequential({
    layers: [tf.layers.dense({ units: 1, inputShape: 1 }), tf.layers.dense({ units: 10, inputShape: 10 }), tf.layers.dense({ units: 1, inputShape: 1 })],
  });
mlModel_rad.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const mlModel_number = tf.sequential({
    layers: [tf.layers.dense({ units: 1, inputShape: 1 }), tf.layers.dense({ units: 10, inputShape: 10 }), tf.layers.dense({ units: 1, inputShape: 1 })],
  });
mlModel_number.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

module.exports = {
    execute_sinus: async (k, param = 17) => {
        k = Number(k)
        await mlModel_sinus.loadWeights('file://./models/model_sin')
        let r = await mlModel_sinus.predict(tf.tensor2d([[k]], [1, 1])).array()
        console.log(`real=${train_arr[k%25]} predicted = ${r[0]}`)
        return r[0]
    },
    execute_radian: async (k, param = 17) => {
        k = Number(k)
        await mlModel_rad.loadWeights('file://./models/model_rad')
        let r = await mlModel_rad.predict(tf.tensor2d([[k]], [1, 1])).array()
        console.log(`predicted = ${r[0]}`)
        return r[0]
    },
    execute_matrix: async (k, param = 17) => {
        k = Number(k)
        await mlModel_number.loadWeights('file://./models/model_number')
        let r = await mlModel_number.predict(tf.tensor2d([[k]], [1, 1])).array()
        console.log(`predicted = ${r[0]}`)
        return r[0]
    }
}