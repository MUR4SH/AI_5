const tf = require('@tensorflow/tfjs')

f = (arg) => Math.sin(arg) //Синус вычисляется по радианам
rad = (arg) => (arg*Math.PI)/180  //Перевод из градусов в радианы - есть погрешность

// Build and compile model.
const mlModel = tf.sequential();

let xs, ys

module.exports = {
    init_sinus: (n) => {
        //24 раза по 15 = 360 + 25 раз, считаем с нуля
        let i = 0
        let arr_1 = []
        let arr_2 = []
        let train_arr = [0, 0.259, 0.5, 0.707, 0.866, 0.966, 1, 0.966, 0.866, 0.707, 0.5, 0.259, 0, -0.259, -0.5, -0.707, -0.866, -0.966, -1, -0.966, -0.866, -0.707, -0.5, -0.259, 0, 0.259]
        // Generate some synthetic data for training.
        // 0.707 0.866 0.966 1 0.966 0.866 0.707 0.5 0.259 0 – 0.259
        // Шаг 15 градусов
        while(i < n) {
            arr_1.push([i])
            arr_2.push([f(rad(Number(i)*15))])
            i++
        }
        xs = tf.tensor2d(arr_1, [n, 1]);
        ys = tf.tensor2d(arr_2, [n, 1]);
    },
    execute_sinus: async (k, param = 52) => {
        module.exports.init_sinus(param) 

        mlModel.add(tf.layers.dense({units: 1, inputShape: [1]}));
        mlModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
        // Train model with fit().
        await mlModel.fit(xs, ys, {epochs: 10000})
        let r = await mlModel.predict(tf.tensor2d([[Number(k)]], [1, 1])).array()
        console.log(`real = ${f(rad(Number(k)*15))}, predicted = ${r[0]}`)
        return r[0]
    }
}