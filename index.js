const tf = require('@tensorflow/tfjs')

f = (arg) => Math.sin(arg) //Синус вычисляется по радианам
rad = (arg) => (arg*3.14)/180  //Перевод из градусов в радианы - есть погрешность

// Build and compile model.
const mlModel = tf.sequential();
mlModel.add(tf.layers.dense({units: 1, inputShape: [1]}));
mlModel.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

let xs, ys

init = (n)=>{
    let i = 0
    let arr_1 = []
    let arr_2 = []
    let train_arr = [0, 0.259, 0.5, 0.707, 0.866, 0.966, 1, 0.966, 0.866, 0.707, 0.5, 0.259, 0, -0.259, -0.5, -0.707, -0.866, -0.966, -1, -0.966, -0.866, -0.707, -0.5, -0.259, 0, 0.259]
    // Generate some synthetic data for training.
    // 0.707 0.866 0.966 1 0.966 0.866 0.707 0.5 0.259 0 – 0.259
    // Шаг 15 градусов
    while(i < n) {
        let k = rad(15*i)
        arr_1.push([i])
        arr_2.push([train_arr[i]])
        i++
    }
    xs = tf.tensor2d(arr_1, [n, 1]);
    ys = tf.tensor2d(arr_2, [n, 1]);

    console.log(arr_1)
    console.log(arr_2)
    ys.print()
    xs.print()
}

init(18) //24 раза по 15 = 360 + 25 раз, считаем с нуля
function train_execute(k){
    // Train model with fit().
    mlModel.fit(xs, ys, {epochs: 2000}).then(()=>{
        mlModel.predict(tf.tensor2d([[k]], [1, 1])).array().then((r)=>{
            console.log(r)
            console.log(`real = ${f(rad(k*15))}, predicted = ${r[0]}`)
        })
    });
}

train_execute(1)