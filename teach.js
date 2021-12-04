const tf = require('@tensorflow/tfjs-node')
const mtx = require('./matrix.json')

f = (arg) => Number(Math.sin(arg).toFixed(3)) //Синус вычисляется по радианам
rad = (arg) => ((arg)*Math.PI)/180  //Перевод из градусов в радианы - есть погрешность
const train_arr = [0, 0.259, 0.5, 0.707, 0.866, 0.966, 1, 0.966, 0.866, 0.707, 0.5, 0.259, 0, -0.259, -0.5, -0.707, -0.866, -0.966, -1, -0.966, -0.866, -0.707, -0.5, -0.259, 0]

let mt = false
let rd = false
let sn = false

// Build and compile model.
const mlModel_sinus = tf.sequential({
    layers: [tf.layers.dense({ units: 1, inputShape: 1 }), tf.layers.dense({ units: 40, inputShape: 40 }), tf.layers.dense({ units: 1, inputShape: 1 })],
});
mlModel_sinus.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const mlModel_rad = tf.sequential({
    layers: [tf.layers.dense({ units: 1, inputShape: 1 }), tf.layers.dense({ units: 40, inputShape: 40 }), tf.layers.dense({ units: 1, inputShape: 1 })],
  });
mlModel_rad.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const mlModel_number = tf.sequential({
    layers: [tf.layers.dense({ units: 1, inputShape: 1 }), tf.layers.dense({ units: 10, inputShape: 10 }), tf.layers.dense({ units: 1, inputShape: 1 })],
  });
mlModel_number.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

let xs, ys

function arr_str(arr) {
    let str = ''
    arr.forEach(el => {
        str+=el
    });
    return Number(str)
}

    init_sinus = async (n = 96) => {
        let i = n/-2
        let j=0;
        let arr_1 = []
        let arr_2 = []
        // Generate some synthetic data for training.
        // Шаг 15 градусов
        while(i < n/2) {
            arr_1.push([j])
            arr_2.push([f(rad(i*15))])
            i++
            j++
        }
        xs = tf.tensor2d(arr_1, [n, 1]);
        ys = tf.tensor2d(arr_2, [n, 1]);
    }
    execute_sinus = async (k, param = 17) => {
        k = Number(k)
        // Train model with fit().
        await mlModel_sinus.fit(xs, ys, {epochs: 2})
        let r = await mlModel_sinus.predict(tf.tensor2d([[k]], [1, 1])).array()
        console.log(`real=${f(rad(k*15))} predicted = ${r[0]}`)
        if(r[0] != Number(r[0])){
            return
        }
        if(train_arr[k%25].toFixed(3) == Number(r[0]).toFixed(3)){
            sn = true
            mlModel_sinus.save('file://./models/model_sinus')
        }
    }
    init_radian = async (n=25) => {
        let i = 0
        let arr_1 = []
        let arr_2 = []
        // Generate some synthetic data for training.
        while(i < n) {
            let s = i*15
            arr_1.push([s])
            arr_2.push([rad(s)])
            i++
        }
        xs = tf.tensor(arr_1);
        ys = tf.tensor(arr_2);
    }
    execute_radian = async (k, param = 17) => {
        k = Number(k)
        // Train model with fit().
        await mlModel_rad.fit(xs, ys, {epochs: 1000})
        let r = await mlModel_rad.predict(tf.tensor2d([[k]], [1, 1])).array()
        console.log(`real=${rad(k).toFixed(3)} predicted = ${r[0]}`)
        if(r[0] != Number(r[0])){
            return
        }
        if(rad(k).toFixed(3) == Number(r[0]).toFixed(3)){
            rd = true
            mlModel_rad.save('file://./models/model_rad')
        }
    }
    init_matrix =(n) => {
        let i = 0
        let arr_1 = []
        let arr_2 = []
        // Generate some synthetic data for training.
        while(i < n) {
            let s = Number((Math.random()*1000).toFixed(2))
            arr_1.push([arr_str(mtx[i].matrix)])
            arr_2.push([mtx[i].value])
            i++
        }
        xs = tf.tensor2d(arr_1, [n, 1]);
        ys = tf.tensor2d(arr_2, [n, 1]);
    }
    execute_matrix = async (k, param = mtx.length) => {
        k = Number(k)
        // Train model with fit().
        await mlModel_number.fit(xs, ys, {epochs: 2000})
        let r = await mlModel_number.predict(tf.tensor2d([[k]], [1, 1])).array()
        console.log(`predicted = ${r[0]}`)
        if(mtx[5].val == r[0]){
            mt = true
            mlModel_number.save('file://./models/model_number')
        }
    }


function final() {
    console.log(`\n----------------------------------------------`)
    console.log(`Sinus training - ${sn?'success':'failed'}`)
    console.log(`Radian training - ${rd?'success':'failed'}`)
    console.log(`Matrix training - ${mt?'success':'failed'}`)
}

async function teach(){
    await init_sinus()
    await execute_sinus(1)

    // await init_radian()
    // await execute_radian(0)

    // await init_matrix()
    // await execute_matrix(arr_str(mtx[5].matrix))

    final()
}

teach()