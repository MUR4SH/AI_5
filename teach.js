const tf = require('@tensorflow/tfjs-node')
const mtx = require('./matrix.json')

f = (arg) => Number(Math.sin(arg).toFixed(3)) //Синус вычисляется по радианам
rad = (arg) => Number((((arg)*Math.PI)/180).toFixed(3)) //Перевод из градусов в радианы - есть погрешность
const train_arr = [0, 0.259, 0.5, 0.707, 0.866, 0.966, 1, 0.966, 0.866, 0.707, 0.5, 0.259, 0, -0.259, -0.5, -0.707, -0.866, -0.966, -1, -0.966, -0.866, -0.707, -0.5, -0.259]

let mt = false
let rd = false
let sn = false

const size = 32

let mlModel = tf.sequential();
mlModel.add(tf.layers.dense({ units: 1, inputShape: [1], outputShape: [size] }));
mlModel.add(tf.layers.dense({ units: size, inputShape: [size], outputShape: [size] }));
mlModel.add(tf.layers.dense({ units: 1, inputShape: [size], outputShape: [1] }));

/*
radian - Adam optimizer, meanSquaredError loss, metrics - MSE
sinus - Adam optimizer, meanSquaredError loss, metrics - MSE
number - Adam optimizer, meanSquaredError loss, metrics - MSE
*/
let xs, ys

function arr_str(arr) {
    let str = ''
    arr.forEach(el => {
        str+=el
    });
    return Number(str)
}

    init_sinus = async (n) => {
        let i = 0
        let arr_1 = []
        let arr_2 = []
        // Generate some synthetic data for training.
        while(i < n) {
            let s = i
            arr_1.push(Number(s))
            arr_2.push(Number(train_arr[s%train_arr.length]))
            i++
        }
        xs = tf.tensor(arr_1);
        ys = tf.tensor(arr_2);
        console.log('\nTrain data for sinus initiated...\n')
    }
    load_sinus = async () => {
        console.log('\nLoading model\n')

        mlModel = await tf.loadLayersModel('file://./models/model_sinus/model.json', 'file://./models/model_sinus/weights.bin')

        console.log('\nModel succesfully loaded\n')
    }
    train = async (epoch) =>{
        console.log('\nInitiate learning\n')
        // Train model with fit().
        await mlModel.fit(xs, ys, {
            batchSize: Math.pow(2,6),
            epochs: epoch,
        })
        console.log('\nLearning completed\n')
    }
    execute_sinus = async (k) => {
        k = Number(k)
        let r = await mlModel.predict(tf.tensor([k])).array()
        console.log(`real=${train_arr[k%train_arr.length]} predicted = ${Number(r[0]).toFixed(3)}`)
        if(r[0] != Number(r[0])){
            return false
        }
        if(train_arr[k%train_arr.length] == Number(r[0]).toFixed(3)){
            return true
        }
    }
    init_radian = async (n = 25) => {
        let i = n/-2
        let arr_1 = []
        let arr_2 = []
        // Generate some synthetic data for training.
        while(i < n/2) {
            let s = i
            arr_1.push(Number(s))
            arr_2.push(Number(rad(s)))
            i++
        }
        xs = tf.tensor(arr_1);
        ys = tf.tensor(arr_2);
        console.log('\nTrain data for radian initiated...\n')
    }
    load_radian = async () => {
        console.log('\nLoading model\n')

        mlModel = await tf.loadLayersModel('file://./models/model_rad/model.json', 'file://./models/model_rad/weights.bin')

        console.log('\nModel succesfully loaded\n')
    }
    execute_radian = async (k) => {
        k = Number(k)
        let r = await mlModel.predict(tf.tensor([k])).array()
        console.log(`real=${rad(k).toFixed(3)} predicted = ${(Number(r[0])).toFixed(3)}`)
        if(r[0] != Number(r[0])){
            return false
        }
        if(Number(r[0]).toFixed(3) == -0){
            r[0] = 0
        }
        if(rad(k).toFixed(3) == Number(r[0]).toFixed(3)){
            return true
        }
    }
    init_matrix =(n) => {
        let i = 0
        let arr_1 = []
        let arr_2 = []

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
        await mlModel.fit(xs, ys, {epochs: 2000})
        let r = await mlModel.predict(tf.tensor2d([[k]], [1, 1])).array()
        console.log(`predicted = ${r[0]}`)
        if(mtx[5].val == r[0]){
            mlModel.save('file://./models/model_number')
        }
    }

function final() {
    console.log(`\n----------------------------------------------`)
    console.log(`Sinus training - ${sn?'success':'failed'}`)
    console.log(`Radian training - ${rd?'success':'failed'}`)
    console.log(`Matrix training - ${mt?'success':'failed'}`)
}

async function teach(){
    await mlModel.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: 'Adam',
        metrics: [tf.metrics.MSE]
    });

    await init_sinus(train_arr.length*10)
    await train(1000)
    let b1 = await execute_sinus(0);
    let b2 = await execute_sinus(1);
    let b3 = await execute_sinus(8);
    if(b1 && b2 && b3){
        sn = true
        mlModel.save('file://./models/model_sin')
    }

    // // await init_radian(1440) 
    // // await train(500)
    // await load_radian()
    // let b4 = await execute_radian(0)
    // let b5 = await execute_radian(90)
    // let b6 = await execute_radian(-90)
    // if(b4 && b5 && b6){
    //     rd = true
    //     mlModel.save('file://./models/model_rad')
    // }
    // await init_matrix()
    // await execute_matrix(arr_str(mtx[5].matrix))

    final()
}

teach()