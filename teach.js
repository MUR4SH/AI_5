const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const mtx = require('./matrix.json')

f = (arg) => Number(Math.sin(arg).toFixed(3)) //Синус вычисляется по радианам
rad = (arg) => Number((((arg)*Math.PI)/180).toFixed(3)) //Перевод из градусов в радианы - есть погрешность
const train_arr = [0, 0.259, 0.5, 0.707, 0.866, 0.966, 1, 0.966, 0.866, 0.707, 0.5, 0.259, 0, -0.259, -0.5, -0.707, -0.866, -0.966, -1, -0.966, -0.866, -0.707, -0.5, -0.259]

let mt = false
let rd = false
let sn = false

const size = 32

let mlModel = tf.sequential();
mlModel.add(tf.layers.dense({units: 40, inputShape:[35], activation: "tanh"}));
mlModel.add(tf.layers.dense({units: 256, activation: "tanh"}));
mlModel.add(tf.layers.dense({units: 1}));

/*
radian - Adam optimizer, meanSquaredError loss, metrics - MSE
sinus - Adam optimizer, meanSquaredError loss, metrics - MSE
number - Adam optimizer, meanSquaredError loss, metrics - MSE
*/
let xs, ys

function arr_str(arr) {
    let str = '1'
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
            batchSize: Math.pow(2,10),
            epochs: epoch,
        })
        console.log('\nLearning completed\n')
    }
    execute_sinus = async (k) => {
        k = Number(k)
        let r = await mlModel.predict(tf.tensor([k])).array()
        console.log(`real=${train_arr[k%train_arr.length].toFixed(2)} predicted = ${Number(r[0]).toFixed(2)}`)
        if(r[0] != Number(r[0])){
            return false
        }
        if(Number(r[0]).toFixed(2) == -0){
            r[0] = 0
        }
        if(train_arr[k%train_arr.length].toFixed(2) == Number(r[0]).toFixed(2)){
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
            let s = i%10//Math.floor(Math.random()*10)
            let x = mtx[s].matrix
            arr_1.push(x)
            arr_2.push(mtx[s].value)
            i++
        }

        xs = tf.tensor(arr_1);
        ys = tf.tensor(arr_2);
    }
    execute_matrix = async (k) => {
        k = Number(k)
        // Train model with fit().
        let r = await mlModel.predict(tf.tensor([mtx[k].matrix])).array()
        console.log(`real = ${mtx[k].value} predicted = ${r[0]}`)
        if(mtx[k].value == Math.round(r[0])){
            return true
        }
    }

function final() {
    console.log(`\n----------------------------------------------`)
    console.log(`Sinus training - ${sn?'success':'failed'}`)
    console.log(`Radian training - ${rd?'success':'failed'}`)
    console.log(`Matrix training - ${mt?'success':'failed'}`)
}

async function teach(num){
    await mlModel.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: 'Adam',
        metrics: [tf.metrics.MSE]
    });
    if(num == 0){
        await init_sinus(train_arr.length*2)
        await train(20000)
        let b1 = await execute_sinus(0);
        let b2 = await execute_sinus(1);
        let b3 = await execute_sinus(3);
        if(b1 && b2 && b3){
            sn = true
            mlModel.save('file://./models/model_sin')
        }
    }else if(num == 1){
        await init_radian(1440) 
        await train(500)
        //await load_radian()
        let b4 = await execute_radian(90)
        let b5 = await execute_radian(90)
        let b6 = await execute_radian(-90)
        if(b4 && b5 && b6){
            rd = true
            mlModel.save('file://./models/model_rad')
        }
    }else if(num == 2){
        await init_matrix(10)
        await train(1000)
        let b7 = await execute_matrix(0)
        b7 &= await execute_matrix(1)
        b7 &= await execute_matrix(2)
        b7 &= await execute_matrix(3)
        b7 &= await execute_matrix(4)
        b7 &= await execute_matrix(5)
        b7 &= await execute_matrix(6)
        b7 &= await execute_matrix(7)
        b7 &= await execute_matrix(8)
        b7 &= await execute_matrix(9)

        if(b7){
            mt = true
            mlModel.save('file://./models/model_number')
        }
    }
    final()
}

teach()