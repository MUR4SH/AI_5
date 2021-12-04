const { app, BrowserWindow } = require('electron')
const server_http = require("http");
const url = require('url');
const mdl = require('./module.js')
const fs = require('fs');
let html

server_http.createServer(async function(req, res){
    res.statusCode = 404;
    if(req.method == "GET"){
        let url_s = (req.url).replace('/127.0.0.1:8080','')
        console.log(url_s)
        if(url_s == '/'){
            res.statusCode = 200;
            html = fs.readFileSync('./index.html','utf-8')
            res.setHeader('Content-type', 'text/html');
            res.write(html)
        }else if(url_s.match(/^\/getsinus/)){
            let urlRequest = url.parse(req.url, true);                
            res.statusCode = 200;
            res.setHeader('Content-type', 'text/json');
            let a = await mdl.execute_sinus(decodeURI(urlRequest.query.index))
            res.write(JSON.stringify({res: a}))
        }else if(url_s.match(/^\/getradian/)){
            let urlRequest = url.parse(req.url, true);                
            res.statusCode = 200;
            res.setHeader('Content-type', 'text/json');
            let a = await mdl.execute_radian(decodeURI(urlRequest.query.degree))
            res.write(JSON.stringify({res: a}))
        }else if(url_s.match(/^\/getnumber/)){
            let urlRequest = url.parse(req.url, true);                
            res.statusCode = 200;
            res.setHeader('Content-type', 'text/json');
            let a = await mdl.execute_matrix(decodeURI(urlRequest.query.matrix))
            res.write(JSON.stringify({res: a}))
        }
        res.end();
    }
}).listen('8080',()=>{
    console.log('SERVER http://127.0.0.1'+':'+'8080');
});

function createWindow () {
    const win = new BrowserWindow({
      width: 1000,
      height: 1000,
      webPreferences: {
        nodeIntegration: true,
        enableRemoteModule: true,
      }
    })
  
    win.loadFile('index.html')
    win.setResizable(false);
}

app.whenReady().then(createWindow)
  
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
  
app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
})