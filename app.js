var express = require("express"),
	bodyParser= require("body-parser"),
    app       = express(),
    fileUpload = require('express-fileupload');
var PythonShell = require('python-shell'),
sys = require('sys');



app.use(bodyParser.urlencoded({extended : true}));
app.set("view engine", "ejs");
app.use(fileUpload());
app.use(express.static(__dirname));

app.get("/", function(req, res){

	res.render("start");
});
app.post('/classify', function(req, res) {
  if (!req.files)
    return res.status(400).send('No files were uploaded.');
  console.log('body'+req.body);
  let sampleFile = req.files.sampleFile;
  var filename = req.files.sampleFile.name;
  var marks1 = parseFloat(req.body.marks1);
  var marks2 = parseInt(req.body.marks2);
  var m = [marks1, marks2];
    // Use the mv() method to place the file somewhere on your server
  sampleFile.mv(__dirname+'/data/'+filename, function(err) {
    if (err)
      return res.status(500).send(err);
 
  var options = {
  mode: 'text',
  pythonPath: "C:/Users/hp/Anaconda3-2/python.exe",
  pythonOptions: ['-u'],
  scriptPath: './',
  args: ["./data/"+filename, m]
};

var shell = new PythonShell('classification.py', options);
shell.on('message', function (message) {

	var len = message.length;
  var fs = require('fs');
  var obj;
  fs.readFile('output', 'utf8', function (err, data) {
  if (err) throw err;
  obj = JSON.parse(data);
  console.log((obj));
  res.render("result", {obj:obj});
});

});

  });
});

app.listen(3000, "localhost", function(err){
	console.log("Server on duty, Mallady!");
});