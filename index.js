let express = require("express");
let app = express();
let port = 8080;

app.use('/camera', express.static("./camera"));
app.use(express.static("./"));

app.listen(port, function () {
  console.log(`Listening at http://localhost:${port}`);
});