if(typeof jQuery!=='undefined') {
    console.log('jQuery Loaded');
} else {
    console.log('jQuery not loaded yet');
}

var js_full_model_url = 'https://raw.githubusercontent.com/iglaweb/HippoYD/master/out_epoch_80_full/tfjs_model_80/model.json'
var js_lite_model_url = 'https://raw.githubusercontent.com/iglaweb/HippoYD/master/out_epoch_80_lite/tfjs_model_80/model.json'


$(".dropdown-menu a").click(function(){
	var selText = $(this).text();
	console.log(selText);
	$('.status').text(this.innerHTML);
	// reload models
	init_models();
});


$("#image-selector").change(function () {
	var fileName = $(this).val();
	//replace the "Choose a file" label
	$(this).next('.custom-file-label').html(fileName);

	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}

	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

let lastSelectedValue = null;

async function init_models() {
	let selectedValue = $('.dropdown-toggle').text().trim();
	if(lastSelectedValue == selectedValue) {
		console.log('Value not changed')
		return;
	}

	$('.progress-bar').show();

	lastSelectedValue = selectedValue;
	var model_url = undefined;
	if(selectedValue == 'Full') {
		model_url = js_full_model_url;
	} else {
		model_url = js_lite_model_url;
	}
	console.log('Use model: ' + model_url);

	try {
		yawn_js_model = await tf.loadLayersModel(model_url);
		console.log("Model loaded successfully")
	} catch(e) {
		console.log("Model could not be loaded")
		console.log(e)
		
		$(".toast").toast('show');
	} finally {
		$('.progress-bar').hide();
	}
}

let yawn_js_model;
$(document).ready(init_models());

predict_image = async function (image) {
	let pre_image = tf.browser.fromPixels(image, 1)
		.resizeNearestNeighbor([100, 100])
		.expandDims()
		.toFloat()
		.div(255.0)
		.reverse(-1);
	console.log(pre_image);
	let predict_result = await yawn_js_model.predict(pre_image).data();
	let probability = predict_result[0];
	return probability;
}

$("#predictBtn").click(async function () {
	let image = $('#selected-image').get(0);
	const start1 = performance.now();
	let probability = await predict_image(image);
	const time1  = Math.round(performance.now() - start1);
    console.log('Face inference time: ' + time1 + ' ms');
	console.log(probability);
	$("#prediction-list").empty();

	let percentOpened = parseInt(Math.trunc(probability * 100));
	let pb_color = percentOpened >= 20 ? 'red' : 'blue';
	let pb_style = "background:" + pb_color + ";width:" + percentOpened + "%";
	pb = "<div class='progress'><div id='progress-bar' class='progress-bar progress-bar-striped active' role='progressbar' aria-valuemin='0' aria-valuemax='100' style='" + pb_style + "'></div></div>";
	$("#prediction-list").append(`<li>probability: ${percentOpened}% opened, time elapsed: ${time1} ms ${pb}</li>`);
});