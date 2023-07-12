//upload image:
$("#uploadbtn").click(function () {
  $("#fileinput").trigger("click");
});
$("#predictbtn").click(function () {
  $("#hiddenbtn").trigger("click");
});
var croppie = null;

// $("#fileinput").change(async function () {
document.getElementById("fileinput").addEventListener("change", function (e) {
  let file = this.files[0];
  let reader = new FileReader();
  reader.onload = function () {
    let dataURL = reader.result;
    $("#img").attr("src", dataURL);
  };
  // let file = $("#fileinput").prop("files")[0];
  reader.readAsDataURL(file);
});

