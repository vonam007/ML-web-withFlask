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
    img.onload = function () {
      // alert(img.width + "x" + img.height);
      if (img.width > screen.width) {
        img.width = img.width / 4;
        img.height = img.height / 4;
      }
      var croppieContainer = document.getElementById("img");
      var viewportSize = { width: screen.width, height: 50 };
      var boundarySize = { width: screen.width, height: img.height };

      if (croppie !== null) {
        croppie.destroy();
      }
      // Initialize Croppie instance
      croppie = new Croppie(croppieContainer, {
        viewport: viewportSize,
        boundary: boundarySize,
        enableZoom: false,
      });
    };
  };
  // let file = $("#fileinput").prop("files")[0];
  reader.readAsDataURL(file);
});

// document.getElementById("predictbtn").addEventListener("click", function () {
//   if (croppie !== null) {
//     croppie.result().then(function (result) {
//       // Display cropped image
//       var imgElement = document.createElement("img");
//       imgElement.src = result;
//       imgElement.id = "new_img";

//       var resultContainer = document.getElementById("new_img");
//       if (resultContainer) {
//         resultContainer.innerHTML = ""; // Clear the container
//         resultContainer.appendChild(imgElement);
//       } else {
//         // Create a new container if it doesn't exist
//         var newContainer = document.createElement("div");
//         newContainer.id = "new_img";
//         newContainer.appendChild(imgElement);
//         document.body.appendChild(newContainer);
//       }
//     });
//   }
// });

function scrollToElement() {
  document.getElementById("targetElement").scrollIntoView({ behavior: "smooth" });
}
