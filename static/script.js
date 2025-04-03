function previewImage(slot) {
    const input = document.getElementById(`imageInput${slot}`);
    const uploadBox = input.closest('.upload-box');
    const file = input.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            let img = uploadBox.querySelector('.image-preview');
            if (!img) {
                img = document.createElement('img');
                img.classList.add('image-preview');
                uploadBox.appendChild(img);
            }
            img.src = e.target.result;
            img.style.display = "block";

            uploadBox.querySelector('p').style.display = "none";
        };
        reader.readAsDataURL(file);
    }
}

function uploadImages() {
    let formData = new FormData();
    let hasFile = false;

    document.querySelectorAll(".image-result").forEach(img => {
        img.style.display = "none";  
        img.src = "";  
    });

    for (let i = 1; i <= 4; i++) {
        let input = document.getElementById("imageInput" + i);
        let file = input.files[0];

        if (file) {
            formData.append("file" + i, file);
            document.getElementById("loader" + i).style.display = "block";
            hasFile = true;
        } else {
            let imgSrc = document.getElementById("preview" + i).src;
            if (imgSrc && !imgSrc.includes("data:image")) {  
                fetch(imgSrc)
                    .then(res => res.blob())
                    .then(blob => {
                        formData.append("file" + i, blob, `image${i}.jpg`);
                    });
                hasFile = true;
            }
        }
    }

    let selectedObject = document.getElementById("objectDropdown").value;
    formData.append("selected_object", selectedObject);

    if (!hasFile) {
        alert("Please select at least one image!");
        return;
    }

    fetch("/predict", {
        method: "POST",
        body: formData,
        cache: "no-store"
    })
    .then(response => response.json())
    .then(data => {
        for (let i = 1; i <= 4; i++) {
            document.getElementById("loader" + i).style.display = "none";

            if (data["image_path" + i]) {
                document.getElementById("resultImage" + i).src = data["image_path" + i] + `?t=${new Date().getTime()}`;
                document.getElementById("resultImage" + i).style.display = "block";
            }
        }
        document.querySelectorAll(".container2").forEach(element => {
            element.style.display = "block";
        });
    })
    .catch(error => console.error("Error:", error));
}

document.addEventListener("DOMContentLoaded", function () {
    const dropdown = document.getElementById("objectDropdown"); 

    document.getElementById("reload-btn").addEventListener("click", function() {
        location.reload();
    });
});


document.getElementById("reload-btn").addEventListener("click", function() {
    location.reload();
});