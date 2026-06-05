// ---------------- TAB SWITCH ----------------
function showTab(type) {
    document.getElementById("textTab").classList.add("hidden");
    document.getElementById("imageTab").classList.add("hidden");
    document.getElementById("docTab").classList.add("hidden");

    document.querySelectorAll(".tab").forEach(btn => btn.classList.remove("active"));

    if (type === "text") {
        document.getElementById("textTab").classList.remove("hidden");
        document.getElementById("tabText").classList.add("active");
    }

    if (type === "image") {
        document.getElementById("imageTab").classList.remove("hidden");
        document.getElementById("tabImage").classList.add("active");
    }

    if (type === "doc") {
        document.getElementById("docTab").classList.remove("hidden");
        document.getElementById("tabDoc").classList.add("active");
    }

    clearResult();
}

// ---------------- LOADING ----------------
function showLoading() {
    document.getElementById("resultBox").innerHTML = `
        <p style="color:#00e0ff;">Analyzing...</p>
    `;
}

function showError(msg) {
    document.getElementById("resultBox").innerHTML = `
        <p style="color:red;">${msg}</p>
    `;
}

function clearResult() {
    document.getElementById("resultBox").innerHTML = "";
}

// ---------------- RESULT UI ----------------
function showResult(data) {

    if (data.error) {
        showError(data.error);
        return;
    }

    let ai = data.ai_score || 60;
    let human = data.human_score || 40;

    let tags = (data.explanation || ["AI pattern detected"])
        .map(t => `<span>${t}</span>`)
        .join("");

    document.getElementById("resultBox").innerHTML = `
        <div class="result-card">

            <p class="result-sub">
                This content is likely <b>${data.verdict || "AI Generated"}</b>.
            </p>

            <div class="bars">

                <div class="bar-row">
                    <span class="ai-text">AI Generated</span>
                    <span>${ai}%</span>
                </div>
                <div class="progress">
                    <div class="fill ai" id="aiBar"></div>
                </div>

                <div class="bar-row">
                    <span class="human-text">Human Written</span>
                    <span>${human}%</span>
                </div>
                <div class="progress">
                    <div class="fill human" id="humanBar"></div>
                </div>

            </div>

            <div class="result-center">
                <div class="circle-container">
                    <svg viewBox="0 0 120 120" class="circle-svg">
                        <circle cx="60" cy="60" r="50" class="bg"></circle>
                        <circle cx="60" cy="60" r="50" class="progress-circle" id="circleBar"></circle>
                    </svg>

                    <div class="circle-text">
                        <h2 id="circlePercent">0%</h2>
                        <span>AI Content</span>
                    </div>
                </div>
            </div>

            <div class="tags">
                ${tags}
            </div>

            ${data.extracted_text ? `
                <button class="toggle-btn" onclick="toggleExtractedText()">
                    View Extracted Text ▼
                </button>

                <div id="extractedTextBox" class="extracted hidden">
                    <p>${data.extracted_text}</p>
                </div>
            ` : ""}

        </div>
    `;

    animateResult(ai, human);
}

// ---------------- ANIMATION ----------------
function animateResult(ai, human) {

    let aiBar = document.getElementById("aiBar");
    let humanBar = document.getElementById("humanBar");
    let circle = document.getElementById("circleBar");
    let percentText = document.getElementById("circlePercent");

    setTimeout(() => {
        aiBar.style.width = ai + "%";
        humanBar.style.width = human + "%";
    }, 200);

    let radius = 50;
    let circumference = 2 * Math.PI * radius;

    circle.style.strokeDasharray = circumference;
    circle.style.strokeDashoffset = circumference;

    let offset = circumference - (ai / 100) * circumference;

    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
    }, 300);

    let count = 0;
    let speed = Math.max(10, 1500 / ai);

    let interval = setInterval(() => {
        if (count >= ai) {
            percentText.innerText = ai + "%";
            clearInterval(interval);
        } else {
            count++;
            percentText.innerText = count + "%";
        }
    }, speed);
}

// ---------------- TOGGLE OCR ----------------
function toggleExtractedText() {
    let box = document.getElementById("extractedTextBox");
    box.classList.toggle("hidden");
}

// ---------------- TEXT ----------------
function checkText() {
    let text = document.getElementById("text").value.trim();

    if (!text) {
        showError("Please enter text");
        return;
    }

    showLoading();

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => showResult(data))
    .catch(() => showError("Error processing request"));
}

// ---------------- IMAGE ----------------
function checkImage() {
    let file = document.getElementById("imageInput").files[0];

    if (!file) {
        showError("Please select an image");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    showLoading();

    fetch("/predict-image", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => showResult(data))
    .catch(() => showError("Error processing image"));
}

// ---------------- DOCUMENT ----------------
function checkDoc() {
    let file = document.getElementById("docInput").files[0];

    if (!file) {
        showError("Please select a document");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    showLoading();

    fetch("/predict-doc", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => showResult(data))
    .catch(() => showError("Error processing document"));
}

// ---------------- IMAGE PREVIEW ----------------
document.getElementById("imageInput").addEventListener("change", function () {
    const file = this.files[0];
    const box = document.getElementById("imagePreviewBox");

    this.parentElement.classList.add("active");

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            box.innerHTML = `
                <img src="${e.target.result}" class="preview-img"/>
                <p class="preview-name">${file.name}</p>
            `;
        };

        reader.readAsDataURL(file);
    }
});

// ---------------- DOCUMENT PREVIEW ----------------
document.getElementById("docInput").addEventListener("change", function () {
    const file = this.files[0];
    const box = document.getElementById("docPreviewBox");

    this.parentElement.classList.add("active");

    if (file) {
        box.innerHTML = `
            <div class="doc-preview">
                <i data-lucide="file-text"></i>
                <p>${file.name}</p>
                <span>${(file.size / 1024).toFixed(1)} KB</span>
            </div>
        `;

        lucide.createIcons();
    }
});