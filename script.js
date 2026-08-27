function toggleTheme() {
    document.body.classList.toggle("light");
}

// Dynamic Alert Example
function addAlert(message) {
    const list = document.getElementById("alertList");
    const li = document.createElement("li");
    li.textContent = "⚠ " + message;
    list.appendChild(li);
}

// Example auto alert
setTimeout(() => {
    addAlert("Heavy rainfall detected!");
}, 3000);