
document.addEventListener("DOMContentLoaded", () => {
    const hour = new Date().getHours();
    if (hour >= 3 && hour <= 5) {
        alert("Operator Signal Detected. Initiating cognitive fork.");
        document.body.style.backgroundColor = "#330000";
    }
});
