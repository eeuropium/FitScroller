const hud = document.createElement('div');
hud.id = 'workout-hud';
document.body.appendChild(hud);

function updateHUD() {
    // Fetch the data from your Python script running locally
    fetch('http://localhost:5001/stats')
        .then(response => response.json())
        .then(data => {
            hud.innerHTML = `
                <div style="font-size: 24px; font-weight: bold;">REPS: ${data.reps}</div>
                <div style="font-size: 14px;">TIME: ${data.time}</div>
            `;
        })
        .catch(err => {
            hud.innerHTML = `<div>Start Python Script...</div>`;
        });
}

setInterval(updateHUD, 500); // Update every half second
