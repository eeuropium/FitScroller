const hud = document.createElement('div');
hud.id = 'workout-hud';
document.body.appendChild(hud);

let lastReps = 0;

function updateHUD() {
    // Fetch the data from your Python script running locally
    fetch('http://localhost:5001/stats')
        .then(response => response.json())
        .then(data => {
            const repsChanged = data.reps !== lastReps;
            lastReps = data.reps;

            hud.innerHTML = `
                <div class="hud-title">⚡ WORKOUT TRACKER ⚡</div>

                <div class="hud-stat">
                    <div class="stat-label">PUSH-UPS</div>
                    <div class="stat-value ${repsChanged ? 'rep-flash' : ''}">${data.reps}</div>
                </div>

                <div class="hud-stat">
                    <div class="stat-label">TIME ELAPSED</div>
                    <div class="stat-value time">${data.time}</div>
                </div>

                <div class="hud-footer">
                    <span class="status-indicator"></span>SYSTEM ACTIVE
                </div>
            `;
        })
        .catch(err => {
            hud.innerHTML = `
                <div class="hud-title">⚡ WORKOUT TRACKER ⚡</div>
                <div class="error-state">
                    <div style="font-size: 24px; margin-bottom: 10px;">⚠️</div>
                    <div>AWAITING CONNECTION</div>
                    <div style="font-size: 9px; margin-top: 10px;">Start Python Script</div>
                </div>
            `;
        });
}

setInterval(updateHUD, 500); // Update every half second
