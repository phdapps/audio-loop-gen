window.addEventListener("load", function () {
    // Native audio player can't play seemless loops
    let audioCtx = null;
    let source = null;
    isPlaying = true;

    let animationFrameId = null;

    // JavaScript to handle file selection and playback
    document
        .getElementById("audioFile")
        .addEventListener("change", async function (event) {
            if (audioCtx === null) {
                audioCtx = new window.AudioContext();
            }
            const file = event.target.files[0];
            const fileUrl = URL.createObjectURL(file);
            const arrayBuffer = await fetch(fileUrl).then((res) =>
                res.arrayBuffer()
            );
            // Convert ArrayBuffer to an AudioBuffer
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            if (source !== null) {
                source.stop();
                source.disconnect(audioCtx.destination);
            }
            source = audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.loop = true;
            source.connect(audioCtx.destination);
            source.start();
            if (audioCtx.state === 'suspended') {
                // Resume the audio context if it's suspended
                await audioCtx.resume();
            }
            isPlaying = true;
            let btn = document.getElementById('pauseResumeBtn');
            btn.disabled = false;
            btn.textContent = "Pause";
            updateProgressBar(audioBuffer.duration);
        });

    // Pause/Resume functionality
    document
        .getElementById("pauseResumeBtn")
        .addEventListener("click", function () {
            if (audioCtx !== null) {
                if (isPlaying) {
                    audioCtx.suspend();
                    this.textContent = "Resume";
                } else {
                    audioCtx.resume();
                    this.textContent = "Pause";
                }
                isPlaying = !isPlaying;
            }
        });

    function updateProgressBar(duration) {
        const progressBar = document.getElementById("progressBar");
        function update() {
            if (!source) return;
            const playTime = audioCtx.currentTime - source.startTime;
            const progress = (playTime / duration) * 100;
            progressBar.value = progress % 100; // Reset progress on loop
            animationFrameId = requestAnimationFrame(update);
        }
        source.startTime = audioCtx.currentTime;
        animationFrameId = requestAnimationFrame(update);
    }
});