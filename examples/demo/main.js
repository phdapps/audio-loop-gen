// Native audio player can't play seemless loops
let audioCtx = null;
let source = null;
let currentMelody = null;

function updateProgressBar(melody, duration) {
    const progressBar = melody.querySelector('.melodyProgress');
    function update() {
        if (!source || melody.dataset.state === "stopped") return;
        const playTime = audioCtx.currentTime - source.startTime;
        const progress = (playTime / duration) * 100;
        progressBar.value = progress % 100; // Reset progress on loop
        requestAnimationFrame(update);
    }
    source.startTime = audioCtx.currentTime;
    // start the progress animation
    requestAnimationFrame(update);
}

async function playMelopdy(melody) {
    const url = new URL(melody.dataset.url, window.location.href);
    if (audioCtx === null) {
        audioCtx = new window.AudioContext();
    }
    const arrayBuffer = await fetch(url).then((res) =>
        res.arrayBuffer()
    );
    // Convert ArrayBuffer to an AudioBuffer
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    stopMelody();

    source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.loop = true;
    source.connect(audioCtx.destination);
    source.start();
    if (audioCtx.state === 'suspended') {
        // Resume the audio context if it's suspended
        await audioCtx.resume();
    }
    currentMelody = melody;
    melody.dataset.state = "playing";

    updateProgressBar(melody, audioBuffer.duration);
}

function stopMelody() {
    if (currentMelody !== null) {
        const button = currentMelody.querySelector('.melodyBtn');
        const progress = currentMelody.querySelector('.melodyProgress');
        button.textContent = 'Loop';
        progress.value = 0;
        currentMelody.dataset.state = "stopped";
    }
    if (source !== null) {
        source.stop();
        source.disconnect(audioCtx.destination);
    }
}

async function loadMusicData() {
    // Load the index JSON from the file
    const indexJson = await fetch('index.json').then(response => response.json());

    const groupsContainer = document.createElement('ul');
    groupsContainer.id = 'groups';

    for (const group of indexJson) {
        const groupElement = document.createElement('li');
        groupElement.className = 'musicGroup';
        groupElement.id = group.id;

        const titleDiv = document.createElement('h1');
        titleDiv.className = 'groupTitle';
        titleDiv.textContent = group.title;
        groupElement.appendChild(titleDiv);

        const commandDiv = document.createElement('div');
        commandDiv.className = 'groupCommand';
        commandDiv.innerHTML = `<label>Command:</label>${group.command}`;
        groupElement.appendChild(commandDiv);

        const melodiesList = document.createElement('ul');
        for (const item of group.items) {
            const melodyData = await fetch(`${item}.json`).then(response => response.json());
            const melodyElement = document.createElement('li');
            melodyElement.className = 'melody';
            melodyElement.dataset.url = `${item}.mp3`;
            melodyElement.dataset.state = "stopped";

            const promptDiv = document.createElement('div');
            promptDiv.className = 'melodyPrompt';
            promptDiv.innerHTML = `<label>Prompt:</label>${melodyData.params.prompt}`;
            melodyElement.appendChild(promptDiv);

            const bpmDiv = document.createElement('div');
            bpmDiv.className = 'melodyBPM';
            bpmDiv.innerHTML = `<label>BPM:</label>${melodyData.params.bpm}`;
            melodyElement.appendChild(bpmDiv);

            const durationDiv = document.createElement('div');
            durationDiv.className = 'melodyDuration';
            durationDiv.innerHTML = `<label>Duration:</label>${(melodyData.duration / 1000).toFixed(2)} sec`;
            melodyElement.appendChild(durationDiv);

            const progressContainer = document.createElement('div');
            progressContainer.className = 'melodyUi';
            const progress = document.createElement('progress');
            progress.className = 'melodyProgress';
            progress.value = 0;
            progress.max = 100;
            progressContainer.appendChild(progress);

            const button = document.createElement('button');
            button.className = 'melodyBtn';
            button.textContent = 'Loop';
            progressContainer.appendChild(button);

            melodyElement.appendChild(progressContainer);

            melodiesList.appendChild(melodyElement);
        }
        groupElement.appendChild(melodiesList);
        groupsContainer.appendChild(groupElement);
    }

    document.body.appendChild(groupsContainer);
}

window.addEventListener("load", async function () {
    await loadMusicData();

    const groups = document.querySelectorAll('.musicGroup');
    for (const group of groups) {
        const command = group.querySelector('.groupCommand').textContent;
        const melodies = group.querySelectorAll('.melody');

        for (const melody of melodies) {
            const button = melody.querySelector('.melodyBtn');
            
            button.addEventListener('click', function () {
                if (melody.dataset.state === "stopped") {
                    playMelopdy(melody);
                    this.textContent = "Pause";
                } else if (audioCtx !== null) {
                    if (melody.dataset.state === "playing") {
                        audioCtx.suspend();
                        this.textContent = "Resume";
                        melody.dataset.state = "paused";
                    } else {
                        audioCtx.resume();
                        this.textContent = "Pause";
                        melody.dataset.state = "playing";
                    }
                }
            });
        }
    }
});