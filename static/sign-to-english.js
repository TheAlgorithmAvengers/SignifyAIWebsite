const video = document.getElementById("video");
const sentenceOutput = document.getElementById("sentenceOutput");
const predictionModeSelect = document.getElementById("predictionMode");
const predictionStatus = document.getElementById("predictionStatus");

let model;
let sentence = "";
let predictionBuffer = [];
let previousChar = "";
let predictionMode = "buffered";
let isPredicting = false;

const BUFFER_SIZE = 30;
const INSTANT_INTERVAL = 8;
let predictionNo = 0;


const observation = ["Space"];

for (let i = 65; i <= 90; i++) {
  observation.push(String.fromCharCode(i));
}

observation.push("Delete", "FullStop", "Clear");


const hands = new Hands({
  locateFile: file =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});


function speak(text) {
  const utter = new SpeechSynthesisUtterance(text);
  speechSynthesis.speak(utter);
}

function updateSentenceOutput() {
  sentenceOutput.value = sentence;
}

function updatePredictionStatus() {
  if (predictionMode === "buffered") {
    predictionStatus.textContent = `Buffered mode uses the last ${BUFFER_SIZE} predictions for a more stable result.`;
    return;
  }

  predictionStatus.textContent = "Instant mode uses the latest prediction and updates faster.";
}

function getBufferedPrediction() {
  if (predictionBuffer.length === 0) return null;

  const counts = {};
  let topPrediction = predictionBuffer[0];
  let topCount = 0;

  predictionBuffer.forEach(prediction => {
    counts[prediction] = (counts[prediction] || 0) + 1;

    if (counts[prediction] > topCount) {
      topPrediction = prediction;
      topCount = counts[prediction];
    }
  });

  return topPrediction;
}

function applyPrediction(prediction) {
  let char = observation[prediction];

  console.log("Predicted:", char);

  if (char === "Space") {
    char = " ";
    if (previousChar === " ") char = "FullStop";
  }

  if (char === "FullStop") {
    char = ".";
    sentence += char;
    previousChar = char;
    updateSentenceOutput();
    speak(sentence);
    sentence = "";
    updateSentenceOutput();
    return;
  }

  if (char === "Clear") {
    sentence = "";
    previousChar = "";
    updateSentenceOutput();
    return;
  }

  if (char === "Delete") {
    sentence = sentence.slice(0, -1);
    previousChar = sentence.slice(-1);
    updateSentenceOutput();
    return;
  }

  sentence += char;
  previousChar = char;
  updateSentenceOutput();

  console.log("Sentence:", sentence);
}

async function predictSign(landmarks) {

  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ landmarks }),
  });

  const result = await response.json();

  // console.log("Prediction:", result);

  return result.label;
}

async function predict(handArray) {
  if (isPredicting) return;

  isPredicting = true;

  try {
    const prediction = await predictSign(handArray);

    predictionBuffer.push(prediction);

    if (predictionBuffer.length > BUFFER_SIZE) {
      predictionBuffer.shift();
    }

    if (predictionMode === "buffered") {
      if (predictionBuffer.length < BUFFER_SIZE) return;

      const bufferedPrediction = getBufferedPrediction();

      if (bufferedPrediction === null) return;

      predictionBuffer = [];
      applyPrediction(bufferedPrediction);
      return;
    }

    predictionNo++;

    if (predictionNo < INSTANT_INTERVAL) return;

    predictionNo = 0;
    applyPrediction(prediction);
  } finally {
    isPredicting = false;
  }
}


hands.onResults(results => {

  if (!results.multiHandLandmarks) return;

  results.multiHandLandmarks.forEach((landmarks, i) => {

    
    const handArray = landmarks
      .map(p => [p.x, p.y, p.z])
      .flat();

    predict(handArray);
  });
});


const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({ image: video });
  },
  width: 480,
  height: 360
});

camera.start();

predictionModeSelect.addEventListener("change", event => {
  predictionMode = event.target.value;
  predictionBuffer = [];
  predictionNo = 0;
  updatePredictionStatus();
});

updateSentenceOutput();
updatePredictionStatus();
