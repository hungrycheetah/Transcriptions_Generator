const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const fileName = document.querySelector("#fileName");
const submitButton = document.querySelector("#submitButton");
const statusText = document.querySelector("#statusText");
const transcriptOutput = document.querySelector("#transcriptOutput");
const downloadLink = document.querySelector("#downloadLink");

let currentJobId = null;
let pollTimer = null;

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name || "MP4, MP3, WAV, M4A, WEBM, MOV";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    return;
  }

  clearInterval(pollTimer);
  downloadLink.classList.add("hidden");
  transcriptOutput.classList.remove("error");
  transcriptOutput.textContent = "Uploading recording...";
  statusText.textContent = "Uploading";
  submitButton.disabled = true;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("model", document.querySelector("#modelInput").value);
  formData.append("language", document.querySelector("#languageInput").value);
  formData.append("speakers", document.querySelector("#speakersInput").checked ? "true" : "false");
  formData.append("speaker_names", document.querySelector("#speakerNamesInput").value);

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Upload failed.");
    }

    currentJobId = payload.id;
    renderJob(payload);
    pollTimer = setInterval(() => pollJob(currentJobId), 3000);
  } catch (error) {
    statusText.textContent = "Failed";
    transcriptOutput.classList.add("error");
    transcriptOutput.textContent = error.message;
    submitButton.disabled = false;
  }
});

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();

    if (!response.ok) {
      throw new Error(job.detail || "Could not fetch job status.");
    }

    renderJob(job);

    if (job.status === "completed") {
      clearInterval(pollTimer);
      await loadTranscript(job);
      submitButton.disabled = false;
    }

    if (job.status === "failed") {
      clearInterval(pollTimer);
      transcriptOutput.classList.add("error");
      transcriptOutput.textContent = job.error || "Transcription failed.";
      submitButton.disabled = false;
    }
  } catch (error) {
    clearInterval(pollTimer);
    statusText.textContent = "Failed";
    transcriptOutput.classList.add("error");
    transcriptOutput.textContent = error.message;
    submitButton.disabled = false;
  }
}

function renderJob(job) {
  const label = {
    queued: "Queued",
    running: "Transcribing",
    completed: "Completed",
    failed: "Failed",
  }[job.status] || job.status;

  statusText.textContent = `${label} • ${job.filename}`;

  if (job.status === "queued") {
    transcriptOutput.textContent = "Waiting for the transcription worker...";
  }

  if (job.status === "running") {
    transcriptOutput.textContent = "Processing audio. Longer recordings can take several minutes on CPU.";
  }

  if (job.download_url) {
    downloadLink.href = job.download_url;
    downloadLink.classList.remove("hidden");
  } else {
    downloadLink.classList.add("hidden");
  }
}

async function loadTranscript(job) {
  const response = await fetch(job.text_url);
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Could not load transcript.");
  }

  transcriptOutput.classList.remove("error");
  transcriptOutput.textContent = payload.text;
}
