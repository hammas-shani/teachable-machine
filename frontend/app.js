/**
 * Teachable Machine Clone - Frontend State Logic
 * Clean State Machine Architecture
 */

// --- 1. STATE ---
let state = {
  classes: [],
  isTrained: false,
  activeCameraId: null
};

// We store actual MediaStream references outside of serializable state
const streams = {};

// --- 2. INITIALIZATION ---
window.addEventListener("DOMContentLoaded", async () => {
  // Wipe backend dataset and state on fresh load
  try {
    await fetch('http://127.0.0.1:8000/api/reset');
    console.log("[Init] Backend reset successfully.");
  } catch (err) {
    console.warn("[Init] Could not reach backend for reset.");
  }

  const container = document.getElementById("classes-container");
  const addBtn = document.querySelector(".add-class-btn");

  // Event Delegation for all class card actions
  container.addEventListener("click", handleCardClick);
  container.addEventListener("input", handleCardInput);
  
  // For 'Hold to Record' functionality
  container.addEventListener("mousedown", handleCardMouseDown);
  container.addEventListener("touchstart", handleCardMouseDown, {passive: false});

  // Global mouseup/touchend to catch releases even if dragged out of button
  window.addEventListener("mouseup", handleGlobalMouseUp);
  window.addEventListener("touchend", handleGlobalMouseUp);

  addBtn.addEventListener("click", () => {
    addClass();
  });

  // Create initial classes as seen in default UI
  addClass("Class 1");
  addClass("Class 2");

  // Hook up Advanced Box
  const advHeader = document.getElementById("advanced-header");
  const advContent = document.getElementById("advanced-content");
  if (advHeader) {
    advHeader.addEventListener("click", () => {
      advHeader.classList.toggle("open");
      advContent.style.display = advHeader.classList.contains("open") ? "flex" : "none";
    });
  }

  // Hook up Train Model Button
  const trainBtn = document.getElementById("train-btn");
  if (trainBtn) {
    trainBtn.addEventListener("click", handleTrainClick);
  }

  // Hook up Export Dropdown
  const exportBtnTop = document.getElementById("export-btn-top");
  const exportMenu = document.getElementById("export-menu");
  if (exportBtnTop && exportMenu) {
    exportBtnTop.addEventListener("click", (e) => {
      e.stopPropagation();
      exportMenu.classList.toggle("open");
    });
    
    // Close dropdown if clicking outside
    document.addEventListener("click", () => {
      exportMenu.classList.remove("open");
    });

    // Handle export format selection
    exportMenu.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      const format = btn.dataset.format;
      if (format) {
        window.location.href = `http://127.0.0.1:8000/api/export/${format}`;
      }
    });
  }
});

// --- 3. STATE MUTATIONS ---
function addClass(defaultName) {
  const newId = Date.now() + Math.random();
  state.classes.push({
    id: newId,
    name: defaultName || `Class ${state.classes.length + 1}`,
    state: "idle", // 'idle' | 'camera_on' | 'recording'
    images: [],
    count: 0
  });
  render();
}

function updateClassState(id, newStateUpdates) {
  const classIndex = state.classes.findIndex(c => c.id === id);
  if (classIndex === -1) return;
  state.classes[classIndex] = { ...state.classes[classIndex], ...newStateUpdates };
  render();
}

// --- 4. EVENT HANDLERS ---
function handleCardInput(e) {
  if (e.target.classList.contains("class-input")) {
    const id = parseFloat(e.target.dataset.id);
    const classIndex = state.classes.findIndex(c => c.id === id);
    if (classIndex !== -1) {
      state.classes[classIndex].name = e.target.value;
      // We do not call render() here because two-way binding on an actively 
      // focused input will cause it to lose focus upon re-render.
    }
  }
}

let holdInterval = null;

function handleCardMouseDown(e) {
  const btn = e.target.closest("button");
  if (!btn) return;
  
  const action = btn.dataset.action;
  const id = parseFloat(btn.dataset.id);

  if (action === "hold") {
    e.preventDefault(); // Prevent text selection while holding
    startRecording(id);
  }
}

function handleGlobalMouseUp(e) {
  // If we were holding, stop it globally when mouse is released
  if (holdInterval) {
    const recordingClass = state.classes.find(c => c.state === "recording");
    if (recordingClass) {
      stopRecording(recordingClass.id);
    }
  }
}

function handleCardClick(e) {
  e.preventDefault(); // Prevent any button from triggering a page reload
  const btn = e.target.closest("button");
  if (!btn) return;
  
  const action = btn.dataset.action;
  const id = parseFloat(btn.dataset.id);

  if (action === "webcam") {
    startCamera(id);
  } else if (action === "upload") {
    openUploader(id);
  } else if (action === "stop_camera") {
    stopCamera(id);
  } else if (action === "delete") {
    deleteClass(id);
  } else if (action === "stop_record") {
    stopRecording(id);
  }
}

// --- 5. LOGIC & ACTIONS ---

async function startCamera(id) {
  const cls = state.classes.find(c => c.id === id);
  if (!cls || cls.state !== "idle") return;

  // Camera Rule: Enforce one active stream, stop any previous one
  if (state.activeCameraId !== null && state.activeCameraId !== id) {
    stopCamera(state.activeCameraId);
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    streams[id] = stream;
    state.activeCameraId = id;
    updateClassState(id, { state: "camera_on" });
  } catch (err) {
    console.error("Error accessing webcam", err);
    alert("Could not access webcam. Please ensure you have given permission.");
  }
}

function stopCamera(id) {
  const stream = streams[id];
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    delete streams[id];
  }
  if (state.activeCameraId === id) {
    state.activeCameraId = null;
  }
  updateClassState(id, { state: "idle" });
}

function startRecording(id) {
  const cls = state.classes.find(c => c.id === id);
  if (!cls || cls.state !== "camera_on") return;
  
  updateClassState(id, { state: "recording" });

  // Capture immediately then set interval
  captureFrame(id); 
  holdInterval = setInterval(() => {
    captureFrame(id);
  }, 200); // Capture ~5 frames per second
}

function stopRecording(id) {
  if (holdInterval) {
    clearInterval(holdInterval);
    holdInterval = null;
  }
  const cls = state.classes.find(c => c.id === id);
  if (cls && cls.state === "recording") {
    updateClassState(id, { state: "camera_on" });
  }
}

function captureFrame(id) {
  const video = document.getElementById(`video-${id}`);
  if (!video || video.readyState < 2) return; // Prevent black screen

  const canvas = document.createElement('canvas');
  canvas.width = 224; // typical ML crop size
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  
  // Crop center of video to fit 224x224
  const size = Math.min(video.videoWidth, video.videoHeight);
  const sx = (video.videoWidth - size) / 2;
  const sy = (video.videoHeight - size) / 2;
  
  ctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
  const file = dataURLtoFile(dataUrl, `webcam_${Date.now()}.jpg`);
  
  const cls = state.classes.find(c => c.id === id);
  if (cls) {
    const newImages = [...cls.images, { dataUrl, file }];
    cls.images = newImages;
    cls.count = newImages.length;
    
    // Live update DOM directly to avoid video stutter from full React-style re-render
    updateCountAndGalleryDOM(id, cls);
    checkTrainButton();
  }
}

function dataURLtoFile(dataurl, filename) {
    var arr = dataurl.split(','),
        mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), 
        n = bstr.length, 
        u8arr = new Uint8Array(n);
        
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, {type:mime});
}

function openUploader(id) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.multiple = true;
  
  input.onchange = (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    
    let processed = 0;
    const cls = state.classes.find(c => c.id === id);
    if (!cls) return;

    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const dataUrl = event.target.result;
        cls.images.push({ dataUrl, file });
        cls.count = cls.images.length;
        processed++;
        if (processed === files.length) {
          render(); // Re-render once all uploaded
          checkTrainButton();
        }
      };
      reader.readAsDataURL(file);
    });
  };
  input.click();
}

async function deleteClass(id) {
  stopCamera(id);
  
  const cls = state.classes.find(c => c.id === id);
  if (cls) {
    try {
      await fetch(`http://127.0.0.1:8000/api/class/${encodeURIComponent(cls.name)}`, { method: 'DELETE' });
      console.log(`[Delete] Deleted class ${cls.name} from backend`);
    } catch (err) {
      console.error(`[Delete] Failed to delete class ${cls.name}`, err);
    }
  }

  state.classes = state.classes.filter(c => c.id !== id);
  render();
  checkTrainButton();
}

// --- 5.1 UPLOAD & TRAIN LOGIC ---
let isTraining = false;

function checkTrainButton() {
  const trainBtn = document.getElementById("train-btn");
  if (!trainBtn) return;
  // Enable if at least one class has images
  const hasData = state.classes.some(c => c.images.length > 0);
  trainBtn.disabled = !hasData || isTraining;
}

async function handleTrainClick(e) {
  if (e) e.preventDefault(); // Absolute Guard Against Browser Reload
  if (isTraining) return;
  const trainBtn = document.getElementById("train-btn");
  const statusDiv = document.getElementById("training-status");
  
  isTraining = true;
  trainBtn.innerText = "Training...";
  trainBtn.classList.add("training");
  trainBtn.disabled = true;
  statusDiv.style.display = "block";
  statusDiv.innerText = "Preparing training data...";

  try {
    let uploadedAny = false;
    for (const cls of state.classes) {
      if (cls.images.length > 0) {
        console.log(`[Train] Uploading ${cls.images.length} images for class: ${cls.name}`);
        statusDiv.innerText = `Uploading ${cls.name}... (${cls.images.length} images)`;
        await uploadClassImages(cls.name, cls.images);
        uploadedAny = true;
      }
    }

    if (!uploadedAny) {
      throw new Error("No images found in any class. Please add images first.");
    }
    
    const totalEpochs = parseInt(document.getElementById("epochs-input")?.value || "50");
    const faceMode = document.getElementById("face-mode-toggle")?.checked || false;
    const confThreshold = parseFloat(document.getElementById("confidence-threshold-input")?.value || "0.75");
    
    console.log(`[Train] Starting backend training with ${totalEpochs} epochs, faceMode=${faceMode}, confThreshold=${confThreshold}`);
    statusDiv.innerText = "Starting training...";
    
    // Trigger backend background training
    const res = await fetch('http://127.0.0.1:8000/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: totalEpochs, face_mode: faceMode, confidence_threshold: confThreshold })
    });
    
    console.log(`[Train] /api/train response: ${res.status}`);
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Failed to start training: HTTP ${res.status} — ${errText}`);
    }
    
    const trainData = await res.json();
    console.log(`[Train] Training started:`, trainData);
    
    // Resilient Polling Mechanism — retries on network blips
    let pollFailCount = 0;
    const MAX_POLL_FAILURES = 10; // ~5 seconds of retries before giving up
    
    const trainInterval = setInterval(async () => {
      try {
        const statusRes = await fetch('http://127.0.0.1:8000/api/training_status');
        if (!statusRes.ok) throw new Error("Status error");
        
        const data = await statusRes.json();
        pollFailCount = 0; // reset failure count on success
        
        if (data.status === "training") {
          const mins = Math.floor(data.currentEpoch / 60).toString().padStart(2, '0');
          const secs = (data.currentEpoch % 60).toString().padStart(2, '0');
          statusDiv.innerText = `${mins}:${secs} — Epoch ${data.currentEpoch} / ${data.totalEpochs}`;
        } else if (data.status === "completed") {
          clearInterval(trainInterval);
          console.log('[Train] Completed! Classes:', data.classes);
          finishTraining(data.classes || []);
        } else if (data.status === "error") {
          clearInterval(trainInterval);
          console.error('[Train] Backend error:', data.error);
          alert("Training failed: " + (data.error || "Unknown error"));
          resetTrainingUI();
        }
        // "idle" or "training_started" — still waiting, keep polling silently
      } catch (pollErr) {
        pollFailCount++;
        console.warn(`[Train] Poll attempt ${pollFailCount}/${MAX_POLL_FAILURES} failed:`, pollErr.message);
        statusDiv.innerText = `Reconnecting... (${pollFailCount}/${MAX_POLL_FAILURES})`;
        if (pollFailCount >= MAX_POLL_FAILURES) {
          clearInterval(trainInterval);
          console.error("[Train] Max polling failures reached. Aborting.");
          alert("Server connection lost during training. Please check that python run.py is running and retry.");
          resetTrainingUI();
        }
      }
    }, 500); // Poll every 500ms

  } catch (err) {
    console.error("[Train] CRITICAL ERROR:", err.message, err);
    alert("Failed to upload or start training!\n\nError: " + err.message);
    resetTrainingUI();
  }
}

async function uploadClassImages(className, images) {
  const formData = new FormData();
  formData.append('class_name', className);
  
  let fileCount = 0;
  images.forEach((img, i) => {
    if (!img || !img.file) {
      console.warn(`[Upload] Skipping image ${i} for class "${className}" — img.file is missing`);
      return;
    }
    formData.append('files', img.file, img.file.name || `image_${i}.jpg`);
    fileCount++;
  });

  if (fileCount === 0) {
    throw new Error(`No valid file objects found for class "${className}". Images may not have loaded correctly.`);
  }

  console.log(`[Upload] Sending ${fileCount} files for class "${className}" to /api/upload`);
  
  // Rule A: Do NOT set Content-Type header — let browser set it with boundary
  const res = await fetch('http://127.0.0.1:8000/api/upload', {
    method: 'POST',
    body: formData
  });
  
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Upload failed for class "${className}": HTTP ${res.status} — ${errText}`);
  }
  
  const result = await res.json();
  console.log(`[Upload] Success for "${className}":`, result);
  return result;
}

function finishTraining(backendClasses) {
  resetTrainingUI();
  
  // Use backend class names if provided, fallback to frontend state
  const classNames = (backendClasses && backendClasses.length > 0)
    ? backendClasses.map(name => ({ name }))
    : state.classes;

  // Show preview
  document.getElementById("preview-empty").style.display = "none";
  document.getElementById("preview-active").style.display = "block";
  
  // Populate Output Bars
  const barsContainer = document.getElementById("preview-bars");
  barsContainer.innerHTML = classNames.map((cls, idx) => {
    const colors = ["#e8710a", "#f28b82", "#fbbc04", "#34a853"];
    const bgColor = ["#fce8e6", "#fce8e6", "#fef7e0", "#e6f4ea"];
    const c = colors[idx % colors.length];
    const bg = bgColor[idx % bgColor.length];
    // Initial state: 0% for all bars until the loop updates them
    const pct = "0%";
    const displayName = cls.name || cls;
    
    return `
      <div class="prob-bar-container" data-class="${displayName}">
        <div class="prob-label" style="color: ${c}">${displayName}</div>
        <div class="prob-bar-bg" style="background: ${bg}">
          <div class="prob-bar-fill" style="background: ${c}; width: ${pct};"></div>
          <div class="prob-percentage">${pct}</div>
        </div>
      </div>
    `;
  }).join("");
  
  // Start webcam and prediction loop
  startPreviewCamera().then(() => {
    startPredictionLoop();
  });
  
  // Setup Export Button logic (now handled in initialization dropdown logic)
}

// SETUP FILE PREVIEW
document.addEventListener("DOMContentLoaded", () => {
  const sourceSelect = document.getElementById("preview-source-select");
  const fileInput = document.getElementById("preview-file-input");
  const video = document.getElementById("preview-video");
  const img = document.getElementById("preview-image");

  if (sourceSelect && fileInput && video && img) {
    sourceSelect.addEventListener("change", (e) => {
      if (e.target.value === "file") {
        video.style.display = "none";
        img.style.display = "block";
        fileInput.click();
      } else {
        video.style.display = "block";
        img.style.display = "none";
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        const reader = new FileReader();
        reader.onload = (event) => {
          img.src = event.target.result;
        };
        reader.readAsDataURL(e.target.files[0]);
      }
    });
  }
});

let predictInterval = null; // Reused as a flag to stop/start for backward compatibility with finishTraining logic
let isPredicting = false;

function startPredictionLoop() {
  if (isPredicting) return;
  isPredicting = true;
  predictFrame();
}

function stopPredictionLoop() {
  isPredicting = false;
}

async function predictFrame() {
  if (!isPredicting) return;
  
  const toggle = document.getElementById("preview-input-toggle");
  if (toggle && !toggle.checked) {
    // If toggled off during prediction, check again soon
    setTimeout(predictFrame, 200);
    return;
  }
  
  const source = document.getElementById("preview-source-select").value;
  let canvas = document.createElement("canvas");
  let ctx = canvas.getContext("2d");
  
  canvas.width = 224;
  canvas.height = 224;
  
  if (source === "webcam") {
    const video = document.getElementById("preview-video");
    if (!video || video.readyState < 2 || video.videoWidth === 0) {
      setTimeout(predictFrame, 100);
      return;
    }
    
    // Crop center
    const size = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - size) / 2;
    const startY = (video.videoHeight - size) / 2;
    
    // IMPORTANT: Even though the video is mirrored via CSS scaleX(-1),
    // the canvas drawing must capture the unmirrored raw frame.
    // CSS transforms don't affect canvas drawImage.
    ctx.drawImage(video, startX, startY, size, size, 0, 0, 224, 224);
  } else {
    const img = document.getElementById("preview-image");
    if (!img || !img.src) {
      setTimeout(predictFrame, 100);
      return;
    }
    
    // Crop center for image too
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    const size = Math.min(w, h);
    const startX = (w - size) / 2;
    const startY = (h - size) / 2;
    ctx.drawImage(img, startX, startY, size, size, 0, 0, 224, 224);
  }
  
  // Convert to Blob asynchronously with a Promise wrapper
  const getBlob = () => new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
  const blob = await getBlob();
  
  if (blob) {
    const formData = new FormData();
    formData.append("file", blob, "preview.jpg");
    
    try {
      const res = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        body: formData
      });
      
      if (res.ok) {
        const data = await res.json();
        updatePredictionBars(data.all_probs);
        
        const badge = document.getElementById("uncertain-badge");
        if (badge) {
          badge.style.display = data.is_uncertain ? "inline" : "none";
        }
      }
    } catch (err) {
      // Silently ignore network errors during rapid polling
    }
  }
  
  // Schedule next frame only AFTER the current one is completely finished
  if (isPredicting) {
    setTimeout(predictFrame, 30); // Small 30ms buffer to let UI breathe
  }
}

function updatePredictionBars(probsDict) {
  if (!probsDict) return;
  const barsContainer = document.getElementById("preview-bars");
  if (!barsContainer) return;
  
  const containers = barsContainer.querySelectorAll('.prob-bar-container');
  containers.forEach(container => {
    const clsName = container.getAttribute('data-class');
    const prob = probsDict[clsName] || 0;
    
    // Convert 0.952 to "95%"
    const pctString = Math.round(prob * 100) + "%";
    
    const fill = container.querySelector('.prob-bar-fill');
    const pctText = container.querySelector('.prob-percentage');
    
    if (fill) fill.style.width = pctString;
    if (pctText) pctText.innerText = pctString;
  });
}


async function startPreviewCamera() {
  const video = document.getElementById("preview-video");
  if (!video) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (e) {
    console.error("Preview camera error", e);
  }
}

function resetTrainingUI() {
  isTraining = false;
  const trainBtn = document.getElementById("train-btn");
  const statusDiv = document.getElementById("training-status");
  if (trainBtn) {
    trainBtn.innerText = "Train Model";
    trainBtn.classList.remove("training");
    checkTrainButton();
  }
  if (statusDiv) {
    statusDiv.style.display = "none";
  }
}

// --- 6. RENDER LOGIC ---

function render() {
  const container = document.getElementById("classes-container");
  
  // Preserve cursor position & focus if user is typing
  const activeInput = document.activeElement;
  let focusedId = null;
  let cursorPosition = null;
  if (activeInput && activeInput.classList.contains("class-input")) {
    focusedId = parseFloat(activeInput.dataset.id);
    cursorPosition = activeInput.selectionStart;
  }

  container.innerHTML = state.classes.map(cls => {
    const isIdle = cls.state === "idle";
    const isCameraOn = cls.state === "camera_on";
    const isRecording = cls.state === "recording";

    return `
      <div class="class-card" id="card-${cls.id}">
        <div class="class-header">
          <input 
            class="class-input" 
            data-id="${cls.id}" 
            value="${escapeHtml(cls.name)}"
          />
          <span class="material-symbols-outlined edit-icon">edit</span>
          <div class="header-spacer" style="flex:1;"></div>
          <button type="button" class="export-btn icon-btn" data-action="delete" data-id="${cls.id}" title="Delete class">
            <span class="material-symbols-outlined">more_vert</span>
          </button>
          <button type="button" class="export-btn icon-btn" data-action="delete" data-id="${cls.id}" title="Delete class">
            <span class="material-symbols-outlined">delete</span>
          </button>
        </div>

        <div class="divider"></div>

        <div class="class-body">
          <div class="controls-layout" style="display: flex; gap: 16px;">
            <div class="left-controls" style="flex: 1; display:flex; flex-direction:column; gap:8px;">
              <p class="sample-text" style="margin-bottom:4px;">Add Image Samples:</p>
              
              ${!isIdle ? `
                <div class="camera-box">
                  <video id="video-${cls.id}" autoplay playsinline muted></video>
                </div>
              ` : ''}

              <div class="sample-buttons" style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: auto;">
                ${isIdle ? `
                  <button type="button" class="sample-btn blue" data-action="webcam" data-id="${cls.id}" style="flex:1">
                    <span class="material-symbols-outlined">videocam</span><br>Webcam
                  </button>
                  <button type="button" class="sample-btn blue-light" data-action="upload" data-id="${cls.id}" style="flex:1">
                    <span class="material-symbols-outlined">upload</span><br>Upload
                  </button>
                ` : ''}
                
                ${isCameraOn ? `
                  <button type="button" class="sample-btn blue record-hold-btn" style="flex:1" data-action="hold" data-id="${cls.id}">
                    Hold to Record
                  </button>
                  <button type="button" class="sample-btn gray icon-only" data-action="stop_camera" data-id="${cls.id}" title="Settings/Stop">
                    <span class="material-symbols-outlined">settings</span>
                  </button>
                  <button type="button" class="sample-btn gray icon-only" data-action="stop_camera" data-id="${cls.id}" title="Close Camera">
                    <span class="material-symbols-outlined">close</span>
                  </button>
                ` : ''}

                ${isRecording ? `
                  <button type="button" class="sample-btn record record-hold-btn" style="flex:1" data-action="stop_record" data-id="${cls.id}">
                    Recording...
                  </button>
                  <button type="button" class="sample-btn gray icon-only" data-action="stop_camera" data-id="${cls.id}" title="Settings/Stop">
                    <span class="material-symbols-outlined">settings</span>
                  </button>
                  <button type="button" class="sample-btn gray icon-only" data-action="stop_camera" data-id="${cls.id}" title="Close Camera">
                    <span class="material-symbols-outlined">close</span>
                  </button>
                ` : ''}
              </div>
            </div>

            ${cls.images.length > 0 || !isIdle ? `
              <div class="right-gallery" style="flex: 1.5; border-left: 1px solid #eef0ef; padding-left: 16px;">
                <div class="counter" id="counter-${cls.id}">${cls.count} Image Sample${cls.count !== 1 ? 's' : ''}</div>
                <div class="gallery" id="gallery-${cls.id}">
                  ${cls.images.map(img => `<img src="${img.dataUrl}" alt="sample" />`).join("")}
                </div>
              </div>
            ` : ''}
          </div>
        </div>
      </div>
    `;
  }).join("");


  // Restore focus if needed
  if (focusedId !== null) {
    const input = container.querySelector(`input[data-id="${focusedId}"]`);
    if (input) {
      input.focus();
      input.setSelectionRange(cursorPosition, cursorPosition);
    }
  }

  // Attach streams to newly rendered video elements
  state.classes.forEach(cls => {
    if ((cls.state === "camera_on" || cls.state === "recording") && streams[cls.id]) {
      const video = document.getElementById(`video-${cls.id}`);
      if (video && video.srcObject !== streams[cls.id]) {
        video.srcObject = streams[cls.id];
        // video.play() is handled by 'autoplay' attribute
      }
    }
  });

  // Ensure connectors align
  if (typeof drawConnectors === 'function') {
    setTimeout(drawConnectors, 50);
  }
}

// Micro-updater for fast live recording
function updateCountAndGalleryDOM(id, cls) {
  const counterEl = document.getElementById(`counter-${id}`);
  const galleryEl = document.getElementById(`gallery-${id}`);
  
  if (counterEl) {
    counterEl.innerText = `${cls.count} Image Sample${cls.count !== 1 ? 's' : ''}`;
  }
  
  if (galleryEl && cls.images.length > 0) {
    const imgStr = cls.images[cls.images.length - 1].dataUrl;
    const imgEl = document.createElement('img');
    imgEl.src = imgStr;
    imgEl.alt = 'sample';
    galleryEl.appendChild(imgEl);
    
    // Auto-scroll gallery to bottom
    galleryEl.scrollTop = galleryEl.scrollHeight;
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// --- 7. CONNECTORS (Preserved for aesthetics) ---
function drawConnectors() {
  const svg = document.querySelector('.connector-svg');
  const center = document.querySelector('.center-panel');
  const right = document.querySelector('.right-panel');
  if (!svg || !center || !right) return;

  svg.querySelectorAll('[data-connector="class"]').forEach(n => n.remove());

  const classCards = Array.from(document.querySelectorAll('.class-card'));
  const trainingBox = center.querySelector('.training-card') || center;
  const trainingRect = trainingBox.getBoundingClientRect();

  classCards.forEach((card, index) => {
    const rect = card.getBoundingClientRect();
    const start = { x: rect.right - 8, y: rect.top + rect.height / 2 };
    
    const total = classCards.length;
    const ratio = total > 0 ? (index + 1) / (total + 1) : 0.5;
    let endY = trainingRect.top + ratio * trainingRect.height;
    endY = Math.max(trainingRect.top + 8, Math.min(trainingRect.bottom - 8, endY));

    const end = { x: trainingRect.left + 12, y: endY };
    const cx = (start.x + end.x) / 2;

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M ${start.x} ${start.y} C ${cx} ${start.y} ${cx} ${end.y} ${end.x} ${end.y}`);
    path.setAttribute('stroke', '#c4c7c5');
    path.setAttribute('stroke-width', '2.5');
    path.setAttribute('fill', 'transparent');
    path.setAttribute('data-connector', 'class');
    svg.appendChild(path);
  });

  svg.querySelectorAll('[data-connector="train-preview"]').forEach(n => n.remove());
  const previewBox = right.querySelector('.preview-card') || right;
  const previewRect = previewBox.getBoundingClientRect();

  const tStart = { x: trainingRect.right - 8, y: trainingRect.top + trainingRect.height / 2 };
  const tEnd = { x: previewRect.left + 8, y: previewRect.top + previewRect.height / 2 };
  const tcx = (tStart.x + tEnd.x) / 2;

  const tPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  tPath.setAttribute('d', `M ${tStart.x} ${tStart.y} C ${tcx} ${tStart.y} ${tcx} ${tEnd.y} ${tEnd.x} ${tEnd.y}`);
  tPath.setAttribute('stroke', '#c4c7c5');
  tPath.setAttribute('stroke-width', '2.5');
  tPath.setAttribute('fill', 'transparent');
  tPath.setAttribute('data-connector', 'train-preview');
  svg.appendChild(tPath);
}

let _ticking = false;
window.addEventListener('scroll', () => {
  if (!_ticking) {
    window.requestAnimationFrame(() => {
      drawConnectors();
      _ticking = false;
    });
    _ticking = true;
  }
});
window.addEventListener('resize', drawConnectors);
