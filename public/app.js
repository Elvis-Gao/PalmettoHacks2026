const elements = {
  input: document.querySelector('#image-input'),
  dropzone: document.querySelector('#dropzone'),
  fileDetails: document.querySelector('#file-details'),
  fileThumb: document.querySelector('#file-thumb'),
  fileName: document.querySelector('#file-name'),
  fileSize: document.querySelector('#file-size'),
  removeImage: document.querySelector('#remove-image'),
  analyzeButton: document.querySelector('#analyze-button'),
  emptyState: document.querySelector('#empty-state'),
  resultView: document.querySelector('#result-view'),
  loadingState: document.querySelector('#loading-state'),
  previewImage: document.querySelector('#preview-image'),
  detectionLayer: document.querySelector('#detection-layer'),
  detectionTotal: document.querySelector('#detection-total'),
  analysisState: document.querySelector('#analysis-state'),
  componentDetails: document.querySelector('#component-details'),
  detailsClose: document.querySelector('#details-close'),
  detailTitle: document.querySelector('#detail-title'),
  detailConfidence: document.querySelector('#detail-confidence'),
  detailCategory: document.querySelector('#detail-category'),
  detailCondition: document.querySelector('#detail-condition'),
  detailAction: document.querySelector('#detail-action'),
  detailDescription: document.querySelector('#detail-description'),
  toast: document.querySelector('#toast')
};

let selectedFile = null;
let selectedImageUrl = null;
let lastAnalysis = null;

const categoryLabels = {
  pole: 'Pole', crossarm: 'Crossarm', insulator: 'Insulator', primary_conductor: 'Primary line',
  secondary_line: 'Secondary line', neutral_line: 'Neutral line', transformer: 'Transformer',
  cutout_fuse: 'Cutout / fuse', arrester: 'Arrester', switch: 'Switch', regulator: 'Regulator',
  capacitor: 'Capacitor', guy_wire: 'Guy wire', streetlight: 'Streetlight', telecom: 'Telecom',
  meter: 'Meter', hardware: 'Hardware', other: 'Other attachment'
};

elements.input.addEventListener('change', () => selectFile(elements.input.files[0]));
elements.removeImage.addEventListener('click', clearImage);
elements.analyzeButton.addEventListener('click', analyzeImage);
elements.detailsClose.addEventListener('click', closeDetails);
elements.previewImage.addEventListener('load', fitBoxLabels);
window.addEventListener('resize', fitBoxLabels);

['dragenter', 'dragover'].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add('is-dragging');
  });
});
['dragleave', 'drop'].forEach((eventName) => {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove('is-dragging');
  });
});
elements.dropzone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));

function selectFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showToast('Choose a JPG, PNG, or WebP image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('Choose an image smaller than 10 MB.');
    return;
  }

  selectedFile = file;
  lastAnalysis = null;
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = URL.createObjectURL(file);
  elements.fileThumb.src = selectedImageUrl;
  elements.previewImage.src = selectedImageUrl;
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = formatFileSize(file.size);
  elements.fileDetails.hidden = false;
  elements.analyzeButton.disabled = false;
  elements.emptyState.hidden = true;
  elements.loadingState.hidden = true;
  elements.resultView.hidden = false;
  closeDetails();
  elements.detectionLayer.replaceChildren();
  elements.detectionTotal.textContent = 'Ready to analyze';
  setState('Ready to analyze', 'ready');
}

function clearImage() {
  selectedFile = null;
  lastAnalysis = null;
  elements.input.value = '';
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = null;
  elements.fileDetails.hidden = true;
  elements.analyzeButton.disabled = true;
  elements.resultView.hidden = true;
  elements.emptyState.hidden = false;
  closeDetails();
  elements.previewImage.removeAttribute('src');
  elements.detectionLayer.replaceChildren();
  setState('Waiting for image', 'waiting');
}

async function analyzeImage() {
  if (!selectedFile) return;
  setLoading(true);
  try {
    const imageBase64 = await fileToBase64(selectedFile);
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageBase64, mimeType: selectedFile.type })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'The inspection could not be completed.');
    lastAnalysis = data.analysis;
    renderAnalysis(lastAnalysis);
    setState('Analysis complete', 'complete');
  } catch (error) {
    console.error(error);
    setState('Analysis interrupted', 'error');
    showToast(error.message || 'The inspection could not be completed.');
  } finally {
    setLoading(false);
  }
}

function renderAnalysis(analysis) {
  const detections = analysis.detections || [];
  elements.detectionLayer.replaceChildren();
  detections.forEach((detection, index) => renderBox(detection, index));
  elements.detectionTotal.textContent = `${detections.length} visible item${detections.length === 1 ? '' : 's'} found`;
  closeDetails();
  if (!detections.length) showToast('No components could be confidently located. Try a sharper, closer photo.');
}

function renderBox(detection, index) {
  if (detection.display_mode === 'marker' || !detection.bounding_box_1000) {
    renderMarker(detection);
    return;
  }
  const [top, left, bottom, right] = detection.bounding_box_1000;
  const box = document.createElement('div');
  box.className = `detection-box category-${detection.category}`;
  box.style.top = `${top / 10}%`;
  box.style.left = `${left / 10}%`;
  box.style.height = `${(bottom - top) / 10}%`;
  box.style.width = `${(right - left) / 10}%`;
  box.dataset.id = detection.id;
  const label = createLabel(detection, 'box-label');
  label.addEventListener('click', () => showDetails(detection));
  box.append(label);
  elements.detectionLayer.append(box);
  requestAnimationFrame(fitBoxLabels);
}

function renderMarker(detection) {
  const [top, left] = detection.anchor_point_1000 || [500, 500];
  const marker = document.createElement('div');
  marker.className = `detection-marker category-${detection.category}`;
  marker.style.top = `${top / 10}%`;
  marker.style.left = `${left / 10}%`;
  marker.dataset.id = detection.id;
  const label = createLabel(detection, 'marker-label');
  label.addEventListener('click', () => showDetails(detection));
  marker.append(label);
  elements.detectionLayer.append(marker);
}

function createLabel(detection, className) {
  const label = document.createElement('button');
  label.type = 'button';
  label.className = className;
  label.setAttribute('aria-label', `${detection.label}, ${Math.round(detection.confidence * 100)} percent confidence. Open details.`);
  label.innerHTML = `<strong>${escapeHtml(detection.label)}</strong><em>${Math.round(detection.confidence * 100)}%</em>`;
  return label;
}

function fitBoxLabels() {
  document.querySelectorAll('.detection-box').forEach((box) => {
    const label = box.querySelector('.box-label');
    if (!label || !box.offsetWidth || !box.offsetHeight) return;
    const shortestSide = Math.min(box.offsetWidth, box.offsetHeight);
    const fontSize = Math.max(7, Math.min(11, Math.round(shortestSide / 9)));
    const maxWidth = Math.max(52, Math.min(155, Math.round(Math.max(box.offsetWidth * 1.1, shortestSide * 2.5))));
    label.style.fontSize = `${fontSize}px`;
    label.style.maxWidth = `${maxWidth}px`;
    label.classList.toggle('is-compact', shortestSide < 38);
  });
}

function showDetails(detection) {
  document.querySelectorAll('.detection-box, .detection-marker').forEach((item) => item.classList.toggle('is-active', item.dataset.id === detection.id));
  elements.detailTitle.textContent = detection.label;
  elements.detailConfidence.textContent = `${Math.round(detection.confidence * 100)}% confidence`;
  elements.detailCategory.textContent = categoryLabels[detection.category] || detection.category.replaceAll('_', ' ');
  elements.detailCondition.textContent = formatCondition(detection.condition);
  elements.detailAction.textContent = formatAction(detection.action);
  elements.detailDescription.textContent = detection.description;
  elements.componentDetails.hidden = false;
  elements.componentDetails.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeDetails() {
  elements.componentDetails.hidden = true;
  document.querySelectorAll('.detection-box, .detection-marker').forEach((item) => item.classList.remove('is-active'));
}

function setLoading(isLoading) {
  elements.analyzeButton.disabled = isLoading || !selectedFile;
  elements.analyzeButton.classList.toggle('is-loading', isLoading);
  elements.loadingState.hidden = !isLoading;
  elements.resultView.hidden = isLoading || !selectedFile;
  if (isLoading) setState('Analyzing image', 'analyzing');
}

function setState(text, state) {
  elements.analysisState.className = `state-pill ${state}`;
  elements.analysisState.innerHTML = '<span></span>';
  elements.analysisState.append(document.createTextNode(text));
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1]);
    reader.onerror = () => reject(new Error('Could not read that image.'));
    reader.readAsDataURL(file);
  });
}

function formatFileSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.round(bytes / 1024)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatAction(action) {
  return { none: 'Observed', review_photo: 'Review photo', field_verify: 'Field verify' }[action] || 'Review photo';
}

function formatCondition(condition) {
  return { appears_normal: 'Appears normal', needs_review: 'Needs review', not_assessable: 'Not assessable' }[condition] || 'Not assessable';
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('is-visible');
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove('is-visible'), 5200);
}

function escapeHtml(value) {
  const span = document.createElement('span');
  span.textContent = String(value || '');
  return span.innerHTML;
}
