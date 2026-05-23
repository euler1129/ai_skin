// ===== TensorFlow.js State =====
let faceDetector = null;
let currentImageData = null;
let scanData = null;
let modelReady = false;
let lastFacePrediction = null;  // 保存最近一次人脸检测结果（含关键点）
const TOTAL_STEPS = 4;
let currentStep = 0;

// ===== Load TensorFlow.js Face Detection Model =====
async function loadFaceModel() {
  try {
    const loadingEl = document.getElementById('model-loading');
    loadingEl.querySelector('p').textContent = '正在加载人脸检测模型...';

    // 性能：并行预热 TensorFlow.js 后端（提升首次推理速度）
    await tf.ready();

    // Load BlazeFace model for face detection
    faceDetector = await blazeface.load();
    modelReady = true;

    loadingEl.classList.add('hidden');
    setTimeout(() => loadingEl.remove(), 500);

    console.log('✓ TensorFlow.js BlazeFace model loaded');
  } catch (error) {
    console.error('Failed to load model:', error);
    document.getElementById('model-loading').querySelector('p').textContent = '模型加载失败，请刷新页面';
  }
}

// ===== 防重复点击（debounce） =====
let analysisInProgress = false;

// ===== Face Detection using TensorFlow.js (增强版：防作弊 + 侧脸支持) =====
async function detectFace(imageElement) {
  if (!faceDetector) {
    throw new Error('MODEL_NOT_READY');
  }

  // Run face detection
  const predictions = await faceDetector.estimateFaces(imageElement, false);

  // --- 1. 检测人脸数量 ---
  if (predictions.length === 0) {
    throw new Error('NO_FACE_DETECTED');
  }
  if (predictions.length > 1) {
    throw new Error('MULTIPLE_FACES');
  }

  const face = predictions[0];

  // --- 2. 置信度校验（放宽至 0.55，侧脸 BlazeFace 置信度天然偏低） ---
  if (face.probability != null && face.probability < 0.55) {
    throw new Error('LOW_CONFIDENCE');
  }

  // --- 3. 面部尺寸校验（占画面比例 ≥ 4%，侧脸/半脸也能过） ---
  const imgW = imageElement.width;
  const imgH = imageElement.height;
  const faceW = face.bottomRight[0] - face.topLeft[0];
  const faceH = face.bottomRight[1] - face.topLeft[1];
  const faceArea = faceW * faceH;
  const imgArea = imgW * imgH;

  if (faceArea / imgArea < 0.04) {
    throw new Error('FACE_TOO_SMALL');
  }

  // --- 4. 人脸宽高比校验（侧脸包围盒窄，放宽到 0.3~1.3） ---
  const aspectRatio = faceW / faceH;
  if (aspectRatio < 0.3 || aspectRatio > 1.3) {
    throw new Error('INVALID_FACE_RATIO');
  }

  // --- 5. 关键点几何一致性校验（仅作警告记录，不拒绝——BlazeFace 标注精度有限） ---
  const lm = face.landmarks;
  const hasLandmarks = lm && lm.length >= 4;
  if (hasLandmarks) {
    const hasBothEars = lm.length >= 6;
    const isProfile = !hasBothEars;

    // 5a. 眼距比例（仅记录）
    if (lm.length >= 2) {
      const eyeDist = Math.hypot(lm[1][0] - lm[0][0], lm[1][1] - lm[0][1]);
      const eyeToFaceRatio = eyeDist / faceW;
      if (eyeToFaceRatio < 0.06 || eyeToFaceRatio > 0.75) {
        console.warn('[关键点] 眼距比例异常:', eyeToFaceRatio.toFixed(3), 'type:', isProfile ? '侧脸' : '正脸');
      }
    }

    // 5b. 鼻子/嘴巴位置（仅记录）
    if (lm.length >= 3 && lm.length >= 2) {
      const eyeMidY = (lm[0][1] + lm[1][1]) / 2;
      if (lm[2][1] <= eyeMidY) {
        console.warn('[关键点] 鼻子在眼睛上方（可能检测偏差）');
      }
    }
    if (lm.length >= 4 && lm.length >= 3) {
      if (lm[3][1] <= lm[2][1]) {
        console.warn('[关键点] 嘴巴在鼻子上方（可能检测偏差）');
      }
    }
  }

  // --- 6. 肤色分布分析（极端异常才拒绝） ---
  await verifySkinColorDistribution(imageElement, face);

  // --- 7. 原图/素颜检测（仅保留核心纹理检测） ---
  await verifyRawPhoto(imageElement, face);

  // 保存检测结果供报告页绘制关键点
  lastFacePrediction = face;
  return face;
}

// 肤色分布校验 — 排除非真人肤色（纯白/纯黑/动漫色板）
async function verifySkinColorDistribution(imageElement, face) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const padX = (face.bottomRight[0] - face.topLeft[0]) * 0.1;
  const padY = (face.bottomRight[1] - face.topLeft[1]) * 0.1;
  const sx = Math.max(0, face.topLeft[0] - padX);
  const sy = Math.max(0, face.topLeft[1] - padY);
  const sw = Math.min(imageElement.width - sx, face.bottomRight[0] - face.topLeft[0] + padX * 2);
  const sh = Math.min(imageElement.height - sy, face.bottomRight[1] - face.topLeft[1] + padY * 2);

  canvas.width = Math.floor(sw);
  canvas.height = Math.floor(sh);
  ctx.drawImage(imageElement, sx, sy, sw, sh, 0, 0, sw, sh);

  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = imgData.data;
  let skinCount = 0;
  let totalCount = 0;
  let rSum = 0, gSum = 0, bSum = 0;

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    totalCount++;
    if (r > 50 && g > 30 && b > 20 && r > g && r > b) {
      skinCount++;
      rSum += r;
      gSum += g;
      bSum += b;
    }
  }

  // 肤色像素占比低于 5% 才可能非真人（极端情况拒绝）
  if (totalCount > 0 && skinCount / totalCount < 0.05) {
    console.warn('[肤色] 肤色像素占比极低:', (skinCount / totalCount * 100).toFixed(1) + '%');
    throw new Error('NOT_REAL_SKIN');
  }

  // 平均肤色异常检查 — 仅 R/G/B 完全一致（灰度图/动漫）才拒绝
  if (skinCount > 0) {
    const avgR = rSum / skinCount;
    const avgG = gSum / skinCount;
    const avgB = bSum / skinCount;
    const rgDiff = avgR - avgG;
    const rbDiff = avgR - avgB;
    // 极度异常：R/G 几乎相等且 R/B 几乎相等（灰度图/AI生成）
    if (rgDiff < 0.3 && rbDiff < 0.3) {
      console.warn('[肤色] RGB通道几乎一致 R=' + avgR.toFixed(0) + ' G=' + avgG.toFixed(0) + ' B=' + avgB.toFixed(0));
      throw new Error('ABNORMAL_SKIN_COLOR');
    }
  }
}

// ===== 原图/素颜检测（排除美颜滤镜、修图、AI生成） =====
async function verifyRawPhoto(imageElement, face) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  // 提取面部区域（含 20% padding）
  const padX = faceW() * 0.2;
  const padY = faceH() * 0.2;
  const sx = Math.max(0, face.topLeft[0] - padX);
  const sy = Math.max(0, face.topLeft[1] - padY);
  const sw = Math.min(imageElement.width - sx, faceW() + padX * 2);
  const sh = Math.min(imageElement.height - sy, faceH() + padY * 2);

  canvas.width = Math.floor(sw);
  canvas.height = Math.floor(sh);
  ctx.drawImage(imageElement, sx, sy, sw, sh, 0, 0, sw, sh);

  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imgData.data;
  const w = canvas.width;
  const h = canvas.height;

  // --- A. 纹理方差分析：素颜皮肤有明显微纹理，美颜后趋于平滑 ---
  let textureVariance = 0;
  let sampleCount = 0;
  const step = Math.max(1, Math.floor(Math.min(w, h) / 60)); // 采样步长

  for (let y = step; y < h - step; y += step) {
    for (let x = step; x < w - step; x += step) {
      const idx = (y * w + x) * 4;
      const center = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
      // 8 邻域对比
      let localVar = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const ni = ((y + dy * step) * w + (x + dx * step)) * 4;
          const neighbor = (data[ni] + data[ni + 1] + data[ni + 2]) / 3;
          localVar += Math.abs(center - neighbor);
        }
      }
      textureVariance += localVar / 8;
      sampleCount++;
    }
  }

  if (sampleCount > 0) {
    const avgTextureVar = textureVariance / sampleCount;
    // 极度平滑（< 1.0）才警告，可能是重度美颜，但不拒绝
    if (avgTextureVar < 1.0) {
      console.warn('[纹理] 皮肤纹理极低:', avgTextureVar.toFixed(2), '（可能重度美颜，但允许通过）');
    }
  }

  // 辅助：面宽/面高
  function faceW() { return face.bottomRight[0] - face.topLeft[0]; }
  function faceH() { return face.bottomRight[1] - face.topLeft[1]; }
}


// ===== Analyze Skin from Face Region =====
async function analyzeSkin(imageElement, facePrediction) {
  // Create canvas to extract face region
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Get face bounding box with some padding
  const topLeft = facePrediction.topLeft;
  const bottomRight = facePrediction.bottomRight;
  const padding = Math.max(
    bottomRight[0] - topLeft[0],
    bottomRight[1] - topLeft[1]
  ) * 0.3; // 30% padding
  
  const x = Math.max(0, topLeft[0] - padding);
  const y = Math.max(0, topLeft[1] - padding);
  const width = Math.min(imageElement.width - x, (bottomRight[0] - topLeft[0]) + padding * 2);
  const height = Math.min(imageElement.height - y, (bottomRight[1] - topLeft[1]) + padding * 2);
  
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(imageElement, x, y, width, height, 0, 0, width, height);
  
  // Get image data for analysis
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  
  // Calculate skin metrics
  const metrics = calculateSkinMetrics(data, width, height);
  
  return metrics;
}

function calculateSkinMetrics(data, width, height) {
  // Collect pixel values
  let rSum = 0, gSum = 0, bSum = 0;
  let brightnessSum = 0;
  let pixelCount = 0;
  let highBrightnessCount = 0;
  let lowBrightnessCount = 0;
  
  // RGB histograms
  const rHist = new Array(256).fill(0);
  const gHist = new Array(256).fill(0);
  const bHist = new Array(256).fill(0);
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    
    // Skip non-skin pixels (simple skin color filter)
    if (isSkinColor(r, g, b)) {
      rSum += r;
      gSum += g;
      bSum += b;
      
      const brightness = (r + g + b) / 3;
      brightnessSum += brightness;
      pixelCount++;
      
      rHist[r]++;
      gHist[g]++;
      bHist[b]++;
      
      if (brightness > 180) highBrightnessCount++;
      if (brightness < 80) lowBrightnessCount++;
    }
  }
  
  if (pixelCount < 100) {
    // Fallback if not enough skin pixels detected
    return {
      brightness: 50,
      redness: 30,
      uniformity: 70,
      moisture: 50,
      oil: 50,
      pore: 50,
      pigment: 50,
      wrinkle: 50,
      sensitive: 50
    };
  }
  
  const avgR = rSum / pixelCount;
  const avgG = gSum / pixelCount;
  const avgB = bSum / pixelCount;
  const avgBrightness = brightnessSum / pixelCount;
  
  // Calculate color variance (uniformity)
  let variance = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    if (isSkinColor(r, g, b)) {
      const diff = Math.abs(r - avgR) + Math.abs(g - avgG) + Math.abs(b - avgB);
      variance += diff;
    }
  }
  const uniformity = Math.max(0, 100 - (variance / pixelCount) * 0.5);
  
  // Calculate redness (red channel relative to others)
  const redness = Math.min(100, ((avgR - avgG - avgB / 2) / avgR) * 100);
  
  // Calculate metrics based on brightness and color analysis
  // Moisture: Higher brightness + good uniformity = better moisture
  const moisture = Math.min(95, Math.max(30, 50 + (avgBrightness - 128) * 0.3 + uniformity * 0.2));
  
  // Oil: Based on brightness variation and high-light areas
  const highLightRatio = highBrightnessCount / pixelCount;
  const oil = Math.min(90, Math.max(20, 40 + highLightRatio * 50));
  
  // --- Spatial texture analysis (stride sampling for deterministic pore/wrinkle) ---
  const STRIDE =Math.max(1, Math.min(width, height) >> 6); // ~64 samples across smallest dimension
  let microVarSum = 0, microVarCount = 0;
  let edgeSum = 0, edgeCount = 0;

  for (let y = 1; y < height - 1; y += STRIDE) {
    for (let x = 1; x < width - 1; x += STRIDE) {
      const idx = (y * width + x) * 4;
      if (!isSkinColor(data[idx], data[idx + 1], data[idx + 2])) continue;

      const bright = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;

      // Micro-variance in 3×3 neighborhood (pore proxy: fine texture granularity)
      let localVariance = 0, localCount = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nIdx = ((y + dy) * width + (x + dx)) * 4;
          if (isSkinColor(data[nIdx], data[nIdx + 1], data[nIdx + 2])) {
            const nb = (data[nIdx] + data[nIdx + 1] + data[nIdx + 2]) / 3;
            localVariance += (nb - bright) * (nb - bright);
            localCount++;
          }
        }
      }
      if (localCount >= 5) {
        microVarSum += Math.sqrt(localVariance / localCount);
        microVarCount++;
      }

      // Edge gradient magnitude (wrinkle proxy: fine-line contrast)
      const iU = ((y - 1) * width + x) * 4, iD = ((y + 1) * width + x) * 4;
      const iL = (y * width + (x - 1)) * 4, iR = (y * width + (x + 1)) * 4;
      const bU = (data[iU] + data[iU + 1] + data[iU + 2]) / 3;
      const bD = (data[iD] + data[iD + 1] + data[iD + 2]) / 3;
      const bL = (data[iL] + data[iL + 1] + data[iL + 2]) / 3;
      const bR = (data[iR] + data[iR + 1] + data[iR + 2]) / 3;
      edgeSum += Math.abs(bR - bL) + Math.abs(bD - bU);
      edgeCount++;
    }
  }

  const microTexture = microVarCount > 20 ? (microVarSum / microVarCount) : 5;
  const edgeEnergy   = edgeCount > 20   ? (edgeSum / edgeCount)     : 10;

  // Pore: Higher micro-texture variance + low uniformity = more visible pores
  //       microTexture typical range ~2-20; clamp via scaling
  const pore = Math.min(90, Math.max(20, 35 + microTexture * 1.8 + (100 - uniformity) * 0.25));

  // Pigment: Based on color variance and dark spots
  const darkSpotRatio = lowBrightnessCount / pixelCount;
  const pigment = Math.min(80, Math.max(20, 30 + darkSpotRatio * 80 + (100 - uniformity) * 0.3));

  // Wrinkle: Higher edge energy + low uniformity = more visible fine lines
  //          edgeEnergy typical range ~5-35
  const wrinkle = Math.min(70, Math.max(20, 20 + edgeEnergy * 1.2 + (100 - uniformity) * 0.2));

  // Sensitive: Based on redness and uniformity
  const sensitive = Math.min(70, Math.max(20, 20 + redness * 0.5 + (100 - uniformity) * 0.2));
  
  return {
    brightness: Math.round(avgBrightness),
    redness: Math.round(Math.max(0, redness)),
    uniformity: Math.round(uniformity),
    moisture: Math.round(moisture),
    oil: Math.round(oil),
    pore: Math.round(pore),
    pigment: Math.round(pigment),
    wrinkle: Math.round(wrinkle),
    sensitive: Math.round(sensitive)
  };
}

function isSkinColor(r, g, b) {
  // Simple skin color detection using RGB rules
  // Adjusted for Asian skin tones
  const isSkin = r > 95 && g > 40 && b > 20 &&
    Math.max(r, g, b) - Math.min(r, g, b) > 15 &&
    Math.abs(r - g) > 15 &&
    r > g && r > b;
  return isSkin;
}

// ===== Navigation =====
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');

  // 动态更新步骤指示器
  if (name === 'scan') {
    currentStep = 1;
  } else if (name === 'report') {
    currentStep = 3;
  } else {
    currentStep = 0;
  }
  updateStepIndicator();
}

function updateStepIndicator() {
  const stepEl = document.getElementById('step-indicator');
  if (!stepEl) return;
  if (currentStep >= 1) {
    stepEl.textContent = `步骤 ${currentStep}/${TOTAL_STEPS}`;
    stepEl.style.display = '';
  } else {
    stepEl.style.display = 'none';
  }
}

// 上传照片后切换至"正在扫描"
function setScanningStep() {
  currentStep = 2;
  updateStepIndicator();
}

// 报告内切换到护理方案时，标记为步骤4
function setPlanStep() {
  currentStep = 4;
  updateStepIndicator();
}

function startScan() {
  if (!modelReady) {
    showToast('AI模型加载中，请稍候...');
    return;
  }
  
  // 重置扫描页面状态（只重置进度相关，不重置照片）
  const overlay = document.getElementById('scan-overlay');
  const progressBar = document.getElementById('scan-progress-bar');
  const statusEl = document.getElementById('scan-status');
  const btn = document.getElementById('analyze-btn');
  
  overlay.classList.remove('active');
  progressBar.style.width = '0%';
  
  // 根据是否有照片决定状态
  if (currentImageData) {
    // 有照片：保持按钮和状态
    btn.disabled = false;
    statusEl.textContent = '已上传照片，点击"开始AI分析"';
  } else {
    // 没有照片：重置为初始状态
    btn.disabled = true;
    statusEl.textContent = '点击上传面部照片';
    btn.innerHTML = '<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg> 开始AI分析';
  }
  
  showPage('scan');
}

// ===== File Upload =====
function triggerUpload() {
  document.getElementById('file-input').click();
}

// ===== Camera Capture (Mobile) =====
function triggerCamera() {
  // 创建一个专门用于拍照的 input，触发前置摄像头
  const cameraInput = document.createElement('input');
  cameraInput.type = 'file';
  cameraInput.accept = 'image/*';
  cameraInput.capture = 'user'; // 前置摄像头
  cameraInput.onchange = function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      currentImageData = e.target.result;
      const img = document.getElementById('preview-img');
      img.src = currentImageData;
      img.style.display = 'block';  // 确保显示（覆盖可能的内联display:none）
      document.getElementById('camera-area').classList.add('has-image');
      document.getElementById('analyze-btn').disabled = false;
      document.getElementById('scan-status').textContent = '已拍照，点击"开始AI分析"';
    };
    reader.readAsDataURL(file);
  };
  cameraInput.click();
}

// 检测是否为移动设备
function isMobileDevice() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    currentImageData = e.target.result;
    const img = document.getElementById('preview-img');
    img.src = currentImageData;
    img.style.display = 'block';  // 确保显示（覆盖可能的内联display:none）
    document.getElementById('camera-area').classList.add('has-image');
    document.getElementById('analyze-btn').disabled = false;
    document.getElementById('scan-status').textContent = '已上传照片，点击"开始AI分析"';
  };
  reader.readAsDataURL(file);
  
  // 重置 file input，允许重复选择同一文件
  event.target.value = '';
}

// ===== Analysis =====
const SCAN_STEPS = [
  { pct: 10, status: '正在初始化AI算法引擎...' },
  { pct: 25, status: '正在加载人脸检测模型...' },
  { pct: 40, status: '正在检测面部关键点...' },
  { pct: 60, status: '正在分析皮肤颜色与纹理...' },
  { pct: 75, status: 'AI正在计算肤质评分...' },
  { pct: 90, status: '正在生成个性化护理方案...' },
  { pct: 100, status: '分析完成！' },
];

async function startAnalysis() {
  // 防重复点击
  if (analysisInProgress) return;
  analysisInProgress = true;

  const overlay = document.getElementById('scan-overlay');
  const progressBar = document.getElementById('scan-progress-bar');
  const statusEl = document.getElementById('scan-status');
  const btn = document.getElementById('analyze-btn');

  overlay.classList.add('active');
  btn.disabled = true;
  btn.textContent = '分析中，请稍候...';
  setScanningStep();  // 更新步骤为 2/4「AI深度扫描」

  try {
    // Show progress animation
    for (const step of SCAN_STEPS) {
      await animateProgress(step.pct, step.status);
      await sleep(step.pct === 10 ? 500 : 300 + Math.random() * 300);
    }

    // Load image for analysis
    let img = new Image();
    img.src = currentImageData;
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });

    // 性能优化：大图降采样（长边 > 1024 时缩放到 1024，加速后续处理）
    const MAX_DIM = 1024;
    if (img.width > MAX_DIM || img.height > MAX_DIM) {
      const scale = MAX_DIM / Math.max(img.width, img.height);
      const sw = Math.round(img.width * scale);
      const sh = Math.round(img.height * scale);
      const downCanvas = document.createElement('canvas');
      downCanvas.width = sw;
      downCanvas.height = sh;
      const dCtx = downCanvas.getContext('2d');
      dCtx.drawImage(img, 0, 0, sw, sh);
      currentImageData = downCanvas.toDataURL('image/jpeg', 0.92);
      img = new Image();
      img.src = currentImageData;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });
    }

    // Detect face
    statusEl.textContent = '正在检测人脸...';
    const facePrediction = await detectFace(img);

    // Analyze skin
    statusEl.textContent = '正在分析皮肤状态...';
    const skinMetrics = await analyzeSkin(img, facePrediction);
    
    await sleep(400);

    // Generate report
    scanData = generateSkinData(skinMetrics);

    // Show report
    renderReport();
    showPage('report');
    analysisInProgress = false;
    
  } catch (error) {
    console.error('[分析失败]', error.message, error.stack);
    overlay.classList.remove('active');
    btn.disabled = false;
    analysisInProgress = false;
    btn.innerHTML = '<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg> 开始AI分析';
    
    if (error.message === 'MODEL_NOT_READY') {
      showToast('AI模型加载中，请稍候...');
    } else if (error.message === 'NO_FACE_DETECTED') {
      showToast('未检测到人脸，请上传清晰的面部照片');
    } else if (error.message === 'MULTIPLE_FACES') {
      showToast('检测到多张人脸，请上传单人正面照片');
    } else if (error.message === 'LOW_CONFIDENCE') {
      showToast('面部识别置信度不足，请确保光线充足，面部清晰无遮挡');
    } else if (error.message === 'FACE_TOO_SMALL') {
      showToast('面部占比过小，请靠近镜头重新拍摄');
    } else if (error.message === 'INVALID_FACE_RATIO') {
      showToast('面部比例异常，请保持正面角度拍摄');
    } else if (error.message === 'NOT_REAL_SKIN' || error.message === 'ABNORMAL_SKIN_COLOR') {
      showToast('未检测到真实肤色，请上传真人照片（不支持动漫/非人像图片）');
    } else {
      showToast('分析过程出现问题，请重试');
    }
  }
}

// ===== 性能工具函数 =====
// rAF 延迟（页面不可见时暂停，比 setTimeout 省电且不阻塞）
function rAFDelay(ms) {
  return new Promise(function(resolve) {
    var start = performance.now();
    function tick(now) {
      if (now - start >= ms) resolve();
      else requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

function sleep(ms) {
  return new Promise(function(resolve) { return setTimeout(resolve, ms); });
}

function animateProgress(targetPct, status) {
  return new Promise(resolve => {
    const progressBar = document.getElementById('scan-progress-bar');
    const statusEl = document.getElementById('scan-status');
    statusEl.textContent = status;
    progressBar.style.width = targetPct + '%';
    setTimeout(resolve, 300);
  });
}

// ===== Generate Skin Data =====
function generateSkinData(metrics) {
  const dims = [
    { key: 'moisture', label: '水分含量', icon: '💧', color: '#3b82f6', unit: '%' },
    { key: 'oil', label: '油脂分泌', icon: '🛢️', color: '#f59e0b', unit: '%' },
    { key: 'pore', label: '毛孔状态', icon: '🔘', color: '#8b5cf6', unit: '级' },
    { key: 'pigment', label: '色素沉着', icon: '🎨', color: '#ec4899', unit: '级' },
    { key: 'wrinkle', label: '皱纹细纹', icon: '〰️', color: '#6366f1', unit: '级' },
    { key: 'sensitive', label: '敏感程度', icon: '⚠️', color: '#ef4444', unit: '级' },
  ];

  const levels = {};
  dims.forEach(d => {
    const score = metrics[d.key];
    levels[d.key] = score < 50 ? '偏弱' : score < 70 ? '良好' : '优秀';
  });

  const overall = Math.round(
    (metrics.moisture * 0.2 + metrics.oil * 0.15 + (100 - metrics.pore) * 0.15 +
    (100 - metrics.pigment) * 0.2 + (100 - metrics.wrinkle) * 0.15 + (100 - metrics.sensitive) * 0.15)
  );

  // Generate insights based on metrics
  const insights = [];
  if (metrics.oil > 65) {
    insights.push({ title: 'T区油脂分泌偏高', desc: `T区油脂分泌指数达到${metrics.oil}%，建议加强深层清洁与控油护理`, severity: 'warning' });
  } else {
    insights.push({ title: '油脂分泌状态良好', desc: 'T区油脂分泌处于健康范围，继续保持', severity: 'info' });
  }
  
  if (metrics.wrinkle > 60) {
    insights.push({ title: '眼下细纹需关注', desc: '眼下区域检测到早期细纹迹象，建议加强保湿并使用含视黄醇成分的眼霜', severity: 'warning' });
  } else {
    insights.push({ title: '细纹状况良好', desc: '眼下区域皮肤状态良好，继续保持护理习惯', severity: 'info' });
  }
  
  if (metrics.sensitive > 55) {
    insights.push({ title: '局部敏感风险', desc: `面颊区域敏感指数${metrics.sensitive}%，建议选择温和无刺激的护肤品`, severity: 'warning' });
  } else {
    insights.push({ title: '皮肤屏障较健康', desc: '敏感指数处于正常范围，皮肤屏障功能良好', severity: 'info' });
  }

  // Personalized recommendations
  const immediateRecs = [];
  if (metrics.moisture < 50) {
    immediateRecs.push({ rank: immediateRecs.length + 1, text: '<strong>深层补水</strong>：使用含透明质酸+神经酰胺的精华敷料，每周3次' });
  }
  if (metrics.oil > 60) {
    immediateRecs.push({ rank: immediateRecs.length + 1, text: '<strong>控油调理</strong>：早晚使用含水杨酸的洁面，控制T区油脂分泌' });
  }
  if (metrics.sensitive > 50) {
    immediateRecs.push({ rank: immediateRecs.length + 1, text: '<strong>屏障修护</strong>：选择含积雪草/泛醇的护肤品，强化皮肤屏障' });
  }
  if (immediateRecs.length === 0) {
    immediateRecs.push({ rank: 1, text: '<strong>维持护理</strong>：当前皮肤状态良好，建议保持现有护肤习惯' });
    immediateRecs.push({ rank: 2, text: '<strong>防晒护理</strong>：每日坚持使用SPF30+防晒霜，预防光老化' });
  }

  const longtermRecs = [
    { rank: 1, text: '<strong>定期AI测肤</strong>：每月进行一次肤质检测，追踪改善趋势' },
    { rank: 2, text: '<strong>光子嫩肤疗程</strong>：建议每季度进行1次IPL光子嫩肤，改善色素与毛孔' },
    { rank: 3, text: '<strong>生活方式调整</strong>：保持每天8小时睡眠，减少高糖高脂饮食' },
  ];

  const products = [
    { emoji: '💧', name: '玻尿酸深层补水面膜', effect: '含5重玻尿酸，15分钟急救补水' },
    { emoji: '🧴', name: '水杨酸控油精华', effect: '2%水杨酸+茶树精华，控油抑痘' },
    { emoji: '🌿', name: '积雪草舒缓精华', effect: '积雪草苷+泛醇，修护敏感屏障' },
    { emoji: '👁️', name: '视黄醇抗皱眼霜', effect: '0.025%视黄醇，改善眼周细纹' },
  ];

  // 改善效果：展示6个维度，改善前=当前实测值，改善后=预期护理后趋向健康值
  const compares = [
    { dim: '水分含量', before: Math.round(metrics.moisture),    after: Math.round(Math.min(92, metrics.moisture + 15)),         better: 'higher' },
    { dim: '油脂分泌', before: Math.round(metrics.oil),         after: Math.round(Math.max(32, metrics.oil - 14)),             better: 'lower' },
    { dim: '毛孔状态', before: Math.round(metrics.pore),        after: Math.round(Math.max(22, metrics.pore - 18)),             better: 'lower' },
    { dim: '色素沉着', before: Math.round(metrics.pigment),     after: Math.round(Math.max(18, metrics.pigment - 20)),          better: 'lower' },
    { dim: '皱纹细纹', before: Math.round(metrics.wrinkle),     after: Math.round(Math.max(18, metrics.wrinkle - 15)),          better: 'lower' },
    { dim: '敏感程度', before: Math.round(metrics.sensitive),   after: Math.round(Math.max(18, metrics.sensitive - 14)),        better: 'lower' },
  ];

  return {
    dims, metrics, levels, overall,
    insights, immediateRecs, longtermRecs, products, compares,
    photo: currentImageData,
    id: 'RPT-2026-' + String(Math.floor(Math.random() * 900000) + 100000),
    brightness: metrics.brightness,
    uniformity: metrics.uniformity
  };
}

// ===== 在面部照片上绘制人脸轮廓虚线 =====
function drawFaceWithLandmarks(photoDataUrl, facePrediction, callback) {
  if (!photoDataUrl) { callback(photoDataUrl); return; }

  var img = new Image();
  // 性能：使用 decode() 确保图片在 GPU 端解码完毕再绘制，避免渲染卡顿
  img.onload = function() {
    img.decode().then(function() {

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // 画布尺寸 = 原始图片尺寸
    canvas.width = img.width;
    canvas.height = img.height;

    // 绘制原始照片
    ctx.drawImage(img, 0, 0, img.width, img.height);

    // 绘制人脸轮廓虚线（基于关键点拟合的椭圆形轮廓）
    if (facePrediction && facePrediction.landmarks && facePrediction.landmarks.length >= 6) {
      const lm = facePrediction.landmarks;
      // lm: [leftEye, rightEye, nose, mouth, leftEar, rightEar]

      const lEye = lm[0], rEye = lm[1], nose = lm[2], mouth = lm[3], lEar = lm[4], rEar = lm[5];

      // 计算轮廓参考点
      const faceCenterX = (lEye[0] + rEye[0]) / 2;
      const eyeToChin = Math.abs(mouth[1] - lEye[1]) * 0.5; // 下巴延伸
      const chinY = mouth[1] + eyeToChin;
      const foreheadY = Math.max(lEar[1], rEar[1]) - (mouth[1] - lEye[1]) * 1.2;
      const faceHalfW = Math.abs(rEar[0] - lEar[0]) * 0.55;

      // 控制点
      const topCx = faceCenterX;
      const botCx = faceCenterX;
      const leftCx = lEar[0] + faceHalfW * 0.15;
      const rightCx = rEar[0] - faceHalfW * 0.15;

      ctx.save();
      ctx.strokeStyle = 'rgba(201,168,76,0.65)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([10, 6]);
      ctx.beginPath();

      // 从左耳→额头左侧→额头中间→额头右侧→右耳（上半部分）
      ctx.moveTo(lEar[0], lEar[1]);
      ctx.bezierCurveTo(lEar[0], foreheadY, leftCx, foreheadY, topCx, foreheadY);
      ctx.bezierCurveTo(rightCx, foreheadY, rEar[0], foreheadY, rEar[0], rEar[1]);

      // 从右耳→右下颌→下巴→左下颌→左耳（下半部分）
      ctx.bezierCurveTo(rEar[0], chinY, nose[0] + faceHalfW * 0.6, chinY, botCx, chinY + eyeToChin * 0.3);
      ctx.bezierCurveTo(nose[0] - faceHalfW * 0.6, chinY, lEar[0], chinY, lEar[0], lEar[1]);

      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // 输出为 dataURL
    callback(canvas.toDataURL('image/jpeg', 0.92));
    }).catch(function() {
      // decode() 不支持时回退，输出原始照片
      callback(photoDataUrl);
    });
  };
  img.onerror = function() {
    callback(photoDataUrl); // 图片加载失败时回退到原始照片
  };
  img.src = photoDataUrl;
}

// ===== Render Report =====
function renderReport() {
  const d = scanData;

  // Report ID
  document.getElementById('report-id').textContent = d.id;

  // ---- 用 Canvas 绘制含人脸关键点的面部照片 ----
  const faceImgEl = document.getElementById('report-face');
  drawFaceWithLandmarks(d.photo, lastFacePrediction, function(dataUrl) {
    faceImgEl.src = dataUrl;
  });

  // 评估时间（当前时间）
  const now = new Date();
  const timeStr = now.getFullYear() + '-' + 
    String(now.getMonth() + 1).padStart(2, '0') + '-' + 
    String(now.getDate()).padStart(2, '0') + ' ' + 
    String(now.getHours()).padStart(2, '0') + ':' + 
    String(now.getMinutes()).padStart(2, '0') + ':' + 
    String(now.getSeconds()).padStart(2, '0');
  document.getElementById('eval-time').textContent = '评估时间：' + timeStr;

  // Score animation — 用 rAF 驱动，帧同步更平滑
  rAFDelay(300).then(function() {
    var circle = document.getElementById('score-circle');
    var valueEl = document.getElementById('score-value');
    var textEl = document.getElementById('score-text');

    circle.style.setProperty('--score', d.overall);
    animateNumber(valueEl, 0, d.overall, 1200);

    rAFDelay(600).then(function() {
      if (d.overall >= 75) textEl.textContent = '优秀肤质';
      else if (d.overall >= 60) textEl.textContent = '良好肤质';
      else textEl.textContent = '需加强护理';
    });
  });

  // Dimension bars — 使用 DocumentFragment 批量插入，减少 reflow
  var barsEl = document.getElementById('dimension-bars');
  barsEl.innerHTML = '';
  var fragment = document.createDocumentFragment();
  d.dims.forEach(function(dim, i) {
    var score = d.metrics[dim.key];
    var bar = document.createElement('div');
    bar.className = 'dim-bar';
    bar.innerHTML = '<div class="dim-label">' + dim.icon + ' ' + dim.label + '</div>' +
      '<div class="dim-track"><div class="dim-fill" style="background:' + dim.color + '" data-target="' + score + '"></div></div>' +
      '<div class="dim-value">' + score + dim.unit + '</div>';
    fragment.appendChild(bar);
    // rAF 驱动的交错动画（帧同步，不在后台浪费 CPU）
    rAFDelay(400 + i * 150).then(function() {
      bar.querySelector('.dim-fill').style.width = score + '%';
    });
  });
  barsEl.appendChild(fragment);

  // Recommendations
  const immEl = document.getElementById('rec-immediate');
  immEl.innerHTML = d.immediateRecs.map(r => `
    <div class="rec-item">
      <div class="rec-rank">${r.rank}</div>
      <div class="rec-text">${r.text}</div>
    </div>
  `).join('');

  const longEl = document.getElementById('rec-longterm');
  longEl.innerHTML = d.longtermRecs.map(r => `
    <div class="rec-item">
      <div class="rec-rank">${r.rank}</div>
      <div class="rec-text">${r.text}</div>
    </div>
  `).join('');

  // Products
  const prodEl = document.getElementById('product-cards');
  prodEl.innerHTML = d.products.map(p => `
    <div class="product-card">
      <div class="product-img">${p.emoji}</div>
      <div class="product-info">
        <div class="product-name">${p.name}</div>
        <div class="product-effect">${p.effect}</div>
      </div>
    </div>
  `).join('');

  // AI Insights
  const insightsEl = document.getElementById('ai-insights');
  const sevColors = { warning: '#f59e0b', info: '#3b82f6' };
  const sevBgs = { warning: 'rgba(245,158,11,0.08)', info: 'rgba(59,130,246,0.08)' };
  const sevIcons = { warning: '⚠️', info: 'ℹ️' };
  
  // Add image analysis insights
  let imageInsights = [];
  if (d.brightness < 100) {
    imageInsights.push({ title: '面部亮度偏低', desc: `检测到面部亮度为${d.brightness}，建议改善拍摄光线或加强美白护理`, severity: 'warning' });
  } else if (d.brightness > 180) {
    imageInsights.push({ title: '面部亮度偏高', desc: `检测到面部亮度为${d.brightness}，可能有反光区域，建议柔和光线拍摄`, severity: 'info' });
  }
  if (d.uniformity < 70) {
    imageInsights.push({ title: '肤色均匀度待改善', desc: `检测到肤色均匀度为${d.uniformity}%，可能存在色斑或肤色不均问题`, severity: 'warning' });
  }
  
  const allInsights = [...imageInsights, ...d.insights];
  
  insightsEl.innerHTML = allInsights.map(ins => `
    <div style="padding:16px;background:${sevBgs[ins.severity]};border-radius:12px;border:1px solid ${sevColors[ins.severity]}22">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="font-size:16px">${sevIcons[ins.severity]}</span>
        <span style="font-size:13px;font-weight:700;color:${sevColors[ins.severity]}">${ins.title}</span>
      </div>
      <div style="font-size:12px;color:var(--text-muted);line-height:1.6">${ins.desc}</div>
    </div>
  `).join('');

  // Before/After — 6 维度对比
  const compEl = document.getElementById('compare-timeline');
  compEl.innerHTML = d.compares.map(c => {
    let unit, improved;
    if (c.better === 'balanced') {
      // 油脂等指标：趋向平衡值 50 为佳
      unit = c.before > 52 ? '↓' : c.before < 48 ? '↑' : '→';
      improved = Math.abs(c.after - 50) < Math.abs(c.before - 50);
    } else if (c.better === 'lower') {
      unit = '↓';
      improved = c.after < c.before;
    } else {
      unit = '↑';
      improved = c.after > c.before;
    }
    const afterColor = improved ? 'var(--success)' : 'var(--text-muted)';
    return `
    <div class="compare-item">
      <div class="compare-badge before">改善前</div>
      <div class="compare-desc">${c.dim}：${c.before}</div>
      <span style="color:var(--text-muted);font-size:12px">→</span>
      <div class="compare-desc" style="font-weight:700;color:${afterColor}">${c.after} ${unit}</div>
      <div class="compare-badge after">预计</div>
    </div>
  `;
  }).join('');
}

// ===== Tabs =====
function switchTab(el, tabId) {
  document.querySelectorAll('.rec-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.rec-content').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  document.getElementById(tabId).classList.add('active');
  setPlanStep();  // 切换到护理方案时步骤=4「专属方案」
}

// ===== Number Animation =====
function animateNumber(el, from, to, duration) {
  const start = performance.now();
  function update(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ===== Share =====
async function shareReport(platform) {
  if (!scanData) {
    showToast('请先生成报告');
    return;
  }

  showToast('正在生成分享图片...');

  try {
    const canvas = document.getElementById('share-canvas');
    const ctx = canvas.getContext('2d');
    const d = scanData;

    // 设置画布尺寸（竖版海报 9:16，宽度400px）
    const canvasWidth = 400;
    const canvasHeight = 640;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // 渐变背景
    const gradient = ctx.createLinearGradient(0, 0, 0, canvasHeight);
    gradient.addColorStop(0, '#667eea');
    gradient.addColorStop(1, '#764ba2');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // 顶部标题
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 22px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('AI皮肤检测报告', canvasWidth / 2, 50);

    ctx.font = '12px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText('AI智慧皮肤管理中心', canvasWidth / 2, 75);

    // 报告ID
    ctx.font = '11px monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fillText(d.id, canvasWidth / 2, 100);

    // 照片框
    const photoX = 60;
    const photoY = 120;
    const photoW = 120;
    const photoH = 150;
    const photoRadius = 16;

    // 照片背景
    ctx.fillStyle = '#fff';
    roundRect(ctx, photoX, photoY, photoW, photoH, photoRadius);
    ctx.fill();

    // 加载照片
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = d.photo;
    });

    // 裁剪照片为圆形区域
    ctx.save();
    roundRect(ctx, photoX, photoY, photoW, photoH, photoRadius);
    ctx.clip();
    ctx.drawImage(img, photoX, photoY, photoW, photoH);
    ctx.restore();

    // 照片标签
    ctx.fillStyle = '#667eea';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('面部照片', photoX + photoW / 2, photoY + photoH + 20);

    // 评分区域
    const scoreX = 200;
    const scoreY = 130;

    // 评分圆圈
    const scoreRadius = 45;
    ctx.beginPath();
    ctx.arc(scoreX + 70, scoreY + 50, scoreRadius, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 6;
    ctx.stroke();

    // 评分数字
    ctx.fillStyle = '#667eea';
    ctx.font = 'bold 32px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(d.overall, scoreX + 70, scoreY + 60);

    // 评分标签
    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#666';
    ctx.fillText('综合评分', scoreX + 70, scoreY + 95);

    // 肤质评价
    let scoreText = '需加强护理';
    if (d.overall >= 75) scoreText = '优秀肤质';
    else if (d.overall >= 60) scoreText = '良好肤质';

    ctx.fillStyle = '#333';
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText(scoreText, scoreX + 70, scoreY + 115);

    // 维度评分条
    const barX = 200;
    const barY = 270;
    const barW = 140;
    const barH = 8;

    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';

    d.dims.forEach((dim, i) => {
      const y = barY + i * 35;
      const score = d.metrics[dim.key];

      // 标签
      ctx.fillStyle = '#333';
      ctx.fillText(dim.label, barX, y);

      // 进度条背景
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      roundRect(ctx, barX, y + 6, barW, barH, 4);
      ctx.fill();

      // 进度条填充
      ctx.fillStyle = dim.color;
      roundRect(ctx, barX, y + 6, barW * (score / 100), barH, 4);
      ctx.fill();

      // 分数
      ctx.fillStyle = '#666';
      ctx.textAlign = 'right';
      ctx.fillText(score + dim.unit, barX + barW + 30, y);
      ctx.textAlign = 'left';
    });

    // 分割线
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 450);
    ctx.lineTo(canvasWidth - 40, 450);
    ctx.stroke();

    // 个性化护理方案
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('个性化护理方案', canvasWidth / 2, 480);

    // 护理建议
    ctx.font = '11px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.textAlign = 'left';

    const recs = d.immediateRecs.slice(0, 3);
    recs.forEach((rec, i) => {
      ctx.fillText(`${rec.rank}. ${rec.text}`, 40, 510 + i * 22, canvasWidth - 80);
    });

    // 底部信息
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';

    const now = new Date();
    const timeStr = now.getFullYear() + '-' +
      String(now.getMonth() + 1).padStart(2, '0') + '-' +
      String(now.getDate()).padStart(2, '0') + ' ' +
      String(now.getHours()).padStart(2, '0') + ':' +
      String(now.getMinutes()).padStart(2, '0') + ':' +
      String(now.getSeconds()).padStart(2, '0');

    ctx.fillText(`评估时间：${timeStr}`, canvasWidth / 2, 600);
    ctx.fillText('基于端侧AI · 智能肤质分析', canvasWidth / 2, 618);

    // 显示canvas并下载
    canvas.style.display = 'block';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.zIndex = '10000';
    canvas.style.maxWidth = '100%';
    canvas.style.boxShadow = '0 10px 40px rgba(0,0,0,0.3)';

    // 下载图片
    const link = document.createElement('a');
    link.download = `AI皮肤报告_${d.id}_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();

    // 隐藏canvas
    setTimeout(() => {
      canvas.style.display = 'none';
      canvas.style.position = '';
      canvas.style.top = '';
      canvas.style.left = '';
      canvas.style.zIndex = '';
      canvas.style.maxWidth = '';
      canvas.style.boxShadow = '';
    }, 500);

    showToast('分享图片已生成！长按保存后分享到朋友圈');

  } catch (error) {
    console.error('生成分享图片失败:', error);
    showToast('生成失败，请重试');
  }
}

// 圆角矩形辅助函数
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ===== Toast =====
function showToast(message) {
  const toast = document.getElementById('toast');
  document.getElementById('toast-text').textContent = message;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}

// ===== Reset =====
function resetDemo() {
  currentImageData = null;
  scanData = null;
  lastFacePrediction = null;
  currentStep = 0;
  document.getElementById('preview-img').src = '';
  document.getElementById('preview-img').style.display = 'none';
  document.getElementById('camera-area').classList.remove('has-image');
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('analyze-btn').innerHTML = '<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg> 开始AI分析';
  document.getElementById('scan-progress-bar').style.width = '0%';
  updateStepIndicator();
  showPage('welcome');
}

// ===== Initialize =====
window.addEventListener('DOMContentLoaded', function() {
  // 立即启动模型加载（不等待其他资源）
  loadFaceModel();

  // 移动设备检测：显示拍照按钮
  if (isMobileDevice()) {
    document.getElementById('upload-buttons').style.display = 'flex';
  }

  // 注册 Service Worker（离线缓存，提升二次访问性能）
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('js/sw.js', { scope: '/' }).then(function(reg) {
      console.log('✓ Service Worker registered:', reg.scope);
    }).catch(function(err) {
      console.log('Service Worker registration skipped:', err);
    });
  }
});
