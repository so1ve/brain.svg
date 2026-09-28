import './style.css';
import { compileSvg, type Network } from './compiler.ts';
import { normalizeInk } from './preprocess.ts';
import model from './model/digits.json';

const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = element<HTMLCanvasElement>('drawing');
const pen = canvas.getContext('2d', { willReadFrequently: true })!;
const output = element<HTMLCanvasElement>('brain-image');
const preview = output.getContext('2d', { willReadFrequently: true })!;
const undo = element<HTMLButtonElement>('undo');
const clear = element<HTMLButtonElement>('clear');
const hint = element('canvas-hint');
const status = element('render-status');
const errorMessage = element('load-error');
const frame = element('result-frame');
const progress = element('recognition-progress');
const createSvg = compileSvg(model as Network);

type Point = { x: number; y: number };
const strokes: Point[][] = [];
let activePointer: number | null = null;
let revision = 0;
let timer: ReturnType<typeof setTimeout>;
let rendering = Promise.resolve();

pen.strokeStyle = pen.fillStyle = '#eff4df';
pen.lineWidth = 24;
pen.lineCap = pen.lineJoin = 'round';

function setBusy(busy: boolean) {
  document.body.classList.toggle('is-recognizing', busy);
  frame.setAttribute('aria-busy', String(busy));
  progress.hidden = !busy;
}

async function render(version: number) {
  const svg = createSvg(normalizeInk(canvas));
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  const image = new Image();
  image.src = url;
  try {
    await image.decode();
    if (version !== revision) return;
    // Obtain rendered pixels first; drawImage below only copies the finished bitmap.
    const bitmap = await createImageBitmap(image);
    try {
      if (version !== revision) return;
      preview.clearRect(0, 0, output.width, output.height);
      preview.drawImage(bitmap, 0, 0);
    } finally {
      bitmap.close();
    }
    // Detect failed browser rendering, without calculating a prediction in JS.
    const [r, g, b, a] = preview.getImageData(30, 230, 1, 1).data;
    if (a !== 255 || r !== g || g !== b)
      throw new Error('SVG rendering failed. Please try drawing again.');
    status.textContent = strokes.length ? 'Recognized' : 'Ready';
  } catch (error) {
    if (version !== revision) return;
    status.textContent = 'Rendering failed';
    errorMessage.textContent =
      error instanceof Error ? error.message : 'SVG rendering failed. Please try again.';
    errorMessage.hidden = false;
  } finally {
    URL.revokeObjectURL(url);
    if (version === revision) setBusy(false);
  }
}

function requestRender() {
  const version = ++revision;
  clearTimeout(timer);
  hint.hidden = strokes.length > 0;
  undo.disabled = clear.disabled = activePointer !== null || strokes.length === 0;
  errorMessage.hidden = true;
  setBusy(activePointer === null);
  status.textContent = activePointer === null ? 'Recognizing…' : 'Release to recognize';
  if (activePointer !== null) return;

  timer = setTimeout(async () => {
    // Paint feedback first, serialize expensive renders, and keep only the latest input.
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    });
    await rendering;
    if (version !== revision) return;
    rendering = render(version);
  }, 180);
}

function drawStroke(stroke: Point[]) {
  pen.beginPath();
  if (stroke.length === 1) {
    pen.arc(stroke[0].x, stroke[0].y, 12, 0, Math.PI * 2);
    pen.fill();
  } else {
    pen.moveTo(stroke[0].x, stroke[0].y);
    for (const point of stroke.slice(1)) pen.lineTo(point.x, point.y);
    pen.stroke();
  }
}

function position(event: PointerEvent): Point {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

function extendStroke(event: PointerEvent) {
  const stroke = strokes.at(-1)!;
  const next = position(event);
  drawStroke([stroke.at(-1)!, next]);
  stroke.push(next);
}

canvas.addEventListener('pointerdown', (event) => {
  if (activePointer !== null || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault();
  activePointer = event.pointerId;
  canvas.setPointerCapture(activePointer);
  strokes.push([position(event)]);
  drawStroke(strokes.at(-1)!);
  requestRender();
});
canvas.addEventListener('pointermove', (event) => {
  if (activePointer !== event.pointerId) return;
  const points = event.getCoalescedEvents();
  for (const point of points.length ? points : [event]) extendStroke(point);
});
function finish(event: PointerEvent) {
  if (event.pointerId !== activePointer) return;
  if (event.type === 'pointerup') extendStroke(event);
  activePointer = null;
  requestRender();
}
canvas.addEventListener('pointerup', finish);
canvas.addEventListener('pointercancel', finish);
canvas.addEventListener('lostpointercapture', finish);

function redraw() {
  pen.clearRect(0, 0, canvas.width, canvas.height);
  for (const stroke of strokes) drawStroke(stroke);
  requestRender();
}
clear.addEventListener('click', () => {
  strokes.length = 0;
  redraw();
});
undo.addEventListener('click', () => {
  strokes.pop();
  redraw();
});

requestRender();
