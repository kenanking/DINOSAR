const copyButton = document.querySelector('#copy-citation');
const copyStatus = document.querySelector('#copy-status');
const copyIcon = copyButton.innerHTML;
const doneIcon = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4 12 9 17 20 6"></polyline></svg>';
copyButton.innerHTML = `<span class="copy-icon-default" aria-hidden="true">${copyIcon}</span><span class="copy-icon-done" aria-hidden="true">${doneIcon}</span>`;
let copyReset;
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    copyStatus.classList.remove('is-error');
    copyStatus.textContent = 'Citation copied.';
    copyButton.classList.add('is-copied');
    copyButton.title = 'Copied';
    clearTimeout(copyReset);
    copyReset = setTimeout(() => {
      copyButton.classList.remove('is-copied');
      copyButton.title = 'Copy BibTeX';
      copyStatus.textContent = '';
    }, 1600);
  } catch {
    clearTimeout(copyReset);
    copyButton.classList.remove('is-copied');
    copyButton.title = 'Copy BibTeX';
    copyStatus.classList.add('is-error');
    copyStatus.textContent = 'Select the BibTeX text below to copy it.';
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
  }
});

const widget = document.querySelector('#dino-widget');
const header = document.querySelector('header');
// Reserve exactly the fixed mobile header height, including wrapped navigation.
let measuredHeaderHeight = 0;
let headerFrame = 0;
const syncHeaderHeight = () => {
  const height = header.offsetHeight;
  if (height === measuredHeaderHeight) return;
  measuredHeaderHeight = height;
  document.documentElement.style.setProperty('--mobile-header-height', `${height}px`);
};
syncHeaderHeight();
new ResizeObserver(() => {
  cancelAnimationFrame(headerFrame);
  headerFrame = requestAnimationFrame(syncHeaderHeight);
}).observe(header);

const dinosaur = document.querySelector('#dinosaur');
const bubble = document.querySelector('#dino-bubble');
const message = document.querySelector('#dino-message');
const highlights = [
  'Rawr! I’m a tiny dino with a big reading list: 7.05 million SAR images in UniSAR-7M. 🦖',
  'My crop trick? Follow the scatterers! CAMC helps my views find informative SAR structures. ✨',
  'No labels on my pretraining menu! I learn from different views of SAR images. 🌱',
  'Small dino, big dino: our ViT-S and ViT-B backbones both trained for 60 epochs. Pick a buddy on Hugging Face! 🤗',
  'A little show-and-tell: ViT-B reaches 77.02% frozen k-NN accuracy on SOC-40. Peek at the charts below! 🔎',
  'Want to see through my eyes? The notebook has SAR examples, patch similarity, and colorful PCA views. 🎨',
  'Still learning! Water is a strong class for DINOSAR in segmentation; Land Use is a tougher puzzle. 🧩'
];
const size = 56;
const idleDelay = 12000;
let highlightIndex = -1;
let x = document.documentElement.clientWidth - size - 8;
let y = window.innerHeight - 100;
let idleTimer;
let automaticTimer;
let automaticPaused = false;
let drag = null;
let suppressClick = false;

function place(nextX, nextY) {
  x = Math.max(0, Math.min(nextX, document.documentElement.clientWidth - size));
  // Never overlap the sticky header, even while dragging or using arrow keys.
  const topLimit = Math.min(header.offsetHeight + 8, window.innerHeight - size - 8);
  y = Math.max(topLimit, Math.min(nextY, window.innerHeight - size - 8));
  widget.style.left = `${x}px`;
  widget.style.top = `${y}px`;
  const width = document.documentElement.clientWidth;
  const bubbleWidth = Math.min(248, width - 80);
  const preferredLeft = x < width / 2 - size / 2 ? x + size + 10 : x - bubbleWidth - 10;
  const bubbleLeft = Math.max(8, Math.min(preferredLeft, width - bubbleWidth - 8));
  widget.style.setProperty('--bubble-x', `${bubbleLeft - x}px`);
  widget.classList.toggle('is-left', x < document.documentElement.clientWidth / 2 - size / 2);
  widget.classList.toggle('is-high', y < 200);
}
function tuck() {
  clearTimeout(idleTimer);
  bubble.hidden = true;
  dinosaur.setAttribute('aria-expanded', 'false');
  widget.classList.add('is-tucked');
  place(x < document.documentElement.clientWidth / 2 - size / 2 ? 0 : document.documentElement.clientWidth - size, y);
}
function scheduleIdle() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => { if (!drag) tuck(); }, idleDelay);
}
function wake(showHighlight = true) {
  widget.classList.remove('is-tucked');
  if (showHighlight) {
    if (highlightIndex < 0) highlightIndex = 0;
    message.textContent = highlights[highlightIndex];
    bubble.hidden = false;
    dinosaur.setAttribute('aria-expanded', 'true');
  }
  scheduleIdle();
}
function nextHighlight() {
  highlightIndex = (highlightIndex + 1) % highlights.length;
  wake();
  dinosaur.classList.remove('is-hopping');
  void dinosaur.offsetWidth;
  dinosaur.classList.add('is-hopping');
}
function scheduleAutomatic(delay = 45000) {
  clearTimeout(automaticTimer);
  if (automaticPaused || document.hidden) return;
  automaticTimer = setTimeout(() => {
    if (!drag && !widget.matches(':hover') && !widget.querySelector(':focus-visible')) nextHighlight();
    scheduleAutomatic();
  }, delay);
}
function dismiss() {
  automaticPaused = true;
  clearTimeout(automaticTimer);
  tuck();
}
place(x, y);
tuck();
widget.hidden = false;
// Commit the initial position before enabling movement transitions.
void widget.offsetWidth;
widget.classList.add('is-positioned');
scheduleAutomatic(3200);
dinosaur.addEventListener('animationend', () => dinosaur.classList.remove('is-hopping'));
dinosaur.addEventListener('click', () => {
  if (suppressClick) { suppressClick = false; return; }
  automaticPaused = false;
  nextHighlight();
  scheduleAutomatic();
});
document.querySelector('#dino-dismiss').addEventListener('click', () => {
  dinosaur.focus({ preventScroll: true });
  dismiss();
});
dinosaur.addEventListener('focus', () => wake());
widget.addEventListener('pointerenter', event => {
  if (event.pointerType === 'mouse') wake();
});
widget.addEventListener('pointermove', () => { if (!drag && !bubble.hidden) scheduleIdle(); });
widget.addEventListener('pointerleave', scheduleIdle);
dinosaur.addEventListener('pointerdown', event => {
  if (!event.isPrimary || event.button !== 0) return;
  suppressClick = false;
  // Use the visible position if a docking transition is still in progress.
  const rect = widget.getBoundingClientRect();
  drag = { id: event.pointerId, startX: event.clientX, startY: event.clientY, x: rect.left, y: rect.top, moved: false };
  widget.classList.add('is-dragging');
  wake(false);
  clearTimeout(idleTimer);
  dinosaur.setPointerCapture(event.pointerId);
});
dinosaur.addEventListener('pointermove', event => {
  if (!drag || drag.id !== event.pointerId) return;
  const dx = event.clientX - drag.startX;
  const dy = event.clientY - drag.startY;
  if (Math.hypot(dx, dy) > 5) drag.moved = true;
  if (drag.moved) {
    bubble.hidden = true;
    dinosaur.setAttribute('aria-expanded', 'false');
    place(drag.x + dx, drag.y + dy);
  }
});
function endDrag(event) {
  if (!drag || drag.id !== event.pointerId) return;
  suppressClick = drag.moved || event.type === 'pointercancel';
  drag = null;
  widget.classList.remove('is-dragging');
  if (dinosaur.hasPointerCapture(event.pointerId)) dinosaur.releasePointerCapture(event.pointerId);
  scheduleIdle();
}
dinosaur.addEventListener('pointerup', endDrag);
dinosaur.addEventListener('pointercancel', endDrag);
dinosaur.addEventListener('lostpointercapture', endDrag);
dinosaur.addEventListener('keydown', event => {
  suppressClick = false;
  if (event.key === 'Escape') { event.preventDefault(); dismiss(); return; }
  const moves = { ArrowLeft: [-24, 0], ArrowRight: [24, 0], ArrowUp: [0, -24], ArrowDown: [0, 24] };
  if (!moves[event.key]) return;
  event.preventDefault();
  wake();
  const [dx, dy] = moves[event.key];
  place(x + dx, y + dy);
});
window.addEventListener('resize', () => {
  if (widget.classList.contains('is-tucked')) {
    // Preserve the docked edge after orientation changes.
    x = widget.classList.contains('is-left') ? 0 : document.documentElement.clientWidth;
    tuck();
  } else place(x, y);
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { clearTimeout(automaticTimer); tuck(); }
  else scheduleAutomatic();
});

if (window.renderMathInElement) {
  renderMathInElement(document.querySelector('main'), {
    delimiters: [{ left: '\\(', right: '\\)', display: false }]
  });
}

const donut = document.querySelector('.donut');
if (donut) {
  const nameEl = donut.querySelector('.donut-name');
  const valueEl = donut.querySelector('.donut-total');
  const shareEl = donut.querySelector('.donut-label');
  const resetDonut = () => {
    nameEl.textContent = 'TOTAL';
    valueEl.textContent = '7,047,666';
    shareEl.textContent = 'SAR IMAGES';
  };
  donut.querySelectorAll('.donut-seg').forEach(seg => {
    const show = () => {
      nameEl.textContent = seg.dataset.name.toUpperCase();
      valueEl.textContent = seg.dataset.value;
      shareEl.textContent = `${seg.dataset.pct} OF CORPUS`;
    };
    seg.addEventListener('pointerenter', show);
    seg.addEventListener('focus', show);
    seg.addEventListener('pointerleave', resetDonut);
    seg.addEventListener('blur', resetDonut);
  });
}

// Highlight paired local views only while a mouse hovers a crop preview.
const cropComparison = document.querySelector('.crop-comparison');
if (cropComparison) {
  const showCrop = (id) => {
    cropComparison.classList.toggle('has-active', Boolean(id));
    cropComparison.querySelectorAll('[data-crop]').forEach(el => {
      el.classList.toggle('is-active', el.dataset.crop === id);
    });
  };
  cropComparison.querySelectorAll('.crop-thumb').forEach(el => {
    el.addEventListener('pointerenter', event => {
      if (event.pointerType === 'mouse') showCrop(el.dataset.crop);
    });
    el.addEventListener('pointerleave', () => showCrop(null));
    el.addEventListener('pointercancel', () => showCrop(null));
  });
  window.addEventListener('blur', () => showCrop(null));
}

// Follow the section crossing the reading line below the navigation.
const sectionLinks = [...document.querySelectorAll('.navlinks a[href^="#"]')];
const readingSections = [...document.querySelectorAll('main > section')];
let activeReadingSection = null;
let navigationFrame = 0;
function updateSectionNavigation() {
  navigationFrame = 0;
  const scrollPadding = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
  const readingLine = Math.max(header.getBoundingClientRect().bottom, scrollPadding) + 24;
  let current = null;
  for (const section of readingSections) {
    const anchorOffset = parseFloat(getComputedStyle(section).scrollMarginTop) || 0;
    if (section.getBoundingClientRect().top - anchorOffset <= readingLine) current = section.id || null;
    else break;
  }
  if (current === activeReadingSection) return;
  activeReadingSection = current;
  sectionLinks.forEach(link => {
    if (link.hash === `#${current}`) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
function scheduleSectionNavigation() {
  if (!navigationFrame) navigationFrame = requestAnimationFrame(updateSectionNavigation);
}
window.addEventListener('scroll', scheduleSectionNavigation, {passive: true});
window.addEventListener('resize', scheduleSectionNavigation);
window.addEventListener('hashchange', scheduleSectionNavigation);
new ResizeObserver(scheduleSectionNavigation).observe(document.querySelector('main'));
scheduleSectionNavigation();
