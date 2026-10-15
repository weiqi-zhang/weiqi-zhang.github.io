/* AEON project page */
(function () {
  'use strict';

  var WORLDS = window.AEON_WORLDS || [];
  var FEATURED = window.AEON_FEATURED || [];
  var BY_ID = {};
  WORLDS.forEach(function (w) { BY_ID[w.id] = w; });

  // Each tri.mp4 holds RGB | depth | normal side by side: 3 x 448 px wide, 81 frames at 16 fps.
  var FPS = 16, FRAMES = 81, POSTER_FRAME = 40, PANEL = 448;
  var PRESETS = { split: [1 / 3, 2 / 3], rgb: [1, 1], depth: [0, 1], normal: [0, 0] };
  var GAP = 12;   // CSS px between panels in the triptych layout

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  function path(id, file) { return 'static/worlds/' + id + '/' + file; }
  function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  /* ---------- Links without a URL yet are shown as "soon" ---------- */
  document.querySelectorAll('a[data-link]').forEach(function (a) {
    var href = (a.getAttribute('href') || '').trim();
    if (!href || href === '#') {
      a.classList.add('is-soon');
      a.setAttribute('aria-disabled', 'true');
      a.removeAttribute('href');
      a.title = 'Coming soon';
    }
  });

  /* ---------- Top bar ---------- */
  var topbar = document.querySelector('.topbar');
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.topbar-links a'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('main > .sec'));
  function updateTopbar() {
    topbar.classList.toggle('is-solid', window.scrollY > 24);
    // Overscroll reveals the root background: dark above the hero, light below the last section.
    var maxScroll = document.documentElement.scrollHeight - window.innerHeight;
    document.documentElement.classList.toggle('ends-light', window.scrollY > maxScroll / 2);
    // The bar takes the light or dark scheme of the section underneath it.
    var probe = document.elementFromPoint(window.innerWidth / 2, 58);
    var sec = probe && probe.closest ? probe.closest('.sec') : null;
    topbar.classList.toggle('topbar--day', !!sec && !sec.classList.contains('sec--night'));
    var current = null;
    sections.forEach(function (s) {
      if (s.id && s.getBoundingClientRect().top <= 120) current = s.id;
    });
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) current = 'bibtex';
    navLinks.forEach(function (a) {
      var on = a.getAttribute('href') === '#' + current;
      if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
    });
  }
  window.addEventListener('scroll', updateTopbar, { passive: true });
  updateTopbar();

  /* ---------- World viewer ---------- */
  function WorldViewer(root, opts) {
    var self = this;
    opts = opts || {};
    this.root = root;
    root.appendChild(document.getElementById('viewer-template').content.cloneNode(true));
    this.stage = root.querySelector('.viewer-stage');
    this.canvas = root.querySelector('.viewer-canvas');
    this.canvas.setAttribute('role', 'img');
    this.ctx = this.canvas.getContext('2d');
    this.labels = Array.prototype.slice.call(root.querySelectorAll('.band-label'));
    this.dividers = Array.prototype.slice.call(root.querySelectorAll('.divider'));
    this.playBtn = root.querySelector('.play');
    this.range = root.querySelector('.time');
    this.ticksDone = root.querySelector('.ticks-done');
    this.readout = root.querySelector('.readout');
    this.modeBtns = Array.prototype.slice.call(root.querySelectorAll('.viewer-modes button'));
    // 'split': one square with draggable RGB | depth | normal bands (hero).
    // 'triptych': the three renderings side by side, each smaller than its native 448 px so it stays sharp (dialog).
    this.mode = opts.layout === 'triptych' ? 'triptych' : 'split';
    root.classList.toggle('viewer--triptych', this.mode === 'triptych');
    this.split = (opts.intro && !reduceMotion) ? [1, 1] : PRESETS.split.slice();
    this.introPending = !!opts.intro && !reduceMotion;
    this.wantPlay = !reduceMotion && opts.autoplay !== false;
    this.visible = true;
    this.suspended = false;
    this.ready = false;
    this.raf = 0;
    this.anim = 0;

    var v = this.video = document.createElement('video');
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'auto';
    v.setAttribute('muted', ''); v.setAttribute('playsinline', ''); v.setAttribute('aria-hidden', 'true');
    v.className = 'viewer-src';           // kept in the DOM (some mobile browsers only decode attached videos)
    this.stage.insertBefore(v, this.canvas);
    v.addEventListener('seeked', function () { self.draw(); });
    v.addEventListener('play', function () { self.loop(); self.updatePlayUI(); });
    v.addEventListener('pause', function () { self.updatePlayUI(); });
    this.poster = new Image();
    this.poster.decoding = 'async';

    // dividers
    this.dividers.forEach(function (d, i) {
      d.addEventListener('pointerdown', function (e) { self.dragDivider(e, i); });
      d.addEventListener('keydown', function (e) {
        var step = e.shiftKey ? 0.1 : 0.02, s = self.split.slice(), x = s[i];
        if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') x -= step;
        else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') x += step;
        else if (e.key === 'Home') x = 0;
        else if (e.key === 'End') x = 1;
        else return;
        e.preventDefault();
        self.setDivider(i, x);
      });
    });

    // modes
    this.modeBtns.forEach(function (b) {
      b.addEventListener('click', function () { self.animateSplit(PRESETS[b.dataset.mode], 520); });
    });

    // play / pause
    this.playBtn.addEventListener('click', function () {
      if (self.video.paused) self.play(); else self.pause();
    });

    // time scrubbing
    var resume = false, scrubbing = false;
    this.range.addEventListener('pointerdown', function () {
      scrubbing = true;
      resume = !self.video.paused;
      self.video.pause();
    });
    this.range.addEventListener('input', function () {
      var f = +self.range.value;
      self.setFrameUI(f);
      if (self.ready) self.seekFrame(f);
    });
    var endScrub = function () {
      if (!scrubbing) return;
      scrubbing = false;
      if (resume && self.wantPlay) self.play();
    };
    this.range.addEventListener('change', endScrub);
    this.range.addEventListener('pointerup', endScrub);
    this.range.addEventListener('pointercancel', endScrub);

    // size and visibility
    if ('ResizeObserver' in window) new ResizeObserver(function () { self.resize(); }).observe(this.stage);
    else window.addEventListener('resize', function () { self.resize(); });
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        self.visible = entries[0].isIntersecting;
        self.syncPlayback();
        if (self.visible && (self.ready || self.poster.complete)) self.maybeIntro();
      }, { threshold: 0.35 }).observe(this.stage);
    }
    document.addEventListener('visibilitychange', function () { self.syncPlayback(); });

    this.resize();
    this.layout();
    this.setFrameUI(POSTER_FRAME);
    this.updatePlayUI();
  }

  WorldViewer.prototype.load = function (world) {
    var self = this, token = {};
    this.token = token;
    this.world = world;
    this.ready = false;
    this.canvas.setAttribute('aria-label', 'Generated world rendered as RGB, depth and normals: ' + world.prompt);
    this.poster.onload = function () { if (self.token === token && !self.ready) { self.draw(); self.maybeIntro(); } };
    this.poster.src = path(world.id, 'tri.webp');
    this.video.src = path(world.id, 'tri.mp4');
    this.video.load();
    this.video.addEventListener('loadeddata', function onData() {
      self.video.removeEventListener('loadeddata', onData);
      if (self.token !== token) return;
      self.ready = true;
      self.seekFrame(POSTER_FRAME);
      self.draw();
      self.maybeIntro();
      self.syncPlayback();
    });
    this.setFrameUI(POSTER_FRAME);
  };

  WorldViewer.prototype.frame = function () {
    return clamp(Math.floor(this.video.currentTime * FPS + 1e-3), 0, FRAMES - 1);
  };
  WorldViewer.prototype.seekFrame = function (f) {
    this.video.currentTime = (f + 0.5) / FPS;
  };
  WorldViewer.prototype.play = function () {
    this.wantPlay = true;
    this.syncPlayback();
    this.updatePlayUI();
  };
  WorldViewer.prototype.pause = function () {
    this.wantPlay = false;
    this.video.pause();
    this.updatePlayUI();
  };
  WorldViewer.prototype.suspend = function (on) {
    this.suspended = on;
    this.syncPlayback();
  };
  WorldViewer.prototype.syncPlayback = function () {
    var self = this;
    var should = this.wantPlay && this.ready && this.visible && !this.suspended && !document.hidden;
    if (should && this.video.paused) {
      var p = this.video.play();
      if (p && p.catch) p.catch(function () { self.wantPlay = false; self.updatePlayUI(); });
    } else if (!should && !this.video.paused) {
      this.video.pause();
    }
  };
  WorldViewer.prototype.loop = function () {
    var self = this;
    cancelAnimationFrame(this.raf);
    var step = function () {
      self.draw();
      self.setFrameUI(self.frame());
      if (!self.video.paused) self.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  };
  WorldViewer.prototype.updatePlayUI = function () {
    var paused = !this.wantPlay;
    this.root.classList.toggle('is-paused', paused);
    this.playBtn.setAttribute('aria-label', paused ? 'Play' : 'Pause');
  };
  WorldViewer.prototype.setFrameUI = function (f) {
    this.range.value = f;
    this.ticksDone.style.setProperty('--p', (f / (FRAMES - 1) * 100).toFixed(3) + '%');
    this.readout.textContent = (f + 1) + ' / ' + FRAMES;
    this.range.setAttribute('aria-valuetext', 'Frame ' + (f + 1) + ' of ' + FRAMES);
  };
  WorldViewer.prototype.setLayout = function (mode) {
    mode = mode === 'triptych' ? 'triptych' : 'split';
    if (mode === this.mode) return;
    this.mode = mode;
    this.root.classList.toggle('viewer--triptych', mode === 'triptych');
    this.resize();
  };
  WorldViewer.prototype.resize = function () {
    var w = this.stage.clientWidth;
    if (!w) return;
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var pw = Math.round(w * dpr), ph = pw;
    if (this.mode === 'triptych') {
      var hCss = Math.max(1, (w - 2 * GAP) / 3);
      this.stage.style.height = hCss + 'px';
      ph = Math.round(hCss * dpr);
    } else {
      this.stage.style.height = '';
    }
    if (this.canvas.width !== pw || this.canvas.height !== ph) { this.canvas.width = pw; this.canvas.height = ph; }
    this.layout();
  };
  WorldViewer.prototype.draw = function () {
    var src = null;
    if (this.ready && this.video.readyState >= 2) src = this.video;
    else if (this.poster.complete && this.poster.naturalWidth) src = this.poster;
    if (!src) return;
    var W = this.canvas.width, H = this.canvas.height, ctx = this.ctx;
    var xs = [0, this.split[0], this.split[1], 1];
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    if (this.mode === 'triptych') {
      var g = Math.round(GAP * W / (this.stage.clientWidth || W)), P = (W - 2 * g) / 3, r = Math.round(4 * W / (this.stage.clientWidth || W));
      for (var t = 0; t < 3; t++) {
        var x0 = Math.round(t * (P + g)), pwid = Math.round(t * (P + g) + P) - x0;
        ctx.save();
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x0, 0, pwid, H, r); else ctx.rect(x0, 0, pwid, H);
        ctx.clip();
        ctx.drawImage(src, t * PANEL, 0, PANEL, PANEL, x0, 0, pwid, H);
        ctx.restore();
      }
      return;
    }
    for (var k = 0; k < 3; k++) {
      var a = xs[k], b = xs[k + 1];
      var dx = Math.round(a * W), dw = Math.round(b * W) - dx;
      if (dw <= 0) continue;
      ctx.drawImage(src, k * PANEL + a * PANEL, 0, (b - a) * PANEL, PANEL, dx, 0, dw, H);
    }
  };
  WorldViewer.prototype.layout = function () {
    var self = this, s = this.split, xs = [0, s[0], s[1], 1], w = this.stage.clientWidth || 1;
    this.dividers.forEach(function (d, i) {
      d.style.left = (s[i] * 100) + '%';
      d.setAttribute('aria-valuenow', String(Math.round(s[i] * 100)));
      // keep the grip fully visible when a boundary sits on the edge of the frame
      d.classList.toggle('at-start', s[i] < 0.03);
      d.classList.toggle('at-end', s[i] > 0.97);
    });
    if (this.mode === 'triptych') {
      var pcss = (w - 2 * GAP) / 3;
      this.labels.forEach(function (l, k) {
        l.style.left = (k * (pcss + GAP)) + 'px';
        l.classList.remove('is-hidden');
      });
      this.draw();
      return;
    }
    this.labels.forEach(function (l, k) {
      l.style.left = (xs[k] * 100) + '%';
      l.classList.toggle('is-hidden', (xs[k + 1] - xs[k]) * w < 74);
    });
    this.modeBtns.forEach(function (b) {
      var p = PRESETS[b.dataset.mode];
      var on = Math.abs(p[0] - s[0]) < 0.01 && Math.abs(p[1] - s[1]) < 0.01;
      b.setAttribute('aria-pressed', String(on));
    });
    this.draw();
  };
  WorldViewer.prototype.setDivider = function (i, x) {
    var s = this.split.slice();
    s[i] = clamp(x, 0, 1);
    if (i === 0 && s[0] > s[1]) s[1] = s[0];
    if (i === 1 && s[1] < s[0]) s[0] = s[1];
    cancelAnimationFrame(this.anim);
    this.split = s;
    this.layout();
  };
  WorldViewer.prototype.dragDivider = function (e, idx) {
    var self = this, d = this.dividers[idx];
    e.preventDefault();
    cancelAnimationFrame(this.anim);
    this.introPending = false;
    var rect = this.stage.getBoundingClientRect();
    var startX = e.clientX, active = idx;
    var overlapped = Math.abs(this.split[0] - this.split[1]) < 0.004;
    d.setPointerCapture(e.pointerId);
    d.classList.add('is-dragging');
    var move = function (ev) {
      if (overlapped) {
        if (Math.abs(ev.clientX - startX) < 3) return;
        // When both boundaries coincide, dragging left opens the band on the left, and vice versa.
        active = ev.clientX < startX ? 0 : 1;
        overlapped = false;
      }
      self.setDivider(active, (ev.clientX - rect.left) / rect.width);
    };
    var up = function () {
      d.classList.remove('is-dragging');
      d.removeEventListener('pointermove', move);
      d.removeEventListener('pointerup', up);
      d.removeEventListener('pointercancel', up);
    };
    d.addEventListener('pointermove', move);
    d.addEventListener('pointerup', up);
    d.addEventListener('pointercancel', up);
  };
  WorldViewer.prototype.animateSplit = function (target, ms) {
    var self = this, from = this.split.slice(), t0 = performance.now();
    cancelAnimationFrame(this.anim);
    if (reduceMotion || !ms) { this.split = target.slice(); this.layout(); return; }
    var step = function (now) {
      var t = clamp((now - t0) / ms, 0, 1), e = easeOutCubic(t);
      self.split = [from[0] + (target[0] - from[0]) * e, from[1] + (target[1] - from[1]) * e];
      self.layout();
      if (t < 1) self.anim = requestAnimationFrame(step);
    };
    this.anim = requestAnimationFrame(step);
  };
  // The page's one orchestrated moment: the dividers open once to reveal depth and normals.
  WorldViewer.prototype.maybeIntro = function () {
    var self = this;
    if (!this.introPending || !this.visible) return;   // wait until the viewer is on screen
    this.introPending = false;
    setTimeout(function () { self.animateSplit(PRESETS.split, 1500); }, 650);
  };

  /* ---------- Caption: condition and prompt ---------- */
  function renderCaption(el, w) {
    var isImage = w.mode === 'image';
    el.innerHTML =
      '<span class="cond">' + (isImage ? 'From a condition image' : 'From a text prompt') + '</span>' +
      '<span class="cond-body">' +
      (isImage ? '<img class="cond-img" src="' + path(w.id, 'input.webp') + '" width="448" height="448" alt="Condition image">' : '') +
      '<q class="prompt">' + esc(w.prompt) + '</q></span>';
  }

  /* ---------- Hero ---------- */
  var heroViewer = null;
  var heroRoot = document.getElementById('hero-viewer');
  if (heroRoot && FEATURED.length) {
    heroViewer = new WorldViewer(heroRoot, { intro: true, autoplay: true });
    var heroCaption = document.getElementById('hero-caption');
    var picker = document.getElementById('hero-picker');
    var selectHero = function (id) {
      var w = BY_ID[id];
      if (!w) return;
      heroViewer.load(w);
      renderCaption(heroCaption, w);
      picker.querySelectorAll('button').forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.dataset.id === id));
      });
    };
    FEATURED.forEach(function (id, i) {
      var w = BY_ID[id];
      if (!w) return;
      var b = document.createElement('button');
      b.type = 'button';
      b.dataset.id = id;
      b.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-label', 'Show world ' + (i + 1) + ': ' + w.prompt.split('.')[0]);
      b.innerHTML = '<img src="' + path(id, 'poster.webp') + '" width="448" height="448" alt="">';
      b.addEventListener('click', function () { selectHero(id); });
      picker.appendChild(b);
    });
    selectHero(FEATURED[0]);
  }

  /* ---------- Gallery ---------- */
  var grid = document.getElementById('world-grid');
  var imageIcon = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M2.5 11.5 6 8l2.5 2.5L10.5 9l3 3"/></svg>';

  function startPreview(btn) {
    if (!finePointer) return;
    var v = btn.querySelector('video');
    if (!v) {
      v = document.createElement('video');
      v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'auto';
      v.setAttribute('muted', ''); v.setAttribute('playsinline', '');
      v.addEventListener('playing', function () { btn.classList.add('is-playing'); });
      v.addEventListener('loadeddata', function onData() {
        v.removeEventListener('loadeddata', onData);
        v.currentTime = (POSTER_FRAME + 0.5) / FPS;   // continue from the poster frame, no jump
      });
      v.addEventListener('seeked', function onSeek() {
        v.removeEventListener('seeked', onSeek);
        if (btn.dataset.hover === '1') { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
      });
      v.src = path(btn.dataset.id, 'tri.mp4');
      btn.insertBefore(v, btn.querySelector('.tile-text'));
    } else if (v.readyState >= 2) {
      var p = v.play(); if (p && p.catch) p.catch(function () {});
    }
  }
  function stopPreview(btn) {
    var v = btn.querySelector('video');
    btn.classList.remove('is-playing');
    if (v) v.pause();
  }

  if (grid) {
    WORLDS.forEach(function (w) {
      var li = document.createElement('li');
      li.innerHTML =
        '<button type="button" class="tile" data-id="' + w.id + '" aria-label="Open world: ' + esc(w.prompt) + '">' +
        '<img src="' + path(w.id, 'poster.webp') + '" width="448" height="448" loading="lazy" decoding="async" alt="">' +
        '<span class="tile-text" aria-hidden="true">' + esc(w.prompt) + '</span>' +
        (w.mode === 'image' ? '<span class="mode-mark" title="From a condition image">' + imageIcon + '</span>' : '') +
        '</button>';
      var btn = li.firstChild;
      btn.addEventListener('pointerenter', function () { btn.dataset.hover = '1'; startPreview(btn); });
      btn.addEventListener('pointerleave', function () { btn.dataset.hover = '0'; stopPreview(btn); });
      btn.addEventListener('click', function () { openDialog(w.id); });
      grid.appendChild(li);
    });
  }

  /* ---------- Dialog viewer ---------- */
  var dlg = document.getElementById('world-dialog');
  var dlgViewer = null, dlgList = [], dlgIndex = 0;
  var dlgCaption = document.getElementById('dialog-caption');
  var dlgCount = document.getElementById('dialog-count');

  function showDialogWorld() {
    var w = BY_ID[dlgList[dlgIndex]];
    dlgViewer.load(w);
    renderCaption(dlgCaption, w);
    dlgCount.textContent = (dlgIndex + 1) + ' / ' + dlgList.length;
  }
  function openDialog(id) {
    if (!dlg || typeof dlg.showModal !== 'function') return;
    dlgList = WORLDS.map(function (w) { return w.id; });
    dlgIndex = Math.max(0, dlgList.indexOf(id));
    if (!dlgViewer) {
      // wide screens: triptych; phones: the split square (three 110 px panels would be too small)
      var wideMq = window.matchMedia('(min-width: 721px)');
      dlgViewer = new WorldViewer(document.getElementById('dialog-viewer'), { autoplay: true, layout: wideMq.matches ? 'triptych' : 'split' });
      var onMq = function (e) { dlgViewer.setLayout(e.matches ? 'triptych' : 'split'); };
      if (wideMq.addEventListener) wideMq.addEventListener('change', onMq); else wideMq.addListener(onMq);
    }
    if (heroViewer) heroViewer.suspend(true);
    dlg.showModal();
    dlg.querySelector('.dialog-close').focus();
    dlgViewer.suspend(false);
    dlgViewer.resize();
    showDialogWorld();
  }
  if (dlg) {
    dlg.querySelectorAll('[data-step]').forEach(function (b) {
      b.addEventListener('click', function () {
        dlgIndex = (dlgIndex + Number(b.dataset.step) + dlgList.length) % dlgList.length;
        showDialogWorld();
      });
    });
    dlg.querySelector('.dialog-close').addEventListener('click', function () { dlg.close(); });
    dlg.addEventListener('click', function (e) { if (e.target === dlg) dlg.close(); });
    dlg.addEventListener('keydown', function (e) {
      var t = e.target;
      if (t && (t.classList.contains('time') || t.classList.contains('divider'))) return;
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        dlgIndex = (dlgIndex + (e.key === 'ArrowRight' ? 1 : -1) + dlgList.length) % dlgList.length;
        showDialogWorld();
      }
    });
    dlg.addEventListener('close', function () {
      if (dlgViewer) dlgViewer.suspend(true);
      if (heroViewer) heroViewer.suspend(false);
    });
  }

  /* ---------- 4DGS primitive illustration ---------- */
  // Follows the representation in the paper: o(t) = o * exp(-(t - t_i)^2 / (2 sigma^2)),
  // sigma = (l_i / 2) / sqrt(-2 ln eps), eps = 0.05; x(t) = x_i + v_i (t - t_i); rotation advances with omega_i.
  (function initExplainer() {
    var svg = document.querySelector('.explainer-svg');
    if (!svg) return;
    var NS = 'http://www.w3.org/2000/svg';
    var EPS = 0.05, K = Math.sqrt(-2 * Math.log(EPS));
    var VBW = 960;                                   // viewBox width = reading width, so labels stay at 1:1
    var X0 = 70, X1 = VBW - 60, BASE = 286, AMP = 74;  // time band
    var SX0 = 40, SX1 = VBW - 40;                      // space band
    var prims = [
      { id: '1', t: 0.24, l: 0.36, x: 0.22, v: 0.62, y: 96, rx: 36, ry: 15, rot: -20, w: 170, color: 'var(--p1)', note: 'moves, short duration', noteDx: -6, noteDy: -30 },
      { id: '2', t: 0.50, l: 1.60, x: 0.53, v: 0.0, y: 126, rx: 86, ry: 26, rot: 0, w: 0, color: 'var(--p2)', note: 'static, long duration', noteDx: 0, noteDy: -40 },
      { id: '3', t: 0.78, l: 0.30, x: 0.84, v: -0.62, y: 70, rx: 30, ry: 13, rot: 26, w: -200, color: 'var(--p3)', note: 'moves, short duration', noteDx: -10, noteDy: -26 }
    ];
    function el(tag, attrs, parent) {
      var n = document.createElementNS(NS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(n);
      return n;
    }
    function tx(t) { return X0 + t * (X1 - X0); }
    function sx(u) { return SX0 + u * (SX1 - SX0); }
    function sigma(p) { return (p.l / 2) / K; }
    function opacity(p, t) { var s = sigma(p); return Math.exp(-((t - p.t) * (t - p.t)) / (2 * s * s)); }
    function sym(x, y, main, sub, anchor) {
      var g = el('g', { class: 'sym-g' });
      var a = el('text', { x: x, y: y, class: 'sym', 'text-anchor': anchor || 'middle' }, g);
      a.textContent = main;
      if (sub) { var b = el('text', { x: x + 6, y: y + 4, class: 'sym-sub' }, g); b.textContent = sub; }
      return g;
    }

    var defs = el('defs', {});
    prims.forEach(function (p) {
      var g = el('radialGradient', { id: 'splat-' + p.id }, defs);
      [[0, 1], [0.45, 0.7], [0.8, 0.22], [1, 0]].forEach(function (st) {
        el('stop', { offset: st[0], style: 'stop-color:' + p.color + ';stop-opacity:' + st[1] }, g);
      });
    });

    el('text', { x: 18, y: 22, class: 'band-title' }).textContent = 'The world at time t';
    el('line', { x1: 18, x2: VBW - 18, y1: 182, y2: 182, class: 'axis faint' });
    el('text', { x: 18, y: 204, class: 'band-title' }).textContent = 'Opacity of each primitive over time';

    // space band: trajectories over each lifespan
    prims.forEach(function (p) {
      if (!p.v) return;
      var a = p.t - p.l / 2, b = p.t + p.l / 2;
      var xa = sx(p.x + p.v * (a - p.t)), xb = sx(p.x + p.v * (b - p.t));
      el('line', { x1: xa, x2: xb, y1: p.y, y2: p.y, class: 'trail', stroke: p.color });
      var dir = xb > xa ? 1 : -1;
      el('path', { d: 'M' + (xb - dir * 7) + ' ' + (p.y - 4) + 'L' + xb + ' ' + p.y + 'L' + (xb - dir * 7) + ' ' + (p.y + 4), class: 'trail-head', stroke: p.color });
    });
    var shapes = prims.map(function (p) {
      var g = el('g', {});
      var ghost = el('ellipse', { rx: p.rx, ry: p.ry, class: 'ghost', stroke: p.color }, g);
      var body = el('ellipse', { rx: p.rx * 1.25, ry: p.ry * 1.25, fill: 'url(#splat-' + p.id + ')' }, g);
      var dot = el('circle', { r: 2.2, fill: p.color }, g);
      var note = el('text', { class: 'note', 'text-anchor': 'middle' });
      note.textContent = p.note;
      return { g: g, ghost: ghost, body: body, dot: dot, note: note };
    });

    // time band: temporal opacity curves
    prims.forEach(function (p) {
      var d = '', a = 'M' + tx(0) + ' ' + BASE;
      for (var i = 0; i <= 180; i++) {
        var t = i / 180, y = BASE - AMP * opacity(p, t);
        d += (i ? 'L' : 'M') + tx(t).toFixed(1) + ' ' + y.toFixed(1);
        a += 'L' + tx(t).toFixed(1) + ' ' + y.toFixed(1);
      }
      a += 'L' + tx(1) + ' ' + BASE + 'Z';
      el('path', { d: a, class: 'area', fill: p.color });
      el('path', { d: d, class: 'curve', stroke: p.color });
      el('line', { x1: tx(p.t), x2: tx(p.t), y1: BASE, y2: BASE + 5, class: 'axis' });
      sym(tx(p.t) - 3, BASE + 20, 't', p.id);
    });
    el('line', { x1: X0, x2: X1, y1: BASE, y2: BASE, class: 'axis' });
    el('text', { x: X1 + 10, y: BASE + 4, class: 'axis-label' }).textContent = 'time';
    // duration of primitive 1: opacity falls to 5% at t1 +/- l1/2
    var p1 = prims[0], ya = BASE + 34, xa = tx(p1.t - p1.l / 2), xb = tx(p1.t + p1.l / 2);
    el('path', { d: 'M' + xa + ' ' + (ya - 5) + 'V' + ya + 'H' + xb + 'V' + (ya - 5), class: 'bracket' });
    sym(tx(p1.t) - 3, ya + 15, 'l', '1');

    var head = el('line', { y1: BASE - AMP - 16, y2: BASE, class: 'playhead' });
    var knob = el('circle', { r: 4.5, cy: BASE - AMP - 16, class: 'playhead-knob' });
    var tval = el('text', { y: BASE - AMP - 24, class: 'tval', 'text-anchor': 'middle' });

    var t = 0.30, auto = !reduceMotion, dragging = false, visible = false, last = 0;
    function render() {
      shapes.forEach(function (s, i) {
        var p = prims[i], dt = t - p.t, o = opacity(p, t);
        var cx = sx(p.x + p.v * dt), ang = p.rot + p.w * dt;
        s.g.setAttribute('transform', 'translate(' + cx.toFixed(1) + ' ' + p.y + ') rotate(' + ang.toFixed(1) + ')');
        s.body.setAttribute('opacity', o.toFixed(3));
        s.dot.setAttribute('opacity', o.toFixed(3));
        // outside its duration a primitive is invisible, so its outline and label go too
        s.ghost.setAttribute('opacity', o > 0.02 ? '0.3' : '0');
        s.note.setAttribute('x', (cx + p.noteDx).toFixed(1));
        s.note.setAttribute('y', p.y + p.noteDy);
        s.note.setAttribute('opacity', clamp(o * 1.6, 0, 1).toFixed(2));
      });
      var x = tx(t);
      head.setAttribute('x1', x); head.setAttribute('x2', x);
      knob.setAttribute('cx', x);
      tval.setAttribute('x', x);
      tval.textContent = 't = ' + t.toFixed(2);
    }
    function setFromEvent(e) {
      var r = svg.getBoundingClientRect(), vb = VBW / r.width;
      t = clamp(((e.clientX - r.left) * vb - X0) / (X1 - X0), 0, 1);
      render();
    }
    svg.addEventListener('pointerdown', function (e) {
      dragging = true; auto = false;
      svg.setPointerCapture(e.pointerId);
      setFromEvent(e);
    });
    svg.addEventListener('pointermove', function (e) { if (dragging) setFromEvent(e); });
    svg.addEventListener('pointerup', function () { dragging = false; });
    svg.addEventListener('pointercancel', function () { dragging = false; });
    render();
    if (!auto || !('IntersectionObserver' in window)) return;
    new IntersectionObserver(function (es) {
      visible = es[0].isIntersecting;
      if (visible) { last = 0; requestAnimationFrame(tick); }
    }, { threshold: 0.3 }).observe(svg);
    function tick(now) {
      if (!visible || !auto) return;
      if (!document.hidden) {
        if (last) t = (t + (now - last) / 8000) % 1;
        last = now;
        render();
      }
      requestAnimationFrame(tick);
    }
  })();

  /* ---------- Copy BibTeX ---------- */
  document.querySelectorAll('[data-copy]').forEach(function (b) {
    b.addEventListener('click', function () {
      var text = document.querySelector(b.dataset.copy).textContent;
      var done = function () {
        b.textContent = 'Copied';
        setTimeout(function () { b.textContent = 'Copy BibTeX'; }, 1600);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
      } else { fallbackCopy(text); done(); }
    });
  });
  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', '');
    ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }
})();
