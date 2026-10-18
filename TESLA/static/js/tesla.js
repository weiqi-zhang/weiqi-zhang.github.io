/* TESLA project page */
(function () {
  'use strict';

  var CLIPS = window.TESLA_CLIPS || [];
  var FEATURED = window.TESLA_FEATURED || [];
  var BY_ID = {};
  CLIPS.forEach(function (c) { BY_ID[c.id] = c; });
  var GALLERY = CLIPS.filter(function (c) { return !c.hidden; });
  var GALLERY_INDEX = {};
  GALLERY.forEach(function (c, i) { GALLERY_INDEX[c.id] = i; });

  // All clips play at 12 fps. Front views hold 24 frames; free viewpoints hold 96 frames,
  // in which the 24-frame animation repeats while the camera moves.
  var FPS = 12;
  function frameCount(c) { return c.view === 'free' ? 96 : 24; }
  function majorEvery(c) { return c.view === 'free' ? 24 : 6; }

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  function clipFile(id, file) { return 'static/clips/' + id + '/' + file; }
  function inputFile(id) { return 'static/inputs/' + id + '.webp'; }
  function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }
  function pad2(n) { return (n < 10 ? '0' : '') + n; }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }
  // A play() request is aborted harmlessly when the source changes; only a refusal to autoplay
  // (NotAllowedError, e.g. in a power-saving mode) means the visitor has to press play.
  function tryPlay(v, onRefused) {
    var p = v.play();
    if (p && p.catch) p.catch(function (err) {
      if (onRefused && err && err.name === 'NotAllowedError') onRefused();
    });
  }
  function cssVar(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

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
  var sections = Array.prototype.slice.call(document.querySelectorAll('main > section[id], main > header[id]'));
  function updateTopbar() {
    if (topbar) topbar.classList.toggle('is-solid', window.scrollY > 24);
    var current = null;
    sections.forEach(function (s) {
      if (s.getBoundingClientRect().top <= 120) current = s.id;
    });
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) current = 'bibtex';
    navLinks.forEach(function (a) {
      if (a.getAttribute('href') === '#' + current) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    });
  }
  window.addEventListener('scroll', updateTopbar, { passive: true });
  updateTopbar();

  /* ---------- Clip viewer: one result with play/pause and a frame scrubber ---------- */
  function ClipViewer(root, opts) {
    var self = this;
    opts = opts || {};
    this.root = root;
    root.appendChild(document.getElementById('viewer-template').content.cloneNode(true));
    this.stage = root.querySelector('.viewer-stage');
    this.video = root.querySelector('.viewer-video');
    this.playBtn = root.querySelector('.play');
    this.timeline = root.querySelector('.timeline');
    this.range = root.querySelector('.time');
    this.ticksDone = root.querySelector('.ticks-done');
    this.readout = root.querySelector('.readout');
    this.stage.setAttribute('role', 'img');
    this.wantPlay = !reduceMotion && opts.autoplay !== false;
    this.visible = true;
    this.suspended = false;
    this.ready = false;
    this.scrubbing = false;
    this.n = 24;
    this.raf = 0;

    var v = this.video;
    v.muted = true; v.loop = true; v.playsInline = true;
    v.addEventListener('play', function () { self.loop(); self.updatePlayUI(); });
    v.addEventListener('pause', function () {
      cancelAnimationFrame(self.raf);
      if (!self.scrubbing) self.setFrameUI(self.frame());
    });

    this.playBtn.addEventListener('click', function () {
      if (self.wantPlay) self.pause(); else self.play();
    });

    // scrubbing: playback pauses while the marker is dragged and resumes afterwards
    var resume = false;
    this.range.addEventListener('pointerdown', function () {
      self.scrubbing = true;
      resume = self.wantPlay;
      self.video.pause();
    });
    this.range.addEventListener('input', function () {
      var f = +self.range.value;
      if (!self.scrubbing && self.wantPlay) self.pause();   // arrow keys: stop on the chosen frame
      self.setFrameUI(f);
      if (self.ready) self.seekFrame(f);
    });
    var endScrub = function () {
      if (!self.scrubbing) return;
      self.scrubbing = false;
      if (resume) self.syncPlayback();
    };
    this.range.addEventListener('change', endScrub);
    this.range.addEventListener('pointerup', endScrub);
    this.range.addEventListener('pointercancel', endScrub);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        self.visible = entries[0].isIntersecting;
        self.syncPlayback();
      }, { threshold: 0.25 }).observe(this.stage);
    }
    document.addEventListener('visibilitychange', function () { self.syncPlayback(); });
    this.updatePlayUI();
  }

  ClipViewer.prototype.load = function (c) {
    var self = this, token = {};
    this.token = token;
    this.clip = c;
    this.ready = false;
    this.n = frameCount(c);
    this.range.max = String(this.n - 1);
    this.timeline.style.setProperty('--n', String(this.n - 1));
    this.timeline.style.setProperty('--maj', String(majorEvery(c)));
    this.stage.setAttribute('aria-label', 'Generated result' + (c.prompt ? ': ' + c.prompt : ''));
    this.stage.style.setProperty('--ar', c.w + ' / ' + c.h);
    this.stage.style.setProperty('--h', String(c.h));
    this.stage.style.setProperty('--w', String(c.w));
    this.video.poster = clipFile(c.id, 'poster.webp');
    this.video.src = clipFile(c.id, 'clip.mp4');
    this.video.load();
    this.video.addEventListener('loadeddata', function onData() {
      self.video.removeEventListener('loadeddata', onData);
      if (self.token !== token) return;
      self.ready = true;
      self.syncPlayback();
    });
    this.setFrameUI(0);
  };
  ClipViewer.prototype.frame = function () {
    return clamp(Math.floor(this.video.currentTime * FPS + 1e-3), 0, this.n - 1);
  };
  ClipViewer.prototype.seekFrame = function (f) {
    this.video.currentTime = (f + 0.5) / FPS;
  };
  ClipViewer.prototype.play = function () {
    this.wantPlay = true;
    this.syncPlayback();
    this.updatePlayUI();
  };
  ClipViewer.prototype.pause = function () {
    this.wantPlay = false;
    this.video.pause();
    this.updatePlayUI();
  };
  ClipViewer.prototype.suspend = function (on) {
    this.suspended = on;
    this.syncPlayback();
  };
  ClipViewer.prototype.syncPlayback = function () {
    var self = this;
    var should = this.wantPlay && this.ready && this.visible && !this.suspended && !this.scrubbing && !document.hidden;
    if (should && this.video.paused) {
      tryPlay(this.video, function () { self.wantPlay = false; self.updatePlayUI(); });
    } else if (!should && !this.video.paused) {
      this.video.pause();
    }
  };
  ClipViewer.prototype.loop = function () {
    var self = this;
    cancelAnimationFrame(this.raf);
    var step = function () {
      self.setFrameUI(self.frame());
      if (!self.video.paused) self.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  };
  ClipViewer.prototype.updatePlayUI = function () {
    var paused = !this.wantPlay;
    this.root.classList.toggle('is-paused', paused);
    this.playBtn.setAttribute('aria-label', paused ? 'Play' : 'Pause');
  };
  ClipViewer.prototype.setFrameUI = function (f) {
    this.range.value = f;
    this.ticksDone.style.setProperty('--p', (f / (this.n - 1) * 100).toFixed(3) + '%');
    this.readout.textContent = pad2(f + 1) + ' / ' + this.n;
    this.range.setAttribute('aria-valuetext', 'Frame ' + (f + 1) + ' of ' + this.n);
  };

  /* ---------- Caption: reference image and prompt, when they are known ---------- */
  function renderCaption(el, c) {
    if (!el) return;
    if (!c.input && !c.prompt) { el.hidden = true; el.innerHTML = ''; return; }
    el.hidden = false;
    var label = c.input && c.prompt ? 'Reference image and prompt' : (c.input ? 'Reference image' : 'Prompt');
    el.innerHTML =
      '<span class="cond">' + label + '</span>' +
      '<span class="cond-body">' +
      (c.input ? '<img class="cond-img" src="' + inputFile(c.id) + '" width="256" height="256" alt="Reference image">' : '') +
      (c.prompt ? '<q class="prompt">' + esc(c.prompt) + '</q>' : '') +
      '</span>';
  }

  /* ---------- Main viewer with a picker ---------- */
  var mainViewer = null;
  var mainRoot = document.getElementById('main-viewer');
  if (mainRoot && FEATURED.length) {
    mainViewer = new ClipViewer(mainRoot, { autoplay: true });
    var mainCaption = document.getElementById('main-caption');
    var picker = document.getElementById('main-picker');
    var selectMain = function (id) {
      var c = BY_ID[id];
      if (!c) return;
      mainViewer.load(c);
      renderCaption(mainCaption, c);
      if (picker) picker.querySelectorAll('button').forEach(function (b) {
        b.setAttribute('aria-pressed', String(b.dataset.id === id));
      });
    };
    if (picker) FEATURED.forEach(function (id, i) {
      var c = BY_ID[id];
      if (!c) return;
      var b = document.createElement('button');
      b.type = 'button';
      b.dataset.id = id;
      b.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-label', 'Show result ' + (i + 1) + (c.prompt ? ': ' + c.prompt : ''));
      b.innerHTML = '<img src="' + clipFile(id, 'poster.webp') + '" width="' + c.w + '" height="' + c.h + '" alt="">';
      b.addEventListener('click', function () { selectMain(id); });
      picker.appendChild(b);
    });
    selectMain(FEATURED[0]);
  }

  /* ---------- Result walls ---------- */
  // Each wall is a grid of square cells. A clip loads only when its cell first comes into view,
  // and plays only while it is on screen (or under the pointer, when reduced motion is requested).
  var wallItems = [];
  var wallPaused = false;   // true while the viewer dialog is open
  var wallObserver = 'IntersectionObserver' in window ? new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var item = e.target._item;
      item.visible = e.isIntersecting;
      syncItem(item);
    });
  }, { threshold: 0.2 }) : null;

  function syncItem(item) {
    var on = reduceMotion ? item.hover : (item.visible || item.hover);
    var should = on && !wallPaused && !document.hidden;
    var v = item.video;
    if (should) {
      if (!v.getAttribute('src')) v.src = clipFile(item.clip.id, 'clip.mp4');
      if (v.paused) tryPlay(v);
    } else if (!v.paused) {
      v.pause();
    }
  }
  function syncWalls() { wallItems.forEach(syncItem); }
  document.addEventListener('visibilitychange', syncWalls);

  // <ul class="wall" data-view="front|free">
  document.querySelectorAll('ul.wall[data-view]').forEach(function (ul) {
    var view = ul.getAttribute('data-view');
    GALLERY.forEach(function (c) {
      if (c.view !== view) return;
      var no = GALLERY_INDEX[c.id] + 1;
      var li = document.createElement('li');
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'obj';
      btn.setAttribute('aria-label', 'Open result ' + no + ' of ' + GALLERY.length + (c.prompt ? ': ' + c.prompt : ''));
      var v = document.createElement('video');
      v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'none';
      v.setAttribute('muted', ''); v.setAttribute('playsinline', ''); v.setAttribute('aria-hidden', 'true');
      v.width = c.w; v.height = c.h;
      v.poster = clipFile(c.id, 'poster.webp');
      btn.appendChild(v);
      if (c.prompt) {
        var s = document.createElement('span');
        s.className = 'obj-prompt';
        s.setAttribute('aria-hidden', 'true');
        s.textContent = c.prompt;
        btn.appendChild(s);
      }
      li.appendChild(btn);
      ul.appendChild(li);
      var item = { clip: c, video: v, visible: !wallObserver, hover: false };
      li._item = item;
      wallItems.push(item);
      if (wallObserver) wallObserver.observe(li);
      btn.addEventListener('pointerenter', function () {
        if (!finePointer) return;
        item.hover = true;
        syncItem(item);
      });
      btn.addEventListener('pointerleave', function () {
        item.hover = false;
        syncItem(item);
      });
      btn.addEventListener('click', function () { openDialog(c.id); });
    });
  });
  syncWalls();

  /* ---------- Dialog viewer ---------- */
  var dlg = document.getElementById('clip-dialog');
  var dlgViewer = null, dlgIndex = 0;
  var dlgCaption = document.getElementById('dialog-caption');
  var dlgCount = document.getElementById('dialog-count');

  function showDialogClip() {
    var c = GALLERY[dlgIndex];
    dlgViewer.load(c);
    renderCaption(dlgCaption, c);
    dlgCount.textContent = pad2(dlgIndex + 1) + ' / ' + GALLERY.length;
  }
  function stepDialog(d) {
    dlgIndex = (dlgIndex + d + GALLERY.length) % GALLERY.length;
    showDialogClip();
  }
  function openDialog(id) {
    if (!dlg || typeof dlg.showModal !== 'function' || !(id in GALLERY_INDEX)) return;
    dlgIndex = GALLERY_INDEX[id];
    if (!dlgViewer) dlgViewer = new ClipViewer(document.getElementById('dialog-viewer'), { autoplay: true });
    if (mainViewer) mainViewer.suspend(true);
    wallPaused = true;
    syncWalls();
    dlg.showModal();
    dlg.querySelector('.dialog-close').focus();
    dlgViewer.wantPlay = !reduceMotion;
    dlgViewer.updatePlayUI();
    dlgViewer.suspend(false);
    showDialogClip();
  }
  if (dlg) {
    dlg.querySelectorAll('[data-step]').forEach(function (b) {
      b.addEventListener('click', function () { stepDialog(Number(b.dataset.step)); });
    });
    dlg.querySelector('.dialog-close').addEventListener('click', function () { dlg.close(); });
    dlg.addEventListener('click', function (e) { if (e.target === dlg) dlg.close(); });
    dlg.addEventListener('keydown', function (e) {
      var t = e.target;
      if (t && t.classList && t.classList.contains('time')) return;   // arrows move the frame marker there
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        stepDialog(e.key === 'ArrowRight' ? 1 : -1);
      }
    });
    dlg.addEventListener('close', function () {
      if (dlgViewer) dlgViewer.suspend(true);
      if (mainViewer) mainViewer.suspend(false);
      wallPaused = false;
      syncWalls();
    });
  }

  /* ---------- Voxel wordmark (style C) ---------- */
  // "TESLA" on a 5 x 7 grid per letter. It assembles once: the occupied cells appear in gray first
  // (the sparse structure), then colour sweeps across them from left to right (the appearance).
  (function initVoxelMark() {
    var svg = document.querySelector('svg.voxel-mark');
    if (!svg) return;
    var NS = 'http://www.w3.org/2000/svg';
    var GLYPHS = {
      T: ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
      E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
      S: ['.####', '#....', '#....', '.###.', '....#', '....#', '####.'],
      L: ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
      A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#']
    };
    var word = 'TESLA', P = 10, G = 1.6;      // pitch and gap in viewBox units
    var cols = word.length * 6 - 1;
    svg.setAttribute('viewBox', '0 0 ' + (cols * P - G) + ' ' + (7 * P - G));

    // The hero grid has the same pitch as the voxels; shift it so every voxel sits in a grid cell.
    var hero = svg.closest('.hero');
    var alignGrid = function () {
      if (!hero) return;
      var pitch = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--pitch')) || 24;
      var hb = hero.getBoundingClientRect(), sb = svg.getBoundingClientRect();
      var gx = ((sb.left - hb.left) % pitch + pitch) % pitch, gy = ((sb.top - hb.top) % pitch + pitch) % pitch;
      hero.style.setProperty('--gx', gx.toFixed(2) + 'px');
      hero.style.setProperty('--gy', gy.toFixed(2) + 'px');
    };
    alignGrid();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(alignGrid);
    if ('ResizeObserver' in window && hero) new ResizeObserver(alignGrid).observe(hero);
    else window.addEventListener('resize', alignGrid);
    var from = cssVar(svg, '--vox-a', '#3361D8'), to = cssVar(svg, '--vox-b', '#8152D6');
    var gray = cssVar(svg, '--vox-gray', '#BFC4CC');
    var defs = document.createElementNS(NS, 'defs');
    var grad = document.createElementNS(NS, 'linearGradient');
    grad.setAttribute('id', 'vox-grad');
    grad.setAttribute('gradientUnits', 'userSpaceOnUse');
    grad.setAttribute('x1', '0'); grad.setAttribute('x2', String(cols * P)); grad.setAttribute('y1', '0'); grad.setAttribute('y2', '0');
    [[0, from], [1, to]].forEach(function (s) {
      var st = document.createElementNS(NS, 'stop');
      st.setAttribute('offset', s[0]);
      st.setAttribute('stop-color', s[1]);
      grad.appendChild(st);
    });
    defs.appendChild(grad);
    svg.appendChild(defs);
    var cells = [];
    word.split('').forEach(function (ch, li) {
      GLYPHS[ch].forEach(function (row, r) {
        row.split('').forEach(function (bit, c) {
          if (bit !== '#') return;
          var x = (li * 6 + c) * P, y = r * P;
          var g = document.createElementNS(NS, 'g');
          var base = document.createElementNS(NS, 'rect');
          var top = document.createElementNS(NS, 'rect');
          [base, top].forEach(function (rc) {
            rc.setAttribute('x', x); rc.setAttribute('y', y);
            rc.setAttribute('width', P - G); rc.setAttribute('height', P - G);
            rc.setAttribute('rx', '0.6');
          });
          base.setAttribute('fill', gray);
          top.setAttribute('fill', 'url(#vox-grad)');
          g.appendChild(base);
          g.appendChild(top);
          svg.appendChild(g);
          cells.push({ g: g, base: base, top: top, x: x });
        });
      });
    });
    if (reduceMotion) return;   // final state: coloured voxels
    var maxX = cols * P;
    cells.forEach(function (cell) {
      cell.g.style.opacity = '0';
      cell.top.style.opacity = '0';
      cell.delayA = Math.random() * 520;                      // structure: scattered
      cell.delayB = 760 + (cell.x / maxX) * 760;               // appearance: left to right
    });
    var t0 = 0;
    function frame(now) {
      if (!t0) t0 = now;
      var t = now - t0, done = true;
      cells.forEach(function (cell) {
        var a = clamp((t - cell.delayA) / 180, 0, 1);
        var b = clamp((t - cell.delayB) / 260, 0, 1);
        cell.g.style.opacity = a.toFixed(3);
        cell.top.style.opacity = b.toFixed(3);
        if (b < 1) done = false;
      });
      if (!done) requestAnimationFrame(frame);
    }
    var start = function () { requestAnimationFrame(frame); };
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (es) {
        if (es[0].isIntersecting) { io.disconnect(); setTimeout(start, 250); }
      });
      io.observe(svg);
    } else start();
  })();

  /* ---------- 4D Gaussian illustration ---------- */
  // One spatial axis x against time t. A Gaussian with temporal centre mt, temporal std st,
  // velocity v (the tilt, Sigma_xt / Sigma_tt) and conditional spatial std sc has
  //   Sigma_tt = st^2, Sigma_xt = v st^2, Sigma_xx = sc^2 + v^2 st^2.
  // Sliced at time t it gives the conditional Gaussian with mean mx + v (t - mt) and std sc,
  // weighted by the marginal exp(-(t - mt)^2 / (2 st^2)), as in the 4DGS rendering the paper uses.
  (function initExplainer() {
    var svg = document.querySelector('.explainer-svg');
    if (!svg) return;
    var NS = 'http://www.w3.org/2000/svg';
    var GS = [
      { key: 'a', mt: 0.24, mx: 0.34, st: 0.085, v: 1.3, sc: 0.045, color: cssVar(svg, '--g1', '#1F6E85') },
      { key: 'b', mt: 0.52, mx: 0.80, st: 0.30, v: 0, sc: 0.042, color: cssVar(svg, '--g2', '#7A86A3') },
      { key: 'c', mt: 0.77, mx: 0.50, st: 0.075, v: -1.7, sc: 0.04, color: cssVar(svg, '--g3', '#6B5CA5') }
    ];
    var peak = parseFloat(cssVar(svg, '--g-peak', '0.6')) || 0.6;
    var desc = svg.querySelector('desc');
    var t = 0.30, auto = !reduceMotion, dragging = false, visible = false, last = 0;
    var L = null;        // current layout
    var parts = [];      // per-Gaussian elements that move with t
    var head, knob, tval;

    svg.setAttribute('tabindex', '0');
    svg.setAttribute('role', 'slider');
    svg.setAttribute('aria-label', 'Time t in the illustration');
    svg.setAttribute('aria-valuemin', '0');
    svg.setAttribute('aria-valuemax', '1');
    svg.removeAttribute('aria-labelledby');
    if (desc) svg.setAttribute('aria-describedby', desc.id);

    function el(tag, attrs, parent) {
      var n = document.createElementNS(NS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(n);
      return n;
    }
    function weight(g, tt) { var d = (tt - g.mt) / g.st; return Math.exp(-0.5 * d * d); }
    function mean(g, tt) { return g.mx + g.v * (tt - g.mt); }

    function build() {
      var W = Math.max(320, Math.round(svg.parentNode.clientWidth || 960));
      if (L && L.W === W) return;
      var narrow = W < 600;
      var H = narrow ? 316 : 346;
      var stripW = narrow ? 92 : 210;
      L = {
        W: W, H: H,
        X0: narrow ? 44 : 64,
        X1: W - stripW - (narrow ? 22 : 40),
        Y0: 76, Y1: H - 40,
        SX: W - stripW / 2 - (narrow ? 2 : 14)
      };
      L.Wt = L.X1 - L.X0;
      L.Hx = L.Y1 - L.Y0;
      while (svg.lastChild && svg.lastChild !== desc) svg.removeChild(svg.lastChild);
      if (desc && desc.parentNode !== svg) svg.appendChild(desc);
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

      var defs = el('defs', {});
      GS.forEach(function (g) {
        var grad = el('radialGradient', { id: 'gauss-' + g.key }, defs);
        // a Gaussian profile out to 2 sigma
        [[0, 1], [0.25, 0.88], [0.5, 0.61], [0.75, 0.32], [1, 0]].forEach(function (s) {
          el('stop', { offset: s[0], 'stop-color': g.color, 'stop-opacity': (peak * s[1]).toFixed(3) }, grad);
        });
      });
      var clip = el('clipPath', { id: 'plane-clip' }, defs);
      el('rect', { x: L.X0, y: L.Y0 - 6, width: L.Wt, height: L.Hx + 6 }, clip);

      el('text', { x: narrow ? 12 : L.X0, y: 24, class: 'band-title' }).textContent = narrow ? 'Space and time' : 'Gaussians over space and time';
      el('text', { x: L.SX, y: 24, class: 'band-title', 'text-anchor': 'middle' }).textContent = narrow ? 'At time t' : 'Rendered at time t';

      el('line', { x1: L.X0, x2: L.X1, y1: L.Y1, y2: L.Y1, class: 'axis' });
      el('line', { x1: L.X0, x2: L.X0, y1: L.Y0, y2: L.Y1, class: 'axis' });
      el('text', { x: L.X1, y: L.Y1 + 20, class: 'axis-label', 'text-anchor': 'end' }).textContent = 'time';
      el('text', { x: L.X0 - 8, y: L.Y0 + 4, class: 'axis-label', 'text-anchor': 'end' }).textContent = 'space';
      el('line', { x1: L.SX, x2: L.SX, y1: L.Y0, y2: L.Y1, class: 'axis-faint' });

      // the Gaussians in the space-time plane, drawn to 2 sigma
      var plane = el('g', { 'clip-path': 'url(#plane-clip)' });
      GS.forEach(function (g) {
        var a = Math.pow(L.Wt * g.st, 2);
        var b = -L.Wt * L.Hx * g.v * g.st * g.st;
        var c = L.Hx * L.Hx * (g.sc * g.sc + g.v * g.v * g.st * g.st);
        var mid = (a + c) / 2, rad = Math.sqrt((a - c) * (a - c) / 4 + b * b);
        var l1 = mid + rad, l2 = Math.max(mid - rad, 1e-6);
        var ang = 0.5 * Math.atan2(2 * b, a - c) * 180 / Math.PI;
        var cx = L.X0 + g.mt * L.Wt, cy = L.Y1 - g.mx * L.Hx;
        var tr = 'translate(' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ') rotate(' + ang.toFixed(2) + ')';
        el('ellipse', { rx: (2 * Math.sqrt(l1)).toFixed(1), ry: (2 * Math.sqrt(l2)).toFixed(1), transform: tr, fill: 'url(#gauss-' + g.key + ')' }, plane);
        el('ellipse', { rx: (2 * Math.sqrt(l1)).toFixed(1), ry: (2 * Math.sqrt(l2)).toFixed(1), transform: tr, class: 'outline', stroke: g.color }, plane);
      });

      // elements that follow t
      parts = GS.map(function (g) {
        var r = Math.max(7, 2 * g.sc * L.Hx);
        return {
          leader: el('line', { class: 'leader', stroke: g.color }),
          slice: el('line', { class: 'slice', stroke: g.color }),
          dot: el('circle', { r: 2.6, fill: g.color }),
          blob: el('ellipse', { rx: r * 1.15, ry: r, fill: 'url(#gauss-' + g.key + ')' }),
          core: el('circle', { r: 2.2, fill: g.color }),
          r: r
        };
      });
      head = el('line', { y1: L.Y0 - 12, y2: L.Y1, class: 'playhead' });
      knob = el('circle', { r: 4.5, cy: L.Y0 - 12, class: 'playhead-knob' });
      tval = el('text', { y: L.Y0 - 23, class: 'tval', 'text-anchor': 'middle' });
      render();
    }

    function render() {
      if (!L) return;
      var x = L.X0 + t * L.Wt;
      GS.forEach(function (g, i) {
        var p = parts[i], w = weight(g, t), m = mean(g, t);
        var y = L.Y1 - m * L.Hx, half = 2 * g.sc * L.Hx;
        var shown = w > 0.01 && m > -0.05 && m < 1.05;
        var o = shown ? w : 0;
        p.slice.setAttribute('x1', x); p.slice.setAttribute('x2', x);
        p.slice.setAttribute('y1', (y - half).toFixed(1)); p.slice.setAttribute('y2', (y + half).toFixed(1));
        p.slice.setAttribute('opacity', (0.9 * o).toFixed(3));
        p.dot.setAttribute('cx', x); p.dot.setAttribute('cy', y.toFixed(1));
        p.dot.setAttribute('opacity', o.toFixed(3));
        p.leader.setAttribute('x1', (x + 6).toFixed(1)); p.leader.setAttribute('x2', (L.SX - p.r * 1.15 - 6).toFixed(1));
        p.leader.setAttribute('y1', y.toFixed(1)); p.leader.setAttribute('y2', y.toFixed(1));
        p.leader.setAttribute('opacity', (0.55 * o).toFixed(3));
        p.blob.setAttribute('cx', L.SX); p.blob.setAttribute('cy', y.toFixed(1));
        p.blob.setAttribute('opacity', o.toFixed(3));
        p.core.setAttribute('cx', L.SX); p.core.setAttribute('cy', y.toFixed(1));
        p.core.setAttribute('opacity', o.toFixed(3));
      });
      head.setAttribute('x1', x); head.setAttribute('x2', x);
      knob.setAttribute('cx', x);
      tval.setAttribute('x', clamp(x, L.X0 + 22, L.X1 - 22));
      tval.textContent = 't = ' + t.toFixed(2);
      svg.setAttribute('aria-valuenow', t.toFixed(2));
      svg.setAttribute('aria-valuetext', 't = ' + t.toFixed(2));
    }

    function setFromEvent(e) {
      var r = svg.getBoundingClientRect(), s = L.W / r.width;
      t = clamp(((e.clientX - r.left) * s - L.X0) / L.Wt, 0, 1);
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
    svg.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 0.1 : 0.02;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') t -= step;
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') t += step;
      else if (e.key === 'Home') t = 0;
      else if (e.key === 'End') t = 1;
      else return;
      e.preventDefault();
      auto = false;
      t = clamp(t, 0, 1);
      render();
    });

    build();
    var pending = 0;
    var onResize = function () {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(build);
    };
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(svg.parentNode);
    else window.addEventListener('resize', onResize);

    if (!auto || !('IntersectionObserver' in window)) return;
    new IntersectionObserver(function (es) {
      visible = es[0].isIntersecting;
      if (visible) { last = 0; requestAnimationFrame(tick); }
    }, { threshold: 0.3 }).observe(svg);
    function tick(now) {
      if (!visible || !auto) return;
      if (!document.hidden) {
        if (last) t = (t + (now - last) / 9000) % 1;
        last = now;
        render();
      }
      requestAnimationFrame(tick);
    }
  })();

  /* ---------- Copy BibTeX ---------- */
  document.querySelectorAll('[data-copy]').forEach(function (b) {
    var label = b.textContent;
    b.addEventListener('click', function () {
      var text = document.querySelector(b.dataset.copy).textContent;
      var done = function () {
        b.textContent = 'Copied';
        setTimeout(function () { b.textContent = label; }, 1600);
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
