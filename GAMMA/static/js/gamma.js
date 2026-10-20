/* GAMMA project page: video playback, time indicators, lifespan illustration, BibTeX copy. */
(() => {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');

  /* ---------- top bar ---------- */
  const bar = document.querySelector('.topbar');
  const onScroll = () => { if (bar) bar.classList.toggle('is-scrolled', window.scrollY > 8); };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  const navLinks = [...document.querySelectorAll('.topbar-links a')];
  const sections = navLinks.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  if ('IntersectionObserver' in window && sections.length) {
    const spy = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (!e.isIntersecting) return;
        navLinks.forEach(a => a.setAttribute('aria-current', a.getAttribute('href') === '#' + e.target.id ? 'true' : 'false'));
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    sections.forEach(s => spy.observe(s));
  }

  /* ---------- videos: play only what is on screen; click to pause ---------- */
  const holders = [...document.querySelectorAll('[data-video]')];
  const start = v => {
    if (v.preload === 'none') v.preload = 'auto';
    const p = v.play();
    if (p && p.catch) p.catch(() => {});
  };
  holders.forEach(h => {
    const v = h.querySelector('video');
    if (!v) return;
    h._video = v;
    h._userPaused = reduce.matches;
    if (reduce.matches) h.classList.add('is-paused');
    const stage = h.querySelector('.clip-stage, .frame-gate') || h;
    stage.addEventListener('click', () => {
      if (v.paused) { h._userPaused = false; h.classList.remove('is-paused'); start(v); }
      else { h._userPaused = true; h.classList.add('is-paused'); v.pause(); }
    });
    stage.setAttribute('role', 'button');
    stage.setAttribute('tabindex', '0');
    stage.setAttribute('aria-label', 'Play or pause this video');
    stage.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); stage.click(); }
    });
  });
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => {
        const h = e.target, v = h._video;
        if (!v) return;
        h._visible = e.isIntersecting;
        if (e.isIntersecting && !h._userPaused) start(v);
        else if (!e.isIntersecting && !v.paused) v.pause();
      });
    }, { threshold: 0.2 });
    holders.forEach(h => io.observe(h));
  } else {
    holders.forEach(h => { if (h._video && !h._userPaused) start(h._video); });
  }

  /* ---------- time indicators (one tick per rendered frame, 10 frames per second) ---------- */
  const timed = [...document.querySelectorAll('[data-time]')].filter(el => el._video || el.querySelector('video'));
  timed.forEach(el => {
    el._video = el._video || el.querySelector('video');
    el._ticks = [...el.querySelectorAll('.ticks i')];
    el._fps = parseFloat(el.dataset.fps || '10');
    el._playEnd = parseFloat(el.dataset.playEnd || 'Infinity');
    el._idx = -2;
  });
  const paint = el => {
    const v = el._video, n = el._ticks.length, kind = el.dataset.time;
    const t = v.currentTime || 0;
    let idx, held;
    if (kind === 'held') { held = true; idx = -1; }
    else if (kind === 'hold' && t >= el._playEnd) { held = true; idx = n - 1; }
    else { held = false; idx = Math.min(n - 1, Math.floor(t * el._fps + 1e-3)); }
    el.classList.toggle('is-held', held);
    if (idx !== el._idx) {
      el._ticks.forEach((tk, i) => { tk.classList.toggle('is-on', i <= idx); tk.classList.toggle('is-now', i === idx); });
      el._idx = idx;
    }
  };
  let rafOn = false;
  const loop = () => {
    let any = false;
    timed.forEach(el => { if (!el._video.paused) { paint(el); any = true; } });
    if (any) requestAnimationFrame(loop); else rafOn = false;
  };
  timed.forEach(el => {
    const v = el._video;
    v.addEventListener('play', () => { if (!rafOn) { rafOn = true; requestAnimationFrame(loop); } });
    ['seeked', 'loadeddata', 'pause'].forEach(ev => v.addEventListener(ev, () => paint(el)));
    paint(el);
  });

  /* ---------- BibTeX copy ---------- */
  document.querySelectorAll('[data-copy]').forEach(btn => {
    const label = btn.textContent;
    btn.addEventListener('click', async () => {
      const src = document.querySelector(btn.dataset.copy);
      if (!src) return;
      let ok = false;
      try { await navigator.clipboard.writeText(src.textContent); ok = true; }
      catch (e) {
        const r = document.createRange(), s = window.getSelection();
        r.selectNodeContents(src); s.removeAllRanges(); s.addRange(r);
        try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
        s.removeAllRanges();
      }
      btn.textContent = ok ? 'Copied' : 'Select and copy';
      setTimeout(() => { btn.textContent = label; }, 1600);
    });
  });

  /* ---------- lifespan illustration (Sec. 3.3) ----------
     Each bar is one Gaussian over normalised time. Opacity follows the temporal Gaussian of the paper:
     o(t) = exp(-(t - t_i)^2 / (2 sigma^2)), sigma = (l_i / 2) / sqrt(-2 ln eps), eps = 0.05,
     so a bar fades to 5% at t_i +- l_i / 2. Values are illustrative, not model output. */
  const SVGNS = 'http://www.w3.org/2000/svg';
  document.querySelectorAll('.lifespan').forEach(root => {
    const svg = root.querySelector('.ls-svg');
    const note = root.querySelector('.ls-note');
    const buttons = [...root.querySelectorAll('.ls-switch button')];
    const W = 920, ROW = 15, TOP = 6, BAR = 9;
    // [timestep, lifespan in stage 2]; sorted by lifespan; the last two are the 10% shortest
    const G = [[.50, .92], [.46, .86], [.55, .80], [.40, .74], [.62, .70], [.33, .64], [.70, .60], [.28, .55], [.58, .52], [.76, .48],
               [.24, .45], [.48, .42], [.79, .40], [.36, .37], [.66, .35], [.18, .33], [.52, .31], [.84, .30], [.57, .10], [.61, .07]];
    const FREE = 2;
    const target = { 1: G.map(([c, l], i) => i >= G.length - FREE ? .30 - (i - G.length + FREE) * .02 : Math.min(1, l + .06)),
                     2: G.map(([, l]) => l) };
    const H = TOP + G.length * ROW + 30;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const eps = 0.05;
    const defs = document.createElementNS(SVGNS, 'defs');
    ['reg', 'free'].forEach(kind => {
      const g = document.createElementNS(SVGNS, 'linearGradient');
      g.id = `ls-${kind}-${Math.random().toString(36).slice(2, 7)}`;
      for (let k = 0; k <= 20; k++) {
        const u = k / 10 - 1;                 // -1 .. 1 across the lifespan
        const s = document.createElementNS(SVGNS, 'stop');
        s.setAttribute('offset', (k / 20 * 100).toFixed(1) + '%');
        s.setAttribute('stop-color', `var(--ls-${kind})`);
        s.style.stopColor = `var(--ls-${kind})`;
        s.setAttribute('stop-opacity', Math.pow(eps, u * u).toFixed(3));
        g.appendChild(s);
      }
      defs.appendChild(g);
      root['_' + kind] = g.id;
    });
    const clip = document.createElementNS(SVGNS, 'clipPath');
    clip.id = 'ls-clip-' + Math.random().toString(36).slice(2, 7);
    const cr = document.createElementNS(SVGNS, 'rect');
    cr.setAttribute('x', 0); cr.setAttribute('y', 0); cr.setAttribute('width', W); cr.setAttribute('height', H);
    clip.appendChild(cr); defs.appendChild(clip);
    svg.appendChild(defs);
    const plot = document.createElementNS(SVGNS, 'g');
    plot.setAttribute('clip-path', `url(#${clip.id})`);
    svg.appendChild(plot);
    const rows = G.map(([c], i) => {
      const y = TOP + i * ROW;
      const g = document.createElementNS(SVGNS, 'g');
      const mk = kind => {
        const r = document.createElementNS(SVGNS, 'rect');
        r.setAttribute('y', y); r.setAttribute('height', BAR); r.setAttribute('rx', 2);
        r.setAttribute('fill', `url(#${root['_' + kind]})`);
        r.style.transition = 'opacity .3s';
        g.appendChild(r);
        return r;
      };
      const reg = mk('reg'), free = mk('free');
      const tick = document.createElementNS(SVGNS, 'rect');
      tick.setAttribute('x', c * W - 1); tick.setAttribute('y', y - 2); tick.setAttribute('width', 2); tick.setAttribute('height', BAR + 4);
      tick.setAttribute('fill', 'var(--ls-reg)'); tick.style.fill = 'var(--ls-reg)';
      g.appendChild(tick);
      plot.appendChild(g);
      return { c, reg, free, tick };
    });
    const split = document.createElementNS(SVGNS, 'line');
    const ySplit = TOP + (G.length - FREE) * ROW - 3;
    split.setAttribute('x1', 0); split.setAttribute('x2', W); split.setAttribute('y1', ySplit); split.setAttribute('y2', ySplit);
    split.setAttribute('stroke', 'var(--ls-axis)'); split.style.stroke = 'var(--ls-axis)';
    split.setAttribute('stroke-dasharray', '4 4'); split.style.transition = 'opacity .3s';
    svg.appendChild(split);
    const axisY = TOP + G.length * ROW + 6;
    const axis = document.createElementNS(SVGNS, 'line');
    axis.setAttribute('x1', 0); axis.setAttribute('x2', W); axis.setAttribute('y1', axisY); axis.setAttribute('y2', axisY);
    axis.setAttribute('stroke', 'var(--ls-axis)'); axis.style.stroke = 'var(--ls-axis)'; axis.setAttribute('stroke-width', 1.5);
    svg.appendChild(axis);
    const label = document.createElementNS(SVGNS, 'text');
    label.setAttribute('x', W); label.setAttribute('y', axisY + 19); label.setAttribute('text-anchor', 'end');
    label.setAttribute('font-size', 14); label.setAttribute('fill', 'currentColor'); label.textContent = 'time';
    svg.appendChild(label);

    let cur = target[1].slice(), anim = 0;
    const draw = (L, stage) => {
      rows.forEach((r, i) => {
        const x = (r.c - L[i] / 2) * W, w = L[i] * W;
        [r.reg, r.free].forEach(el => { el.setAttribute('x', x.toFixed(1)); el.setAttribute('width', w.toFixed(1)); });
        const isFree = stage === 2 && i >= rows.length - FREE;
        r.reg.style.opacity = isFree ? 0 : 1;
        r.free.style.opacity = isFree ? 1 : 0;
        r.tick.style.fill = isFree ? 'var(--ls-free)' : 'var(--ls-reg)';
      });
      split.style.opacity = stage === 2 ? 1 : 0;
    };
    const notes = {
      1: 'Stage 1: every Gaussian pays a penalty for a short lifespan, so the model explains motion by moving Gaussians instead of swapping them in and out.',
      2: 'Stage 2: the penalty stays on the 90% longest-lived Gaussians. The shortest 10% are free to end early, for example where the motion stops or turns.'
    };
    const setStage = stage => {
      buttons.forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.stage) === stage)));
      if (note) note.textContent = notes[stage];
      const from = cur.slice(), to = target[stage];
      cancelAnimationFrame(anim);
      if (reduce.matches) { cur = to.slice(); draw(cur, stage); return; }
      const t0 = performance.now(), D = 520;
      const step = now => {
        const k = Math.min(1, (now - t0) / D), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
        cur = from.map((f, i) => f + (to[i] - f) * e);
        draw(cur, stage);
        if (k < 1) anim = requestAnimationFrame(step);
      };
      anim = requestAnimationFrame(step);
    };
    buttons.forEach(b => b.addEventListener('click', () => setStage(Number(b.dataset.stage))));
    draw(cur, 1);
    setStage(1);
  });
})();
