/* TESLA project page: results shown on the page.
   Every clip was cut at native size from the supplementary video and plays at 12 fps.
   view  'front': fixed front camera, 24 frames
         'free' : moving camera, 96 frames (the 24-frame animation repeats while the camera moves)
   w, h  native size in pixels; the page never shows a clip larger than this
   input true when static/inputs/<id>.webp (the reference image) exists
   prompt '' when the prompt is not known; fill it in and the page shows it
   The order inside each wall was chosen so that the rows come out at similar heights on a 960 px column. */
window.TESLA_CLIPS = [
  // Front views (supplementary Figs. 11 and 12)
  { id: 'vampire',      view: 'front', w: 466, h: 470, input: true,  prompt: 'The vampire flower made a bite downward.' },
  { id: 'trex',         view: 'front', w: 212, h: 262, input: true,  prompt: 'The dinosaur ran forward with its steps.' },
  { id: 'mannequin',    view: 'front', w: 230, h: 358, input: false, prompt: '' },
  { id: 'propball',     view: 'front', w: 354, h: 400, input: true,  prompt: 'The propeller is rotating.' },
  { id: 'tulip',        view: 'front', w: 378, h: 336, input: true,  prompt: 'The flowers slowly bloom.' },
  { id: 'cowboy',       view: 'front', w: 214, h: 504, input: true,  prompt: 'The man preparing to punch took a step forward.' },
  { id: 'heli',         view: 'front', w: 394, h: 202, input: true,  prompt: 'The propeller is rotating.' },
  { id: 'kirlia',       view: 'front', w: 302, h: 388, input: false, prompt: '' },
  { id: 'blade',        view: 'front', w: 376, h: 258, input: true,  prompt: 'The blade extends into the handle.' },
  { id: 'foxy',         view: 'front', w: 240, h: 452, input: false, prompt: '' },
  { id: 'palm',         view: 'front', w: 350, h: 456, input: true,  prompt: 'The tree leaned slightly.' },
  { id: 'robot',        view: 'front', w: 304, h: 410, input: true,  prompt: 'The robot moves forward.' },
  { id: 'fan',          view: 'front', w: 324, h: 414, input: true,  prompt: 'The fan starts to spin.' },
  { id: 'person',       view: 'front', w: 246, h: 436, input: true,  prompt: 'This person is looking down.' },
  { id: 'rock',         view: 'front', w: 378, h: 490, input: false, prompt: '' },
  { id: 'huggy',        view: 'front', w: 200, h: 406, input: false, prompt: '' },

  // Free viewpoints (supplementary Figs. 13 and 14)
  { id: 'spider',       view: 'free',  w: 432, h: 422, input: false, prompt: 'The spider moved forward slowly.' },
  { id: 'fan-olive',    view: 'free',  w: 406, h: 474, input: false, prompt: 'The electric fan is rotating.' },
  { id: 'plane',        view: 'free',  w: 538, h: 482, input: false, prompt: '' },
  { id: 'clouds',       view: 'free',  w: 478, h: 498, input: false, prompt: 'The clouds keep raining.' },
  { id: 'sonic',        view: 'free',  w: 312, h: 336, input: false, prompt: '' },
  { id: 'rhino',        view: 'free',  w: 454, h: 326, input: false, prompt: 'The rhinoceros walked forward.' },
  { id: 'golem',        view: 'free',  w: 276, h: 376, input: false, prompt: '' },
  { id: 'turret',       view: 'free',  w: 338, h: 350, input: false, prompt: '' },
  { id: 'trex-orbit',   view: 'free',  w: 486, h: 380, input: false, prompt: '' },
  { id: 'spider-black', view: 'free',  w: 506, h: 364, input: false, prompt: 'The spider is crawling.' },
  { id: 'birds',        view: 'free',  w: 564, h: 448, input: false, prompt: 'A flock of birds flies forward.' },
  { id: 'gift',         view: 'free',  w: 340, h: 524, input: false, prompt: '' },
  { id: 'bud',          view: 'free',  w: 398, h: 424, input: false, prompt: 'The flower bud blossoms.' },
  { id: 'drone',        view: 'free',  w: 626, h: 436, input: false, prompt: 'The wings of the drone are rotating.' },
  { id: 'dragonfly',    view: 'free',  w: 316, h: 398, input: false, prompt: '' },

  // Hero only: the method demo from the supplementary video. It appears to be the same generation as 'bud'
  // (about 39 dB PSNR between the two cuts), so it is not repeated in the gallery.
  { id: 'flower-demo',  view: 'free',  w: 398, h: 424, input: true,  prompt: 'The flower bud slowly blooms into a flower.', hidden: true }
];

// Hero picker, in order; the first one is shown on load.
window.TESLA_FEATURED = ['flower-demo', 'vampire', 'cowboy', 'trex', 'fan', 'propball', 'heli', 'blade'];
