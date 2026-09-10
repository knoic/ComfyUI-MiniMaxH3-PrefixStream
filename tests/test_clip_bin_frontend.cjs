// Exercise actual gallery/modal rendering with hostile metadata, without ComfyUI.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const elements = [];
const htmlWrites = [];
class Element {
    constructor(tag) {
        this.tag = tag;
        this.children = [];
        this.style = {};
        this.classList = { add() {} };
        elements.push(this);
    }
    set innerHTML(value) { htmlWrites.push(value); this.children = []; }
    appendChild(child) { this.children.push(child); child.parentNode = this; }
    addEventListener() {}
    querySelector() { return new Element('placeholder'); }
    querySelectorAll() { return []; }
    remove() {}
    pause() {}
}

(async () => {
    const hostile = '<img src=x onerror="globalThis.pwned=true"> & \'quoted\'';
    const clip = { clip_id: hostile, shot_tag: hostile, prompt: hostile, parent_clip_id: hostile,
        created_at: hostile, frames: hostile, duration_seconds: hostile, fps: hostile,
        rating: 3, has_video: true, video_url: '/view?filename=video.mp4' };
    const timers = [];
    const listeners = new Map();
    const context = vm.createContext({
        document: { createElement: tag => new Element(tag), getElementById: () => null,
            head: new Element('head'), body: new Element('body') },
        window: { addEventListener() {}, removeEventListener() {} },
        app: { registerExtension() {} },
        api: { fetchApi: async () => ({ ok: true, json: async () => ({ clips: [clip] }) }),
            addEventListener: (key, fn) => listeners.set(key, fn),
            removeEventListener: key => listeners.delete(key) },
        setTimeout: fn => { timers.push(fn); return timers.length; }, clearTimeout() {},
        URL, console, encodeURIComponent,
    });
    const source = fs.readFileSync(path.join(__dirname, '../web/clip_bin_picker.js'), 'utf8')
        .replace(/^import .*;\r?\n/gm, '')
        .replaceAll('import.meta.url', '"https://localhost/extensions/clip_bin_picker.js"');
    vm.runInContext(source, context);
    const node = { id: 1, size: [520, 380], addDOMWidget() {}, widgets: [
        { name: 'project_name', value: hostile }, { name: 'clip_selection', value: hostile },
        { name: 'filter_rating', value: 'All' },
    ] };
    context.setupClipBinPickerWidget(node);
    await timers.shift()();
    const play = elements.find(e => e.className === 'minimax-clip-play-overlay');
    assert.ok(play, 'gallery renders the video card');
    play.onclick({ stopPropagation() {} });
    assert.ok(elements.find(e => e.className === 'minimax-modal-meta'), 'modal renders');
    for (const value of htmlWrites) {
        assert.ok(!value.includes('<img'), 'metadata must never become an HTML image element');
        assert.ok(!value.includes(hostile), 'metadata must be escaped in every HTML sink');
    }
    assert.ok(htmlWrites.some(value => value.includes('&lt;img')), 'hostile text remains visible as text');
    assert.equal(context.pwned, undefined);
    assert.equal(listeners.size, 2);
    node.onRemoved();
    assert.equal(listeners.size, 0, 'deleted nodes release API subscriptions');
    console.log('PASS: gallery/modal metadata escaping and node subscription cleanup');
})().catch(error => { console.error(error); process.exitCode = 1; });
