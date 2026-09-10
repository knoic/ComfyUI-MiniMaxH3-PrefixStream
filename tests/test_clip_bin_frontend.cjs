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
        this.classList = { add() {}, remove() {} };
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
    const clip1 = { clip_id: 'shot1_id', shot_tag: 'Shot 1', prompt: hostile, parent_clip_id: '',
        created_at: '2026-09-08 12:00:00', frames: 124, duration_seconds: 5.16, fps: 24,
        rating: 4, has_video: true, video_url: '/view?filename=shot1.mp4' };
    const clip2 = { clip_id: hostile, shot_tag: hostile, prompt: hostile, parent_clip_id: 'shot1_id',
        created_at: hostile, frames: hostile, duration_seconds: hostile, fps: hostile,
        rating: 3, has_video: true, video_url: '/view?filename=video.mp4' };
    const timers = [];
    const listeners = new Map();
    const context = vm.createContext({
        document: { createElement: tag => new Element(tag), getElementById: () => null,
            head: new Element('head'), body: new Element('body') },
        window: { addEventListener() {}, removeEventListener() {} },
        app: { registerExtension() {} },
        api: { fetchApi: async () => ({ ok: true, json: async () => ({ clips: [clip1, clip2] }) }),
            addEventListener: (key, fn) => listeners.set(key, fn),
            removeEventListener: key => listeners.delete(key) },
        setTimeout: fn => { timers.push(fn); return timers.length; }, clearTimeout() {},
        URL, console, encodeURIComponent,
    });
    const source = fs.readFileSync(path.join(__dirname, '../web/clip_bin_picker.js'), 'utf8')
        .replace(/^import .*;\r?\n/gm, '')
        .replaceAll('import.meta.url', '"https://localhost/extensions/clip_bin_picker.js"');
    vm.runInContext(source, context);

    async function drainTimers() {
        while (timers.length > 0) {
            await timers.shift()();
        }
    }

    // Test 1: Deck View with MiniMaxClipBinPicker
    const node = { id: 1, size: [520, 380], addDOMWidget() {}, widgets: [
        { name: 'project_name', value: hostile }, { name: 'clip_selection', value: hostile },
        { name: 'filter_rating', value: 'All' },
    ] };
    context.setupClipBinPickerWidget(node);
    await drainTimers();
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
    await drainTimers();
    assert.equal(listeners.size, 0, 'deleted nodes release API subscriptions');

    // Test 2: Tree View with MiniMaxClipBinTreePicker
    const treeNode = { id: 2, type: 'MiniMaxClipBinTreePicker', size: [680, 440], addDOMWidget() {}, setSize() {}, widgets: [
        { name: 'project_name', value: hostile }, { name: 'clip_selection', value: hostile },
        { name: 'filter_rating', value: 'All' }, { name: 'view_mode', value: 'Tree (关系树)' },
    ] };
    context.setupClipBinPickerWidget(treeNode);
    await drainTimers();
    assert.ok(elements.some(e => e.className && e.className.includes('minimax-tree-auto-node')), 'tree renders auto root');
    assert.ok(elements.some(e => e.className && e.className.includes('minimax-tree-node')), 'tree renders clip node');
    for (const value of htmlWrites) {
        assert.ok(!value.includes('<img'), 'metadata in tree must never become an HTML image element');
        assert.ok(!value.includes(hostile), 'metadata in tree must be escaped in every HTML sink');
    }
    treeNode.onRemoved();
    await drainTimers();

    console.log('PASS: gallery/modal/tree metadata escaping and node subscription cleanup');
})().catch(error => { console.error(error); process.exitCode = 1; });
