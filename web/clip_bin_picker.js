import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/**
 * MiniMax H3 Clip Bin - Visual Non-Linear Timeline & Interactive Card Deck Extension.
 *
 * Implements:
 * - Horizontal carousel of clip cards directly inside MiniMaxClipBinPicker node.
 * - Thumbnail previews (First Frame & Tail Handover Frame).
 * - Interactive 5-star ratings with real-time API persistence.
 * - Click-to-select active continuation source (highlights card, syncs clip_selection widget).
 * - Special Auto / Initial mode card.
 * - Auto-refresh on generation completion.
 */

// Inject CSS stylesheet into page header
const styleId = "minimax-clip-bin-styles";
if (!document.getElementById(styleId)) {
    const link = document.createElement("link");
    link.id = styleId;
    link.rel = "stylesheet";
    link.type = "text/css";
    link.href = new URL("./clip_bin_picker.css", import.meta.url).href;
    document.head.appendChild(link);
}

app.registerExtension({
    name: "MiniMaxH3.ClipBinPicker",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "MiniMaxClipBinPicker") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupClipBinPickerWidget(this);
            return r;
        };
    }
});

function setupClipBinPickerWidget(node) {
    // Find relevant widgets
    const projectWidget = node.widgets?.find(w => w.name === "project_name");
    const selectionWidget = node.widgets?.find(w => w.name === "clip_selection");
    const ratingFilterWidget = node.widgets?.find(w => w.name === "filter_rating");

    // Container DOM
    const container = document.createElement("div");
    container.className = "minimax-clip-bin-container";

    // Header
    const header = document.createElement("div");
    header.className = "minimax-clip-bin-header";

    const titleWrap = document.createElement("div");
    titleWrap.className = "minimax-clip-bin-title";
    titleWrap.innerHTML = `🎞️ MiniMax Project Clip Bin: <span class="minimax-clip-bin-project-tag">${projectWidget?.value || "Default_Project"}</span>`;

    const actionsWrap = document.createElement("div");
    actionsWrap.className = "minimax-clip-bin-actions";

    const refreshBtn = document.createElement("button");
    refreshBtn.className = "minimax-clip-bin-refresh-btn";
    refreshBtn.innerText = "🔄 刷新素材库";
    actionsWrap.appendChild(refreshBtn);

    header.appendChild(titleWrap);
    header.appendChild(actionsWrap);
    container.appendChild(header);

    // Deck carousel
    const deck = document.createElement("div");
    deck.className = "minimax-clip-bin-deck";
    container.appendChild(deck);

    // Footer bar
    const footer = document.createElement("div");
    footer.className = "minimax-clip-bin-footer";
    const selectionInfo = document.createElement("div");
    selectionInfo.className = "minimax-clip-bin-selection-info";
    selectionInfo.innerHTML = `选中镜头: <span class="minimax-clip-bin-selected-target">${selectionWidget?.value || "latest"}</span>`;

    const hintText = document.createElement("div");
    hintText.innerText = "👉 点击卡片即可设为当前接力源";
    footer.appendChild(selectionInfo);
    footer.appendChild(hintText);
    container.appendChild(footer);

    // Add DOM widget to node
    const widget = node.addDOMWidget("clip_bin_gallery", "gallery", container, {
        serialize: false,
        hideOnZoom: false,
    });

    // Ensure node has enough width/height to display deck
    if (node.size[0] < 500) {
        node.setSize([520, Math.max(node.size[1], 360)]);
    }

    // Function to render stars
    function renderStars(rating, clipId, projectName) {
        const starWrap = document.createElement("div");
        starWrap.className = "minimax-clip-rating";
        for (let i = 1; i <= 5; i++) {
            const star = document.createElement("span");
            star.className = `minimax-clip-star ${i <= rating ? "filled" : "empty"}`;
            star.innerText = "★";
            star.title = `评级: ${i} 星 (点击修改)`;
            star.onclick = async (e) => {
                e.stopPropagation();
                try {
                    const resp = await api.fetchApi("/minimax/clip_bin/rate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ project: projectName, clip_id: clipId, rating: i })
                    });
                    if (resp.ok) {
                        loadClips();
                    }
                } catch (err) {
                    console.error("[Clip Bin] Failed to update rating:", err);
                }
            };
            starWrap.appendChild(star);
        }
        return starWrap;
    }

    // Function to load and render clips
    async function loadClips() {
        const currentProject = projectWidget?.value || "Default_Project";
        const currentSelection = (selectionWidget?.value || "latest").trim();
        titleWrap.innerHTML = `🎞️ MiniMax Project Clip Bin: <span class="minimax-clip-bin-project-tag">${currentProject}</span>`;

        try {
            const res = await api.fetchApi(`/minimax/clip_bin/list?project=${encodeURIComponent(currentProject)}`);
            if (!res.ok) {
                deck.innerHTML = `<div style="padding: 10px; color: #94a3b8; font-size: 11px;">未连接到后台服务或素材库为空</div>`;
                return;
            }
            const data = await res.json();
            const clips = data.clips || [];

            deck.innerHTML = "";

            // 1. Always append "Auto / Initial" Special Card
            const autoCard = document.createElement("div");
            const isAutoActive = currentSelection.toLowerCase() === "latest" || currentSelection.toLowerCase() === "auto" || currentSelection === "";
            autoCard.className = `minimax-clip-card auto-card ${isAutoActive ? "active" : ""}`;
            autoCard.innerHTML = `
                <div class="minimax-clip-thumb-wrap">
                    <div class="minimax-clip-thumb-placeholder">⚡</div>
                    ${isAutoActive ? '<div class="minimax-clip-active-badge">当前接力源</div>' : ""}
                </div>
                <div class="minimax-clip-body">
                    <div class="minimax-clip-shot-name">✨ Auto / 自动最新</div>
                    <div class="minimax-clip-metrics">
                        <span>首段开辟 / 持续自动</span>
                    </div>
                    <div class="minimax-clip-lineage">智能递推 | 零手动配置</div>
                </div>
            `;
            autoCard.onclick = () => {
                if (selectionWidget) {
                    selectionWidget.value = "latest";
                    selectionWidget.callback?.(selectionWidget.value);
                }
                updateSelectionDisplay("latest");
                loadClips();
            };
            deck.appendChild(autoCard);

            // 2. Filter clips by rating if applicable
            let minRating = 1;
            const rfVal = ratingFilterWidget?.value || "";
            if (rfVal.includes("⭐⭐⭐⭐⭐")) minRating = 5;
            else if (rfVal.includes("⭐⭐⭐⭐")) minRating = 4;
            else if (rfVal.includes("⭐⭐⭐")) minRating = 3;

            const filteredClips = clips.filter(c => (c.rating || 3) >= minRating);

            if (filteredClips.length === 0) {
                const emptyMsg = document.createElement("div");
                emptyMsg.style.cssText = "padding: 20px 10px; color: #64748b; font-size: 11px; white-space: nowrap;";
                emptyMsg.innerText = "素材库暂无片段，执行生成后将自动收入...";
                deck.appendChild(emptyMsg);
            } else {
                filteredClips.forEach(clip => {
                    const card = document.createElement("div");
                    const isActive = currentSelection === clip.clip_id;
                    card.className = `minimax-clip-card ${isActive ? "active" : ""}`;

                    // Thumbnail
                    const thumbWrap = document.createElement("div");
                    thumbWrap.className = "minimax-clip-thumb-wrap";

                    if (clip.thumbnail_url) {
                        const img = document.createElement("img");
                        img.className = "minimax-clip-thumb";
                        img.src = clip.thumbnail_url;
                        img.loading = "lazy";
                        img.onerror = () => {
                            thumbWrap.innerHTML = `<div class="minimax-clip-thumb-placeholder">🎬</div>`;
                        };
                        thumbWrap.appendChild(img);
                    } else {
                        thumbWrap.innerHTML = `<div class="minimax-clip-thumb-placeholder">🎬</div>`;
                    }

                    if (isActive) {
                        const badge = document.createElement("div");
                        badge.className = "minimax-clip-active-badge";
                        badge.innerText = "当前接力源";
                        thumbWrap.appendChild(badge);
                    }
                    card.appendChild(thumbWrap);

                    // Body
                    const body = document.createElement("div");
                    body.className = "minimax-clip-body";

                    // Stars
                    body.appendChild(renderStars(clip.rating || 3, clip.clip_id, currentProject));

                    // Shot tag
                    const shotName = document.createElement("div");
                    shotName.className = "minimax-clip-shot-name";
                    shotName.innerText = clip.shot_tag || "Shot";
                    shotName.title = `${clip.shot_tag} (${clip.clip_id})`;
                    body.appendChild(shotName);

                    // Metrics
                    const metrics = document.createElement("div");
                    metrics.className = "minimax-clip-metrics";
                    metrics.innerHTML = `<span>${clip.frames || 124}帧</span><span>${clip.duration_seconds || 5.2}s</span>`;
                    body.appendChild(metrics);

                    // Lineage / Parent
                    if (clip.parent_clip_id) {
                        const lineage = document.createElement("div");
                        lineage.className = "minimax-clip-lineage";
                        lineage.innerText = `↳ 衍生自: ${clip.parent_clip_id.slice(-8)}`;
                        lineage.title = `父镜头: ${clip.parent_clip_id}`;
                        body.appendChild(lineage);
                    }

                    card.appendChild(body);

                    // Click to select
                    card.onclick = () => {
                        if (selectionWidget) {
                            selectionWidget.value = clip.clip_id;
                            selectionWidget.callback?.(selectionWidget.value);
                        }
                        updateSelectionDisplay(clip.shot_tag ? `${clip.shot_tag} (${clip.clip_id.slice(-8)})` : clip.clip_id);
                        loadClips();
                    };

                    deck.appendChild(card);
                });
            }
        } catch (e) {
            console.warn("[Clip Bin] Error loading clips:", e);
        }
    }

    function updateSelectionDisplay(val) {
        selectionInfo.innerHTML = `选中镜头: <span class="minimax-clip-bin-selected-target">${val}</span>`;
    }

    // Bind refresh button
    refreshBtn.onclick = (e) => {
        e.stopPropagation();
        loadClips();
    };

    // Watch projectWidget changes
    if (projectWidget) {
        const origCallback = projectWidget.callback;
        projectWidget.callback = function (v) {
            const r = origCallback ? origCallback.apply(this, arguments) : undefined;
            loadClips();
            return r;
        };
    }

    // Watch rating filter changes
    if (ratingFilterWidget) {
        const origRfCallback = ratingFilterWidget.callback;
        ratingFilterWidget.callback = function (v) {
            const r = origRfCallback ? origRfCallback.apply(this, arguments) : undefined;
            loadClips();
            return r;
        };
    }

    // Watch clip_selection changes
    if (selectionWidget) {
        const origSelCallback = selectionWidget.callback;
        selectionWidget.callback = function (v) {
            const r = origSelCallback ? origSelCallback.apply(this, arguments) : undefined;
            updateSelectionDisplay(v);
            loadClips();
            return r;
        };
    }

    // Initial load
    setTimeout(loadClips, 200);

    // Auto-refresh when generation execution finishes
    api.addEventListener("executed", (e) => {
        if (e.detail?.node === String(node.id) || e.detail?.output?.ui?.images) {
            setTimeout(loadClips, 500);
        }
    });

    api.addEventListener("status", (e) => {
        if (e.detail?.exec_info?.queue_remaining === 0) {
            setTimeout(loadClips, 600);
        }
    });
}
