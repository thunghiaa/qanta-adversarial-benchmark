style_content = """
pre, code {
    background-color: #272822;
}
    .scrollable {
        font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;
        height: 500px;
        overflow: auto;
    }
    """
dark_mode_gradio_js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

light_mode_gradio_js = """
function refresh() {
    const url = new URL(window.location);
    console.log("in refresh function");
    console.log(url.searchParams.get('__theme'));

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

custom_css = """
:root {
    --card-bg-light: #fff;
    --card-bg-dark: #23272e;
    --border-light: #e0e0e0;
    --border-dark: #444a57;
    --text-light: #222;
    --text-dark: #f2f2f2;
    --block-bg-light: #f9f9f9;
    --block-bg-dark: #2c313a;
}
@media (prefers-color-scheme: dark) {
    .stage-card {
        background: var(--card-bg-dark) !important;
        border-color: var(--border-dark) !important;
        color: var(--text-dark) !important;
    }
    .status-block {
        background: var(--block-bg-dark) !important;
        border-color: var(--border-dark) !important;
        color: var(--text-dark) !important;
    }
    .stage-title, .status-block .label {
        color: var(--text-dark) !important;
    }
}
@media (prefers-color-scheme: light), (prefers-color-scheme: no-preference) {
    .stage-card {
        background: var(--card-bg-light) !important;
        border-color: var(--border-light) !important;
        color: var(--text-light) !important;
    }
    .status-block {
        background: var(--block-bg-light) !important;
        border-color: var(--border-light) !important;
        color: var(--text-light) !important;
    }
    .stage-title, .status-block .label {
        color: var(--text-light) !important;
    }
}
.cell-wrap span {
    user-select: text !important;
}
.cell-wrap {
    user-select: text !important;
}
.pipeline-container {
    display: flex;
    flex-direction: row;
    gap: 12px;
    justify-content: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
}
.stage-card {
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    padding: 12px 10px;
    min-width: 160px;
    display: flex;
    flex-direction: column;
    align-items: center;
    border: 1.5px solid var(--border-light);
    position: relative;
    margin: 0;
}
.stage-title {
    font-size: 1.05em;
    font-weight: bold;
    margin-bottom: 7px;
    letter-spacing: 0.5px;
}
.status-row {
    display: flex;
    flex-direction: row;
    gap: 6px;
}
.status-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    padding: 6px 6px;
    min-width: 38px;
    min-height: 38px;
    font-size: 1em;
    font-weight: 500;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    background: var(--block-bg-light);
    border: 1.2px solid var(--border-light);
    transition: box-shadow 0.2s;
    margin: 0;
}
.status-block .emoji {
    font-size: 1.15em;
    margin-bottom: 0px;
}
.status-block .count {
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 0px;
}
.status-block .label {
    font-size: 0.8em;
    color: var(--text-light);
}

.my-btn {
    white-space: pre-line;
    text-align: center;
}
"""
