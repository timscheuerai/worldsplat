"""Step 4: Web viewer for navigating Gaussian Splat scenes.

Serves a local web page that loads the .ply or .splat file and lets you
fly around the 3D scene in your browser.

Uses antimatter15/splat (pure WebGL, zero dependencies).
"""

import http.server
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WorldSplat Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; overflow: hidden; font-family: system-ui; }
  #info {
    position: fixed; top: 10px; left: 10px; color: #fff;
    background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 6px;
    font-size: 13px; z-index: 10; pointer-events: none;
  }
  #info h3 { margin-bottom: 4px; }
  #info p { opacity: 0.7; font-size: 11px; }
  canvas { width: 100vw; height: 100vh; display: block; }
</style>
</head>
<body>
<div id="info">
  <h3>WorldSplat Viewer</h3>
  <p>Click & drag to rotate. Scroll to zoom. WASD to move.</p>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.164.0/build/three.module.js"
  }
}
</script>
<script type="module">
import * as GaussianSplats3D from 'https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.5/build/gaussian-splats-3d.module.js';

const viewer = new GaussianSplats3D.Viewer({
  cameraUp: [0, -1, 0],
  initialCameraPosition: [0, -2, -6],
  initialCameraLookAt: [0, 0, 0],
  selfDrivenMode: true,
});
viewer.addSplatScene('SPLAT_URL', { splatAlphaRemovalThreshold: 5 })
  .then(() => { viewer.start(); });
</script>
</body>
</html>"""


def launch_viewer(ply_path: Path, port: int = 8080, open_browser: bool = True):
    """Launch a local web viewer for a Gaussian Splat .ply file.

    Serves the PLY file over HTTP and opens antimatter15/splat viewer
    pointing at it.

    Args:
        ply_path: Path to the .ply file to view.
        port: Local port to serve on.
        open_browser: Whether to open the browser automatically.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    serve_dir = ply_path.parent
    ply_name = ply_path.name

    # Create the HTML page that embeds antimatter15/splat
    splat_url = f"http://localhost:{port}/{ply_name}"
    html_content = VIEWER_HTML.replace("SPLAT_URL", splat_url)
    html_path = serve_dir / "viewer.html"
    html_path.write_text(html_content)

    # Serve the directory
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress request logs

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
            super().end_headers()

    server = http.server.HTTPServer(("0.0.0.0", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    viewer_url = f"http://localhost:{port}/viewer.html"
    logger.info(f"Serving PLY at http://localhost:{port}/{ply_name}")
    logger.info(f"Viewer: {viewer_url}")

    if open_browser:
        import webbrowser
        webbrowser.open(viewer_url)

    print(f"\nViewer running at: {viewer_url}")
    print(f"PLY file served at: http://localhost:{port}/{ply_name}")
    print(f"\nYou can also drag-drop the .ply file into https://antimatter15.com/splat/")
    print(f"Or open it in SuperSplat: https://superspl.at/editor")
    print(f"\nPress Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print("\nServer stopped.")
