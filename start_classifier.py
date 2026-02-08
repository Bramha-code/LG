"""
Simple HTTP server to serve the standalone ONNX classifier.
Run this, then open http://localhost:8080 in your browser.

Usage:
    python start_classifier.py
"""

import http.server
import socketserver
import webbrowser
import os

PORT = 8080
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("LQC Defect Code Logger - Standalone ONNX")
print("=" * 50)
print(f"\nServing at: http://localhost:{PORT}")
print(f"Open: http://localhost:{PORT}/realtime_classifier.html")
print("\nFiles to load in the browser:")
print(f"  Model:   trained/audio_classifier.onnx")
print(f"  Mapping: trained/class_mapping.json")
print(f"  LQC:     LQC Code System.xlsx (optional)")
print("\nPress Ctrl+C to stop.\n")

webbrowser.open(f"http://localhost:{PORT}/realtime_classifier.html")

handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
