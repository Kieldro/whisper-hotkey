#!/usr/bin/env python3
"""
Floating status overlay for whisper-hotkey on GNOME.
Watches the status JSON file and displays a small widget in the corner.
"""

import json
import os
import sys

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, Gdk, GLib

STATUS_FILE = os.path.join(
    os.getenv("XDG_RUNTIME_DIR", "/tmp"),
    "whisper-hotkey-status.json"
)

# State -> (icon, label, css_class)
STATE_CONFIG = {
    "recording":    ("\U0001f534", "REC", "recording"),
    "processing":   ("\u2699\ufe0f", "Processing", "processing"),
    "transcribing": ("\U0001f4dd", "Transcribing", "transcribing"),
    "idle":         ("\u23f8\ufe0f", "Idle", "idle"),
    "error":        ("\u26a0\ufe0f", "Error", "error"),
}

CSS = b"""
window {
    border-radius: 12px;
}

.overlay-box {
    padding: 12px 28px;
    border-radius: 16px;
    font-family: monospace;
    font-size: 26px;
    font-weight: bold;
}

.recording {
    background: rgba(220, 38, 38, 0.92);
    color: white;
}

.recording .dot {
    color: #ff6b6b;
    font-size: 18px;
}

.processing {
    background: rgba(217, 119, 6, 0.92);
    color: white;
}

.transcribing {
    background: rgba(37, 99, 235, 0.92);
    color: white;
}

.idle {
    background: rgba(55, 65, 81, 0.85);
    color: rgba(255, 255, 255, 0.7);
}

.error {
    background: rgba(127, 29, 29, 0.92);
    color: white;
}
"""


class StatusOverlay(Gtk.Window):
    def __init__(self):
        super().__init__(type=Gtk.WindowType.TOPLEVEL)

        # Window setup
        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_resizable(False)
        self.set_type_hint(Gdk.WindowTypeHint.UTILITY)

        # Transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)
        self.set_app_paintable(True)

        # CSS
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(
            screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Layout
        self.box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.box.get_style_context().add_class("overlay-box")
        self.box.get_style_context().add_class("idle")
        self.box.set_halign(Gtk.Align.CENTER)
        self.add(self.box)

        self.icon_label = Gtk.Label()
        self.icon_label.set_text("\u23f8\ufe0f")
        self.box.pack_start(self.icon_label, False, False, 0)

        self.label = Gtk.Label()
        self.label.set_markup("<b>Idle</b>")
        self.box.pack_start(self.label, False, False, 0)

        self.current_state = None
        self.hide_timeout_id = None
        self._dragging = False
        self._drag_x = 0
        self._drag_y = 0
        self._custom_position = None

        # Enable mouse events for dragging
        self.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK |
            Gdk.EventMask.BUTTON_RELEASE_MASK |
            Gdk.EventMask.POINTER_MOTION_MASK
        )
        self.connect("button-press-event", self._on_button_press)
        self.connect("button-release-event", self._on_button_release)
        self.connect("motion-notify-event", self._on_motion)

        self.show_all()
        self._position_window()

        # Poll status file every 200ms
        GLib.timeout_add(200, self._poll_status)

    def _position_window(self):
        display = Gdk.Display.get_default()
        seat = display.get_default_seat()
        pointer = seat.get_pointer() if seat else None
        if pointer:
            _, x, y = pointer.get_position()
            monitor = display.get_monitor_at_point(x, y)
        else:
            monitor = display.get_primary_monitor() or display.get_monitor(0)
        geom = monitor.get_geometry()
        self.resize(1, 1)  # force size recalc

        # Wait for allocation
        GLib.idle_add(self._do_position, geom)

    def _do_position(self, geom):
        if self._custom_position:
            self.move(*self._custom_position)
            return False
        alloc = self.get_allocation()
        # Top-right corner (default)
        x = geom.x + geom.width - alloc.width - 16
        y = geom.y + 8
        self.move(x, y)
        return False

    def _on_button_press(self, widget, event):
        if event.button == 1:
            self._dragging = True
            self._drag_x = event.x
            self._drag_y = event.y

    def _on_button_release(self, widget, event):
        if event.button == 1:
            self._dragging = False

    def _on_motion(self, widget, event):
        if self._dragging:
            x = int(event.x_root - self._drag_x)
            y = int(event.y_root - self._drag_y)
            self.move(x, y)
            self._custom_position = (x, y)

    def _poll_status(self):
        try:
            with open(STATUS_FILE, "r") as f:
                data = json.load(f)
            state = data.get("state", "idle")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            state = None

        if state is None:
            # No daemon running — hide
            if self.get_visible():
                self.hide()
            self.current_state = None
            return True

        if state != self.current_state:
            self.current_state = state
            self._update_display(state)

        return True

    def _update_display(self, state):
        icon, label_text, css_class = STATE_CONFIG.get(state, STATE_CONFIG["idle"])

        # Update CSS class
        ctx = self.box.get_style_context()
        for cls in ("recording", "processing", "transcribing", "idle", "error"):
            ctx.remove_class(cls)
        ctx.add_class(css_class)

        self.icon_label.set_text(icon)
        self.label.set_markup(f"<b>{label_text}</b>")

        # Cancel any pending hide
        if self.hide_timeout_id:
            GLib.source_remove(self.hide_timeout_id)
            self.hide_timeout_id = None

        if not self.get_visible():
            self.show_all()

        self._position_window()

        # Auto-hide after 3s when idle
        if state == "idle":
            self.hide_timeout_id = GLib.timeout_add(3000, self._auto_hide)

    def _auto_hide(self):
        self.hide_timeout_id = None
        if self.current_state == "idle":
            self.hide()
        return False


def main():
    overlay = StatusOverlay()
    try:
        Gtk.main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
