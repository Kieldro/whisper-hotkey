#!/usr/bin/env python3
"""Floating status overlay for whisper-hotkey on macOS (AppKit / pyobjc).

Watches whisper-hotkey-status.json and shows a small rounded pill in the
top-right corner with the current recording/transcribing/idle state.
Auto-hides 3 s after returning to idle. Drag by the pill body to move.
"""
import json
import os
import sys

from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory,
    NSPanel, NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSBackingStoreBuffered, NSScreenSaverWindowLevel,
    NSScreen, NSColor, NSTextField, NSFont,
    NSTimer, NSTextAlignmentCenter, NSMakeRect, NSMakePoint,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSEvent,
)
from Foundation import NSObject
from Quartz import CGColorCreateGenericRGB
import objc


STATUS_FILE = os.path.join(
    os.getenv("TMPDIR", "/tmp").rstrip("/"),
    "whisper-hotkey-status.json",
)

# state -> (icon, label, (r, g, b), alpha)
# Icons are monochrome Unicode symbols so they render white against the
# colored pill background. The emoji equivalents (🔴 ⏸️ 📝 ⚙️ ⚠️) all force
# their own colors and clash with the fill.
STATE_CONFIG = {
    "recording":    ("●", "REC",          (0.86, 0.15, 0.15), 0.92),
    "processing":   ("◐", "Processing",   (0.85, 0.47, 0.02), 0.92),
    "transcribing": ("✎", "Transcribing", (0.15, 0.39, 0.92), 0.92),
    "idle":         ("·", "Idle",         (0.22, 0.25, 0.32), 0.85),
    "error":        ("!", "Error",        (0.50, 0.11, 0.11), 0.92),
}

WIDTH = 220.0
HEIGHT = 44.0
EDGE_MARGIN = 16.0
MENU_BAR_CLEARANCE = 24.0


class StatusOverlay(NSObject):
    """Owns the panel and polls the status file on a timer."""

    def init(self):
        self = objc.super(StatusOverlay, self).init()
        if self is None:
            return None

        mask = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIDTH, HEIGHT),
            mask, NSBackingStoreBuffered, False,
        )
        self.window.setLevel_(NSScreenSaverWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setHasShadow_(True)
        self.window.setMovableByWindowBackground_(True)
        self.window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        self.label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIDTH, HEIGHT))
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setAlignment_(NSTextAlignmentCenter)
        self.label.setFont_(NSFont.boldSystemFontOfSize_(18))
        self.label.setTextColor_(NSColor.whiteColor())
        self.label.setWantsLayer_(True)
        self.label.layer().setCornerRadius_(12.0)
        self.label.layer().setMasksToBounds_(True)
        self.window.setContentView_(self.label)

        self.current_state = None
        self.hide_timer = None
        return self

    def tick_(self, timer):
        try:
            with open(STATUS_FILE) as f:
                state = json.load(f).get("state", "idle")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            state = None
        if state != self.current_state:
            self.current_state = state
            self._render(state)

    def autoHide_(self, timer):
        self.hide_timer = None
        if self.current_state == "idle":
            self.window.orderOut_(None)

    @objc.python_method
    def _render(self, state):
        if state is None:
            if self.window.isVisible():
                self.window.orderOut_(None)
            return
        icon, label, (r, g, b), a = STATE_CONFIG.get(state, STATE_CONFIG["idle"])
        self.label.setStringValue_(f"{icon}  {label}")
        self.label.layer().setBackgroundColor_(CGColorCreateGenericRGB(r, g, b, a))
        self._reposition()
        if not self.window.isVisible():
            self.window.orderFrontRegardless()
        if self.hide_timer is not None:
            self.hide_timer.invalidate()
            self.hide_timer = None
        if state == "idle":
            self.hide_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                3.0, self, "autoHide:", None, False,
            )

    @objc.python_method
    def _reposition(self):
        """Put the pill in the top-right of the screen the mouse is on.

        NSScreen.mainScreen() is unreliable in a headless LSUIElement
        process — with multiple monitors it can return any of them. The
        mouse cursor is a stable "where the user is looking" heuristic.
        """
        mouse = NSEvent.mouseLocation()
        target = None
        for s in NSScreen.screens():
            f = s.frame()
            if (f.origin.x <= mouse.x < f.origin.x + f.size.width
                    and f.origin.y <= mouse.y < f.origin.y + f.size.height):
                target = s
                break
        if target is None:
            target = NSScreen.mainScreen()
        f = target.frame()
        x = f.origin.x + f.size.width - WIDTH - EDGE_MARGIN
        y = f.origin.y + f.size.height - HEIGHT - EDGE_MARGIN - MENU_BAR_CLEARANCE
        self.window.setFrameOrigin_(NSMakePoint(x, y))


# objc needs explicit selector signatures for methods called via NSTimer/target
StatusOverlay.tick_ = objc.selector(StatusOverlay.tick_, signature=b"v@:@")
StatusOverlay.autoHide_ = objc.selector(StatusOverlay.autoHide_, signature=b"v@:@")


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    overlay = StatusOverlay.alloc().init()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.2, overlay, "tick:", None, True,
    )
    app.run()


if __name__ == "__main__":
    main()
