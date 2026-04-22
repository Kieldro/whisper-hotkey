#!/usr/bin/env python3
"""Floating status overlay for whisper-hotkey on macOS (AppKit / pyobjc).

A small HUD pill in the top-right of the screen the cursor is on.
The pill fill is a native blur (NSVisualEffectView) with a translucent
color tint on top, so recording reads red-at-a-glance, transcribing
reads blue, etc. — without looking like a painted solid color. A
prominent colored dot reinforces the state; during active recording
the dot pulses.
"""
import json
import os

from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory,
    NSPanel, NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSBackingStoreBuffered, NSScreenSaverWindowLevel,
    NSScreen, NSColor, NSTextField, NSFont, NSFontWeightSemibold,
    NSTimer, NSTextAlignmentLeft, NSMakeRect, NSMakePoint,
    NSView, NSVisualEffectView,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectStateActive,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSEvent,
    NSViewWidthSizable, NSViewHeightSizable,
)
from Foundation import NSObject
from Quartz import CGColorCreateGenericRGB, CABasicAnimation
from Foundation import NSNumber
import objc


STATUS_FILE = os.path.join(
    os.getenv("TMPDIR", "/tmp").rstrip("/"),
    "whisper-hotkey-status.json",
)

# state -> (label, color rgb, tint alpha)
STATE_CONFIG = {
    "recording":    ("Recording",    (0.98, 0.28, 0.30), 0.55),
    "processing":   ("Processing",   (0.98, 0.62, 0.15), 0.50),
    "transcribing": ("Transcribing", (0.30, 0.58, 1.00), 0.50),
    "idle":         ("Idle",         (0.62, 0.66, 0.72), 0.25),
    "error":        ("Error",        (0.92, 0.30, 0.30), 0.60),
}

WIDTH = 164.0
HEIGHT = 34.0
CORNER_RADIUS = HEIGHT / 2     # true capsule
DOT_SIZE = 12.0
DOT_LEFT = 12.0
LABEL_LEFT = DOT_LEFT + DOT_SIZE + 10.0
LABEL_RIGHT_PAD = 14.0
EDGE_MARGIN = 14.0
MENU_BAR_CLEARANCE = 28.0


class StatusOverlay(NSObject):
    """Owns the HUD panel and polls the status file on a timer."""

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

        # Layer 1: native blur/material
        blur = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIDTH, HEIGHT))
        blur.setMaterial_(NSVisualEffectMaterialHUDWindow)
        blur.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        blur.setState_(NSVisualEffectStateActive)
        blur.setWantsLayer_(True)
        blur.layer().setCornerRadius_(CORNER_RADIUS)
        blur.layer().setMasksToBounds_(True)
        blur.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        self.window.setContentView_(blur)

        # Layer 2: colored tint on top of the blur (this is what makes the
        # whole pill feel red/blue/orange at a glance, without looking like
        # a flat painted rect).
        self.tint = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WIDTH, HEIGHT))
        self.tint.setWantsLayer_(True)
        self.tint.layer().setCornerRadius_(CORNER_RADIUS)
        self.tint.layer().setMasksToBounds_(True)
        self.tint.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        blur.addSubview_(self.tint)

        # Layer 3: state dot
        dot_y = (HEIGHT - DOT_SIZE) / 2
        self.dot = NSView.alloc().initWithFrame_(
            NSMakeRect(DOT_LEFT, dot_y, DOT_SIZE, DOT_SIZE))
        self.dot.setWantsLayer_(True)
        self.dot.layer().setCornerRadius_(DOT_SIZE / 2)
        blur.addSubview_(self.dot)

        # Layer 4: label
        label_rect = NSMakeRect(
            LABEL_LEFT, 0, WIDTH - LABEL_LEFT - LABEL_RIGHT_PAD, HEIGHT)
        self.label = NSTextField.alloc().initWithFrame_(label_rect)
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setAlignment_(NSTextAlignmentLeft)
        self.label.setFont_(NSFont.systemFontOfSize_weight_(13.5, NSFontWeightSemibold))
        self.label.setTextColor_(NSColor.whiteColor())
        self.label.setStringValue_("")
        blur.addSubview_(self.label)

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
        label, (r, g, b), tint_alpha = STATE_CONFIG.get(state, STATE_CONFIG["idle"])
        self.label.setStringValue_(label)
        self.tint.layer().setBackgroundColor_(
            CGColorCreateGenericRGB(r, g, b, tint_alpha))
        self.dot.layer().setBackgroundColor_(
            CGColorCreateGenericRGB(r, g, b, 1.0))
        self.dot.layer().removeAllAnimations()
        if state == "recording":
            pulse = CABasicAnimation.animationWithKeyPath_("opacity")
            pulse.setFromValue_(NSNumber.numberWithDouble_(1.0))
            pulse.setToValue_(NSNumber.numberWithDouble_(0.35))
            pulse.setDuration_(0.9)
            pulse.setAutoreverses_(True)
            pulse.setRepeatCount_(1e9)
            self.dot.layer().addAnimation_forKey_(pulse, "pulse")
        # Only reposition when the pill is becoming visible for a new
        # lifecycle. Moving the cursor mid-recording shouldn't drag the
        # pill across displays; it picks the "starting" screen and stays
        # there until the next idle auto-hide cycle.
        if not self.window.isVisible():
            self._reposition()
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
