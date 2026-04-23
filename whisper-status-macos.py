#!/usr/bin/env python3
"""Floating status overlay for whisper-hotkey on macOS (AppKit / pyobjc).

A compact HUD pill in the top-right of the screen the cursor is on.
Dark translucent blur base, bright colored state dot on the left, clean
SF Pro label on the right. Designed to read as "native macOS HUD" the
way Focus / Do-Not-Disturb and screen-recording indicators do.
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
    NSLineBreakByTruncatingHead,
)
from Foundation import NSObject, NSNumber
from Quartz import CGColorCreateGenericRGB, CABasicAnimation
import objc


STATUS_FILE = os.path.join(
    os.getenv("TMPDIR", "/tmp").rstrip("/"),
    "whisper-hotkey-status.json",
)

# state -> (label, bright dot color, pulse bool)
STATE_CONFIG = {
    "recording":    ("Recording",    (1.00, 0.25, 0.28), True),
    "processing":   ("Processing",   (1.00, 0.72, 0.20), False),
    "transcribing": ("Transcribing", (0.35, 0.70, 1.00), False),
    "idle":         ("Idle",         (0.72, 0.76, 0.82), False),
    "error":        ("Error",        (1.00, 0.35, 0.35), False),
}

HEIGHT = 44.0
CORNER_RADIUS = HEIGHT / 2         # true capsule
DOT_SIZE = 14.0
SIDE_PAD = 20.0                    # left pad to dot & right pad after label
DOT_TO_LABEL_GAP = 12.0
FONT_POINT_SIZE = 18.0
TOP_MARGIN = 64.0                  # below the menu bar with breathing room
MAX_LABEL_WIDTH = 700.0            # cap on live-transcript width; truncate past this


class StatusOverlay(NSObject):
    """Owns the HUD panel and polls the status file on a timer."""

    def init(self):
        self = objc.super(StatusOverlay, self).init()
        if self is None:
            return None

        mask = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        # Start with a provisional width; _render resizes to fit each label.
        self.window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 160, HEIGHT),
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

        # Dark translucent HUD blur as the entire pill background.
        self.blur = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, 160, HEIGHT))
        self.blur.setMaterial_(NSVisualEffectMaterialHUDWindow)
        self.blur.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        self.blur.setState_(NSVisualEffectStateActive)
        self.blur.setWantsLayer_(True)
        self.blur.layer().setCornerRadius_(CORNER_RADIUS)
        self.blur.layer().setMasksToBounds_(True)
        self.blur.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        self.window.setContentView_(self.blur)

        # State dot — the color element. Bright, high-contrast against the
        # dark blur. Eye anchor at a glance.
        dot_y = (HEIGHT - DOT_SIZE) / 2
        self.dot = NSView.alloc().initWithFrame_(
            NSMakeRect(SIDE_PAD, dot_y, DOT_SIZE, DOT_SIZE))
        self.dot.setWantsLayer_(True)
        self.dot.layer().setCornerRadius_(DOT_SIZE / 2)
        self.blur.addSubview_(self.dot)

        # Label: SF Pro semibold, white on dark blur. Frame is set in _render.
        self.label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 100, HEIGHT))
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setAlignment_(NSTextAlignmentLeft)
        self.label.setFont_(NSFont.systemFontOfSize_weight_(FONT_POINT_SIZE, NSFontWeightSemibold))
        self.label.setTextColor_(NSColor.whiteColor())
        # Truncate from the left when live text exceeds MAX_LABEL_WIDTH so
        # the most recent words stay visible as the user dictates.
        self.label.cell().setLineBreakMode_(NSLineBreakByTruncatingHead)
        self.label.setStringValue_("")
        self.blur.addSubview_(self.label)

        self.current_state = None
        self.current_live_text = ""
        self.hide_timer = None
        return self

    def tick_(self, timer):
        state = None
        live_text = ""
        try:
            with open(STATUS_FILE) as f:
                data = json.load(f)
            state = data.get("state", "idle")
            live_text = data.get("live_text", "") or ""
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            state = None
        if state != self.current_state or live_text != self.current_live_text:
            self.current_state = state
            self.current_live_text = live_text
            self._render(state, live_text)

    def autoHide_(self, timer):
        self.hide_timer = None
        if self.current_state == "idle":
            self.window.orderOut_(None)

    @objc.python_method
    def _render(self, state, live_text=""):
        if state is None:
            if self.window.isVisible():
                self.window.orderOut_(None)
            return
        label, (r, g, b), pulses = STATE_CONFIG.get(state, STATE_CONFIG["idle"])
        # If we have live transcript text, show that instead of the state
        # label (streaming engines stay in "recording" state but fill the
        # pill with partials as they arrive).
        display_text = live_text if live_text else label
        self.label.setStringValue_(display_text)
        # Size label to its natural width, then cap at MAX_LABEL_WIDTH.
        self.label.sizeToFit()
        text_w = min(self.label.frame().size.width, MAX_LABEL_WIDTH)
        label_x = SIDE_PAD + DOT_SIZE + DOT_TO_LABEL_GAP
        pill_w = label_x + text_w + SIDE_PAD
        label_frame = self.label.frame()
        self.label.setFrame_(NSMakeRect(label_x,
                                        (HEIGHT - label_frame.size.height) / 2,
                                        text_w, label_frame.size.height))
        # Resize window + blur view (autoresizing mask handles the blur).
        # Anchor to horizontal center so the pill grows/shrinks from both
        # sides as the live transcript updates, and stays centered on
        # screen regardless of content length.
        f = self.window.frame()
        center_x = f.origin.x + f.size.width / 2
        new_x = center_x - pill_w / 2
        self.window.setFrame_display_(NSMakeRect(new_x, f.origin.y, pill_w, HEIGHT), True)
        self.dot.layer().setBackgroundColor_(
            CGColorCreateGenericRGB(r, g, b, 1.0))
        self.dot.layer().removeAllAnimations()
        if pulses:
            pulse = CABasicAnimation.animationWithKeyPath_("opacity")
            pulse.setFromValue_(NSNumber.numberWithDouble_(1.0))
            pulse.setToValue_(NSNumber.numberWithDouble_(0.3))
            pulse.setDuration_(0.85)
            pulse.setAutoreverses_(True)
            pulse.setRepeatCount_(1e9)
            self.dot.layer().addAnimation_forKey_(pulse, "pulse")
        # Reposition only when the pill is transitioning from hidden to
        # visible — we don't want the pill to jump displays mid-recording
        # if the cursor moves.
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
        sf = target.frame()
        win_w = self.window.frame().size.width
        # Horizontally centered on the chosen screen, TOP_MARGIN from the top.
        x = sf.origin.x + (sf.size.width - win_w) / 2
        y = sf.origin.y + sf.size.height - HEIGHT - TOP_MARGIN
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
