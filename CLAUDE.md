# Whisper Hotkey

Push-to-talk voice transcription daemon for macOS and Linux. Hotkey toggles recording, transcribes locally (Apple SFSpeech streaming, Parakeet TDT or faster-whisper), pastes text into active window.

## Architecture

- `transcribe-daemon.py` — Main daemon. Loads model once, stays resident. Signal-based IPC (SIGUSR2 start, SIGUSR1 stop) with client script.
- `scripts/hotkey-wrapper.sh` — Client script bound to hotkey. Signals the daemon (after checking the lock-file PID really is `transcribe-daemon.py`), starts it on first press.
- `scripts/setup-hotkey.sh` — Configures hotkey binding. macOS: installs skhd + writes `~/.config/skhd/skhdrc`. Linux: i3/sway/GNOME/KDE.
- `whisper-status-macos.py` (AppKit, venv Python) / `whisper-status.py` (GTK3, system `python3`) — Status overlay pill. Polls `whisper-hotkey-status.json`. Spawned by the daemon when `ENABLE_OVERLAY=true`.
- `install.sh` — One-command installer (deps, venv, engine, GPU, hotkey, model download).

## Key Design Decisions

- **State file toggle**: `whisper-hotkey-state` in runtime dir (`$XDG_RUNTIME_DIR` on Linux, `$TMPDIR` on macOS) exists = recording. Hotkey wrapper creates/removes it; the daemon removes it when a start fails or max duration auto-stops.
- **Auto-start on first press**: Daemon creates state file before model load, plays wind chime during load, auto-starts recording when ready. Clears queued signals after load to avoid race conditions.
- **Lazy daemon**: Starts on first hotkey press, optionally unloads after IDLE_TIMEOUT. Not a systemd service by default.
- **Main loop sleeps on signals**: `signal.set_wakeup_fd` + a CFRunLoop source (macOS) / `select` (Linux). On macOS the main CFRunLoop must keep running, or SFSpeech callbacks never fire.
- **Instant paste**: Uses CGEvent Cmd+V (macOS, clipboard written via NSPasteboard), `xdotool type --delay 0` (X11), or `ydotool key Shift+Insert` with text in both CLIPBOARD and PRIMARY (Wayland — works in terminals and GUI apps alike).
- **Shift-to-submit**: Detects Shift at paste time via Quartz/CGEvent (macOS), X11 XQueryKeymap (Linux X11), or evdev EVIOCGKEY on `/dev/input/by-{id,path}/*-event-kbd` (Wayland; needs read access, e.g. a udev `TAG+="uaccess"` rule for keyboards); Shift counts if held at any point from the stop hotkey until paste (watcher thread), and GNOME gets a second `<Shift><Super>space` binding since `<Super>space` doesn't fire with Shift held; presses Enter after paste (0.6 s later on Wayland, `SUBMIT_DELAY`).
- **Wayland focus wait**: Something around the stop hotkey can briefly take focus off the target window (mostly on the first dictation after a pause), losing the paste. A system-python AT-SPI helper is spawned at stop and, just before the paste key, waits (≤1.5 s) until an app window is ACTIVE; result logged as `Focus before paste:`.
- **Status overlay**: child process of the daemon, 200ms poll, auto-hides 3s after idle/error, exits when the daemon dies; respawned unless it crashes on launch 3 times in a row.

## Engines

- **apple-streaming** (macOS): SFSpeechRecognizer on-device, live transcript in the overlay. Apple finalizes a task on ~1-2s silence, so the daemon chains tasks and dedupes Apple's re-transcriptions (`_sfs_handle`).
- **Parakeet TDT** (default): `nemo-parakeet-tdt-0.6b-v3` via onnx-asr. ~0.3s transcription. 25 languages.
- **faster-whisper**: CTranslate2-based. Configurable model sizes. ~0.5s on GPU. Multilingual. No Apple GPU backend (`DEVICE=mps` fails and falls back to CPU).

## Config

All config in `.env` (loaded via python-dotenv). See `.env.example` for all options.

## Testing

- Importing the daemon module writes its log and status files to `$TMPDIR`; point `TMPDIR` at a temp dir first or a test pops the live overlay.
- `venv/bin/python test_sfs_handle.py` — streaming handler logic with fake results.
- `venv/bin/python test_sfs_live.py` — real SFSpeechRecognizer fed from `say` audio (no mic/paste); needs Speech Recognition granted to the terminal.
- Kill daemon: `pkill -f transcribe-daemon` (the next hotkey press cold-starts it)
- Check logs: `tail -f "${XDG_RUNTIME_DIR:-$TMPDIR}/whisper-hotkey.log"`
- macOS permissions (Mic, Speech Recognition, Accessibility for paste) are granted to skhd, the process that launches the daemon.

## Known Limitations

- Shift-to-submit on Wayland needs read access to the keyboard evdev nodes (udev uaccess rule or `input` group); without it Shift reads as not held
- Wayland paste uses ydotool Shift+Insert (clipboard + primary based) to avoid ydotool type Unicode bugs
- onnxruntime pinned to avoid version 1.21 (HF symlink bug)
- ctranslate2 pinned to 4.4.0 for Ubuntu 24.04 cuDNN 8 compatibility

## Dependencies

Linux system: pulseaudio-utils, xclip/wl-clipboard, xdotool/ydotool, libnotify-bin, python3-gi (GTK3 for overlay)
macOS system: skhd (brew); recording, paste and speech go through pyobjc (see requirements.txt)
Python: see requirements.txt + onnx-asr[hub] for parakeet engine
