# Whisper Hotkey

Push-to-talk voice transcription daemon for macOS and Linux. Hotkey toggles recording, transcribes locally via Parakeet TDT or faster-whisper, pastes text into active window.

## Architecture

- `transcribe-daemon.py` — Main daemon. Loads model once, stays resident. Signal-based IPC (SIGUSR1/SIGUSR2) with client script.
- `scripts/hotkey-wrapper.sh` — Client script bound to hotkey. Sends signals to daemon, starts daemon on first press.
- `scripts/setup-hotkey.sh` — Configures hotkey binding. macOS: installs skhd + writes `~/.config/skhd/skhdrc`. Linux: i3/sway/GNOME/KDE.
- `whisper-status.py` — GTK3 floating status overlay. Polls `whisper-hotkey-status.json`, shows recording/transcribing/idle state. Launched automatically by daemon when `ENABLE_OVERLAY=true`.
- `scripts/whisper.sh` — Legacy standalone transcription script.
- `install.sh` — One-command installer (deps, venv, engine, GPU, hotkey, model download).

## Key Design Decisions

- **State file toggle**: `whisper-hotkey-state` in runtime dir (`$XDG_RUNTIME_DIR` on Linux, `$TMPDIR` on macOS) exists = recording. Hotkey wrapper creates/removes it.
- **Auto-start on first press**: Daemon creates state file before model load, plays wind chime during load, auto-starts recording when ready. Clears queued signals after load to avoid race conditions.
- **Lazy daemon**: Starts on first hotkey press, optionally unloads after IDLE_TIMEOUT. Not a systemd service by default.
- **Instant paste**: Uses CGEvent Cmd+V (macOS), `xdotool type --delay 0` (X11), or `ydotool key Ctrl+V` (Wayland).
- **Shift-to-submit**: Detects Shift at paste time via Quartz/CGEvent (macOS) or X11 XQueryKeymap (Linux X11); presses Enter after paste. Wayland not supported.
- **Status overlay**: GTK3 widget spawned as a child process of the daemon. Reads `whisper-hotkey-status.json` (200ms poll). Auto-hides after 3s idle. Draggable. Killed when daemon exits.

## Engines

- **Parakeet TDT** (default): `nemo-parakeet-tdt-0.6b-v3` via onnx-asr. ~0.3s transcription. 25 languages.
- **faster-whisper**: CTranslate2-based. Configurable model sizes. ~0.5s on GPU. Multilingual.

## Config

All config in `.env` (loaded via python-dotenv). See `.env.example` for all options.

## Testing

- Kill daemon: `pkill -f transcribe-daemon`
- Check logs: `tail -f "${XDG_RUNTIME_DIR:-$TMPDIR}/whisper-hotkey.log"`
- Test recording (Linux): `parecord --format=s16le --rate=16000 --channels=1 test.wav`
- Test recording (macOS): use the daemon directly; sounddevice/PortAudio handles input
- Check daemon status: `ps aux | grep transcribe-daemon`

## Known Limitations

- Shift-to-submit only works on X11 (Wayland blocks global key reads)
- Wayland paste uses ydotool Ctrl+V (clipboard-based) to avoid ydotool type Unicode bugs
- onnxruntime pinned to avoid version 1.21 (HF symlink bug)
- ctranslate2 pinned to 4.4.0 for Ubuntu 24.04 cuDNN 8 compatibility

## Dependencies

Linux system: pulseaudio-utils, xclip/wl-clipboard, xdotool/ydotool, libnotify-bin, python3-gi (GTK3 for overlay)
macOS system: skhd (brew), `pbcopy`/`osascript` built-in, sounddevice via PortAudio (`brew install portaudio`)
Python: see requirements.txt + onnx-asr[hub] for parakeet engine
