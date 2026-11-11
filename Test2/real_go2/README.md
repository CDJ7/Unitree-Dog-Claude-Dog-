# Q1 on the real Unitree Go2: zone recognition in your own room

The simulated Q1 taught a ResNet18 the five zones of the MuJoCo house. Those zones only exist in
simulation, so the same method is applied to **your real room**:

1. Define a few zones.
2. Drive the real Go2 around while recording labelled camera frames.
3. Retrain with the unchanged Q1 training code.
4. Run the classifier live on the dog's camera and record the video.

> **Status:** everything here was tested end to end with a *fake* camera source (replayed frames) on a
> machine without the robot: recording, training, evaluation, live classification and the MP4. The
> connections to the real dog (`--source sdk` and `--source webrtc`) have **not yet been run against the
> robot**. The first session with the Go2 is the real test; report any error and it will be fixed.

| Script | Step |
|---|---|
| [`record_real.py`](record_real.py) | Drive the dog and record labelled frames → `data_real/run_XXX/` |
| `q1_supervised/train.py --config real_go2/real_config.yaml` | Retrain ResNet18 on your room (same recipe as Q1) |
| [`evaluate_real.py`](evaluate_real.py) | Test accuracy on the held-out last session, per-zone scores, confusion matrix |
| [`live_zone.py`](live_zone.py) | Live recognition on the dog's camera, with an on-screen panel, recorded to MP4 |
| [`sources.py`](sources.py) | The connections: `sdk` (Ethernet), `webrtc` (Wi-Fi), `fake` (no robot) |
| [`real_config.yaml`](real_config.yaml) | Your zone names, data folder, training settings |

Driving reuses Test 1's keyboard teleop ([`go2_stream/teleop.py`](../../go2_stream/teleop.py)), with its
speed limits and watchdog.

---

## 1. One-time setup (Windows + WSL2)

### 1.1 Turn on WSL "mirrored" networking (needed for Ethernet; recommended for Wi-Fi too)
With WSL2's default networking, Linux sits behind a virtual network, and the Unitree SDK's device
discovery (DDS multicast) cannot reach the dog. Mirrored mode (Windows 11 22H2 or later) gives WSL the
real network adapters.

1. In Windows, create or edit `C:\Users\<you>\.wslconfig`:
   ```
   [wsl2]
   networkingMode=mirrored
   ```
2. In PowerShell, run `wsl --shutdown`, then reopen VS Code / WSL.
3. Check in WSL with `ip -br addr`: you should now see the same adapters and IPs as Windows.

### 1.2 Python packages (inside WSL)
Use the same environment as Test 1 (it already has `unitree_sdk2py`, `pynput`, `opencv`), plus Test 2's:
```bash
cd ~/Unitree-Dog-Claude-Dog-/Test2
pip install -r requirements.txt            # torch, torchvision, scikit-learn, ...
python -m pytest -q tests                  # 14 tests, no robot needed
```
For Wi-Fi only, also install:
```bash
sudo apt install -y portaudio19-dev        # the WebRTC library pulls in pyaudio
pip install unitree_webrtc_connect
```

### 1.3 Try everything without the dog first
```bash
python real_go2/record_real.py --source fake --headless --max-frames 200   # makes data_real/run_000
```
This writes into `data_real/`. Delete the test runs afterwards (`rm -r data_real`) before recording real
sessions.

### 1.4a Ethernet connection (`--source sdk`) — recommended
1. Plug the laptop into the Go2 EDU's Ethernet port.
2. In **Windows** network settings, give that Ethernet adapter a static IPv4 address: **192.168.123.222**,
   mask **255.255.255.0**, no gateway. With mirrored networking, WSL inherits it; this replaces Test 1's
   `setup_network.sh`.
3. In WSL, find the interface name and check the dog answers:
   ```bash
   ip -br addr | grep 192.168.123          # the first word is the interface name, e.g. eth1
   ping -c 3 192.168.123.161
   ```
4. Pass that name as `--iface` (or set it as `network.iface` in `go2_stream/config.yaml`).

### 1.4b Wi-Fi connection (`--source webrtc`)
1. In the Unitree app, connect the dog to your Wi-Fi (STA mode) and note its IP address. Your laptop must
   be on the same network.
2. Check the dog's **firmware version** in the app. On **1.1.15 or newer**, the Wi-Fi handshake needs the
   dog's personal AES key. Fetch it once with your Unitree account, then store it:
   ```bash
   unitree-fetch-aes-key --email you@example.com --device-type Go2   # prompts for the password
   export GO2_AES_KEY=<the 32-character key>                         # or pass --aes-key
   ```
3. **Close the Unitree app** on your phone. The dog accepts only one WebRTC connection at a time.
4. Use `--source webrtc --ip <dog IP>`. If your laptop is on the dog's own hotspot instead, use
   `--source webrtc --ap`.

---

## 2. Choose your zones
Pick **3–5 areas** that look different from each other, for example near the door, the desk, the couch
and the window. Put the names in [`real_config.yaml`](real_config.yaml) under `classes`. Keys 1, 2, 3, …
select them in this order.

Areas that look alike (two bare walls) will be confused, exactly as in the simulation.

## 3. Record (about 15 minutes)
```bash
python real_go2/record_real.py --source sdk --iface eth1          # Ethernet
python real_go2/record_real.py --source webrtc --ip 192.168.1.50  # Wi-Fi
```

**Driving** (keys work anywhere):

| Key | Action |
|---|---|
| R | stand up |
| W / S | forward / back |
| A / D | step sideways |
| Q / E | turn |
| SPACE | stop |
| F | lie down |
| ESC | quit |

The speed limit is `--max-speed`, 0.3 m/s by default.

**Labelling** (click the recorder window first):

| Key | Action |
|---|---|
| 1..N | "the dog is now in this zone" |
| 0 | no zone; frames are not saved (use it while crossing between zones) |
| P | pause / resume |

The window shows the current zone, the number of frames saved per zone, and an orange box marking the
part of the image that is saved.

**Protocol.** Record **4–5 separate sessions**: quit with ESC and start again. Each session becomes one
`run_XXX`. The **last session is the test set** and the **second-to-last is validation**, so the test
measures how well the model recognises the room on a new walk.
- **Coverage:** in each session, visit every zone for about 30–60 s.
  - Approach from different directions and distances.
  - Turn on the spot (Q/E) so the camera sees each zone from all sides.
- **Variety:** change something between sessions, such as the time of day, lights on or off, or a moved
  chair. Then the test checks generalisation, as the unseen scene did in simulation.
- **Labels:** press 0 while walking between zones, so boundary frames are not mislabelled.
- **Balance:** aim for roughly equal frames per zone. The counts are on screen.

## 4. Train and evaluate
```bash
python q1_supervised/train.py --config real_go2/real_config.yaml     # about 10–20 min on CPU for ~3,000 frames
python real_go2/evaluate_real.py                                      # test accuracy on the last session
```
Results go to `results/real_q1/`:
- `metrics.json`: accuracy, macro-F1, per-zone precision/recall/F1, confusion counts;
- `confusion_resnet18.png` and `curves_resnet18.png`;
- `resnet18/best.pt`.

Training can also run on Colab: open `q1_supervised/Q1_colab.ipynb`, upload `data_real/` and
`real_go2/real_config.yaml`, then run
`!Q1_CONFIG=real_go2/real_config.yaml python q1_supervised/train.py`.

## 5. Live demo and filming
```bash
python real_go2/live_zone.py --source sdk --iface eth1      # or --source webrtc --ip ...
```

**What the window shows:**
- the dog's camera;
- the **predicted zone** and the confidence for every zone;
- a **1-second majority vote**;
- the **true zone** you key in, and the **live accuracy**.

**Keys:**

| Key | Action |
|---|---|
| V | start / stop recording the MP4 |
| 1..N | the true zone right now |
| 0 | true zone unknown |
| ESC | quit |

Driving keys are the same as in the recorder. Output, in `results/real_q1/`:
- `live_<time>.mp4`: 10 fps, real time;
- `live_<time>.csv`: every prediction;
- `live_<time>.json`: summary with the live accuracy.

**Filming checklist:**
1. Fully charged dog, a clear area, and someone filming the dog with a phone. Mount the phone or hold it
   steady, and keep the dog and the laptop screen in view, or film them separately.
2. Start `live_zone.py`, press **R** (stand), then **V** (record).
3. Walk a tour through every zone at a slow speed. In each zone, turn once on the spot.
4. Keep pressing the number of the zone you are in, so the video shows **true vs predicted** and the
   running accuracy.
5. Also walk one zone the model was *not* trained on, if there is one, to show the honest failure case.
6. ESC, then use the MP4 (screen view) next to the phone footage (the dog) in your video.

**What to report** to your professor:
- the test accuracy and per-zone table from `metrics.json`;
- the confusion matrix;
- the live accuracy from `live_<time>.json`;
- the comparison with the simulated Q1 (0.882 test accuracy).

## Safety
- Use flat floor and open space. Keep people and pets out of the dog's path. Start with
  `--max-speed 0.2`.
- **SPACE** stops immediately. **ESC** stops and quits, and always sends StopMove. Test 1's watchdog
  stops the dog if the control loop stalls.
- Keep the Go2 remote or app within reach as a backup. With Wi-Fi the app must be closed, so use the
  remote.
- Do not use X then Y (damp) while the dog is standing: it collapses.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No network interface 'enp0s31f6'` | Pass `--iface` with the name from `ip -br addr \| grep 192.168.123`. |
| SDK: "waiting for the camera", no state messages | Is mirrored networking on (`ip -br addr` shows 192.168.123.222)? Does `ping 192.168.123.161` answer? Is the Windows firewall blocking WSL's inbound UDP? In an admin PowerShell: `Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow`, then `wsl --shutdown`. |
| Wi-Fi: `AesKeyRequiredError` / `AesKeyRejectedError` | Firmware ≥ 1.1.15: fetch or check the key (section 1.4b). |
| Wi-Fi: `RobotBusyError` | The Unitree app (or another script) is connected. Close it. |
| Wi-Fi: `LocalSignalingPortError` | Wrong IP, or laptop and dog are not on the same network. |
| Keys do not drive the dog | `pynput` needs an X display (WSLg provides one). Press R first: the dog must be standing. |
| The model confuses two zones | They look alike from the dog's height. Pick more distinct zones, or record more varied sessions. |
