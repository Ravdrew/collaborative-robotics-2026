# Remote GPU Training over Tailscale

Use the GPU computer remotely from the robot. All via SSH over Tailscale. **All commands run from the robot.**

## Network Layout

| Machine | Role | Tailscale IP |
|---------|------|-------------|
| TidyBot2 NUC (robot) | Collects data, launches jobs, pulls results | `100.106.67.118` |
| GPU workstation | Runs training/inference (started remotely) | `100.77.113.90` |


No ROS2 needed on the GPU computer. Just SSH + file transfer.

---

## One-Time Setup (GPU computer)

Do these once, then never touch the machine again.

### 1. Install Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# Authenticate with the SAME Tailscale account as the robot
```

### 2. Install Python + CUDA

```bash
# PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision numpy

# Verify GPU works
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

Install any ML libraries your team needs (SAM, etc.):
```bash
pip install sam2  # or whatever model you're using
```

### 3. Install tmux

```bash
sudo apt install -y tmux
```

### 4. Create workspace directory

```bash
mkdir -p ~/gpu_workspace
```

---

## One-Time Setup (Robot)

### 1. Install Tailscale (if not already done)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

### 2. Set up passwordless SSH

```bash
ssh-keygen -t ed25519           # Enter, Enter, Enter (no passphrase)
ssh-copy-id giuse@100.77.113.90 # Enter GPU computer password once
```

Verify:
```bash
ssh giuse@100.77.113.90 "echo OK"   # should print OK, no password prompt
```


## Troubleshooting

### Cannot reach GPU computer
```bash
tailscale status              # both machines should appear
tailscale ping 100.77.113.90  # should succeed
```

### SSH fails
```bash
ssh giuse@100.77.113.90 "echo OK"
# If prompted for password, run: ssh-copy-id giuse@100.77.113.90
```
