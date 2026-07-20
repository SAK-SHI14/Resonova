import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def main():
    print("=" * 60)
    print("RESONOVA — WAV2LIP AUTOMATIC SETUP & COMPATIBILITY PATCHER")
    print("=" * 60)

    project_root = Path(__file__).parent.resolve()
    wav2lip_dir = project_root / "Wav2Lip"

    # 1. Clone Wav2Lip if missing
    if not wav2lip_dir.exists():
        print("[1/3] Cloning Wav2Lip repository...")
        try:
            subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", str(wav2lip_dir)], check=True)
            print("  -> Cloned successfully.")
        except Exception as e:
            print(f"❌ ERROR: Failed to clone Wav2Lip repository: {e}")
            sys.exit(1)
    else:
        print("[1/3] Wav2Lip repository already exists. Skipping clone.")

    # 2. Download Wav2Lip GAN checkpoint if missing
    ckpt_dir = wav2lip_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "wav2lip_gan.pth"

    if not ckpt_path.exists():
        print("[2/3] Downloading Wav2Lip GAN checkpoint (~400 MB) from Hugging Face mirror...")
        url = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            )
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                block_size = 1024 * 1024  # 1MB chunks
                with open(ckpt_path, 'wb') as out_file:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        if total_size > 0:
                            percent = min(100.0, (downloaded / total_size) * 100.0)
                            print(f"\r  -> Downloading: {percent:.1f}% ({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)", end="")
                        else:
                            print(f"\r  -> Downloading: {downloaded / 1e6:.1f} MB", end="")
            print("\n  -> Download completed successfully.")
        except Exception as e:
            # Clean up partial file on failure
            if ckpt_path.exists():
                try:
                    ckpt_path.unlink()
                except Exception:
                    pass
            print(f"\n❌ ERROR: Failed to download checkpoint: {e}")
            sys.exit(1)
    else:
        print("[2/3] Wav2Lip GAN checkpoint already exists. Skipping download.")

    # 3. Patch Wav2Lip for NumPy 1.24+ / NumPy 2.x compatibility
    print("[3/3] Patching Wav2Lip code for compatibility with modern NumPy...")
    replacements = [
        ('np.bool,', 'bool,'), ('np.bool)', 'bool)'), ('np.bool ', 'bool '),
        ('np.int,', 'int,'), ('np.int)', 'int)'), ('np.int ', 'int '),
        ('np.float,', 'float,'), ('np.float)', 'float)'), ('np.float ', 'float '),
        ('np.complex,', 'complex,'), ('np.complex)', 'complex)'), ('np.complex ', 'complex '),
        ('np.object,', 'object,'), ('np.object)', 'object)'), ('np.object ', 'object '),
        ('np.str,', 'str,'), ('np.str)', 'str)'), ('np.str ', 'str '),
        ('librosa.filters.mel(hp.sample_rate, hp.n_fft,', 'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,'),
        ("command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')", "command = 'ffmpeg -y -i \"{}\" -strict -2 \"{}\"'.format(args.audio, 'temp/temp.wav')"),
        ("command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)", "command = 'ffmpeg -y -i \"{}\" -i \"{}\" -strict -2 -q:v 1 \"{}\"'.format(args.audio, 'temp/result.avi', args.outfile)"),
    ]

    patched_count = 0
    for root, _, files in os.walk(str(wav2lip_dir)):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    patched = content
                    for old, new in replacements:
                        patched = patched.replace(old, new)
                    if patched != content:
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(patched)
                        patched_count += 1
                except Exception as e:
                    print(f"  [WARN] Could not patch file {fname}: {e}")

    print(f"  -> Patched {patched_count} source files in Wav2Lip for modern NumPy compatibility.")

    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("To run the application with Lip-Sync enabled, execute:")
    print("")
    print("PowerShell:")
    print(f'  $env:WAV2LIP_REPO_PATH = "{wav2lip_dir}"')
    print(f'  $env:WAV2LIP_CHECKPOINT_PATH = "{ckpt_path}"')
    print("  python -m resonova.app.launch")
    print("=" * 60)

if __name__ == "__main__":
    main()
