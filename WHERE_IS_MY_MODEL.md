# 🎯 Where to Find Your Trained Edo TTS Model

## 📍 **Model Location After Training**

Your trained model will be saved in the `outputs/` directory:

```
edo-tts-model/
├── outputs/                    ← YOUR MODEL IS HERE
│   ├── best_model.pth         ← Main trained model (MOST IMPORTANT)
│   ├── config.json            ← Model configuration (REQUIRED)
│   ├── scale_stats.npy        ← Audio normalization data (REQUIRED)
│   ├── speakers.pth           ← Speaker info (if multi-speaker)
│   ├── events.out.tfevents.*  ← TensorBoard training logs
│   └── checkpoints/           ← Training snapshots
│       ├── checkpoint_10000.pth
│       ├── checkpoint_20000.pth
│       └── ...
```

## 📦 **Essential Files for Your Model**

**You need these 3 files to use your model:**

1. **`outputs/best_model.pth`** - The actual trained neural network
2. **`outputs/config.json`** - Configuration settings
3. **`outputs/scale_stats.npy`** - Audio preprocessing statistics

## 🔍 **How to Check if Training Completed**

Run this command to see if your model is ready:

```bash
ls -la outputs/
```

**You should see:**
- `best_model.pth` (several MB in size)
- `config.json` (small text file)
- Training logs and checkpoints

## 💾 **How to Download/Backup Your Model**

### **Method 1: Direct File Copy (Local Training)**
```bash
# Copy essential files to a backup folder
mkdir edo_model_backup
cp outputs/best_model.pth edo_model_backup/
cp outputs/config.json edo_model_backup/
cp outputs/scale_stats.npy edo_model_backup/
```

### **Method 2: Create Archive**
```bash
# Create a compressed archive of your model
tar -czf edo_tts_model.tar.gz outputs/
```

### **Method 3: Upload to Cloud (Optional)**
```bash
# Upload to Google Drive, Dropbox, etc.
# Or push to GitHub (if model file is under 100MB)
```

## 🧪 **Test Your Model**

After training completes, run:

```bash
python test_model.py
```

This will:
- ✅ Check if all model files exist
- 🎵 Generate test audio files
- 📊 Show model information
- 🚀 Create API server script

## 🎵 **Quick Model Usage**

Once you have the model files:

```python
from TTS.api import TTS

# Load your trained Edo model
tts = TTS(
    model_path="outputs/best_model.pth",
    config_path="outputs/config.json"
)

# Generate Edo speech
tts.tts_to_file(
    text="ẹ̀dó",  # Edo text
    file_path="my_edo_speech.wav"
)
```

## 🔄 **If Training is Still Running**

**Check progress:**
```bash
# View training logs
tail -f outputs/train.log

# Or use TensorBoard
tensorboard --logdir outputs/
```

**Estimated training time:** 6-24 hours (depending on your hardware)

## ⚠️ **Common Issues**

**Problem:** No `outputs/` directory
- **Solution:** Training hasn't started or failed to begin

**Problem:** Empty `outputs/` directory  
- **Solution:** Training is still in progress

**Problem:** Only checkpoint files, no `best_model.pth`
- **Solution:** Training incomplete - wait longer or check logs

**Problem:** Model file very small (< 1MB)
- **Solution:** Training failed - check `outputs/train.log`

## 📤 **Sharing Your Model**

To share your trained Edo TTS model:

1. **Archive the essentials:**
   ```bash
   ./create_package.sh  # Run this after test_model.py
   ```

2. **Upload the package:**
   - `edo_tts_model_package.tar.gz` (contains everything needed)

3. **Recipients can use it with:**
   ```bash
   tar -xzf edo_tts_model_package.tar.gz
   cd edo_tts_package
   python example.py
   ```

---

**🎉 Your model will be in the `outputs/` folder when training completes!**
