import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name} loaded successfully.")
    except ImportError as e:
        print(f"❌ FAILED to load {module_name}: {e}")
    except Exception as e:
        print(f"⚠️ Error loading {module_name}: {e}")

print(f"--- Checking Python {sys.version.split()[0]} Dependencies ---")

# Core Data & App
check_import('streamlit')
check_import('pandas')
check_import('ollama')
check_import('sqlalchemy')
check_import('plotly')

# Audio & ML
check_import('torch')
check_import('librosa')
check_import('pydub')

# Translation
check_import('langdetect')
check_import('googletrans')

# PDF/Image
check_import('reportlab')
check_import('PIL') # Imports as PIL but installed as Pillow

# WhisperX (The heavy lifter)
try:
    import whisperx
    print(f"✅ whisperx loaded successfully.")
except ImportError:
    print("❌ FAILED to load whisperx. Try: pip install git+https://github.com/m-bain/whisperx.git")
except Exception as e:
    print(f"⚠️ Error loading whisperx (Ensure FFmpeg is installed): {e}")

print("\n--- System Check ---")
# Check FFmpeg
import shutil
if shutil.which("ffmpeg"):
    print("✅ FFmpeg is found in system PATH.")
else:
    print("❌ FFmpeg NOT found. Audio processing will fail.")