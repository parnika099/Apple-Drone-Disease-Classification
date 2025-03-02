import sys
import os

print("Verifying imports...")

def check_import(module_name, display_name):
    try:
        __import__(module_name)
        print(f"✅ {display_name} imported")
        return True
    except ImportError as e:
        print(f"❌ {display_name} failed: {e}")
        return False

success = True
success &= check_import("src.dashboard", "src.dashboard")
success &= check_import("src.detection", "src.detection")
success &= check_import("src.quality_prediction", "src.quality_prediction")
success &= check_import("src.gemini_integration", "src.gemini_integration")
success &= check_import("src.chatbot", "src.chatbot")
success &= check_import("src.batch_analyzer", "src.batch_analyzer")
success &= check_import("src.soil_service", "src.soil_service")

if success:
    print("Verification complete: SUCCESS")
else:
    print("Verification complete: FAILED")
    sys.exit(1)
