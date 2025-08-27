import sys
import os
from pathlib import Path
import pickle
import sklearn

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.config.config import get_model_path, get_model_info, MODELS_DIR, validate_environment
    from src.services.prediction_service import PredictionService
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def check_sklearn_version():
    print(f"Scikit-learn version: {sklearn.__version__}")
    
    version = sklearn.__version__
    major, minor = map(int, version.split('.')[:2])
    
    if major == 1 and minor >= 4:
        print("Warning: Using scikit-learn >= 1.4, might have compatibility issues")
        print("   Recommended: scikit-learn==1.3.2")
    else:
        print("Scikit-learn version looks compatible")

def check_directories():
    print(f"\nChecking directories:")
    print(f"   Models directory: {MODELS_DIR}")
    print(f"   Exists: {MODELS_DIR.exists()}")
    
    if MODELS_DIR.exists():
        files = list(MODELS_DIR.iterdir())
        print(f"   Files in models dir: {len(files)}")
        for file in files:
            print(f"     - {file.name} ({file.stat().st_size} bytes)")
    
    return validate_environment()

def check_model_files():
    print(f"\nChecking model files:")
    
    model_info = get_model_info()
    print(f"   Models directory: {model_info['models_dir']}")
    
    for model_name, info in model_info['available_models'].items():
        status = "EXISTS" if info['exists'] else "MISSING"
        size = f" ({info['size']} bytes)" if info['exists'] else ""
        print(f"   {model_name}: {status}{size}")
        print(f"     Path: {info['path']}")
    
    if model_info['missing_models']:
        print(f"\nMissing models: {', '.join(model_info['missing_models'])}")
        return False
    
    return True

def test_model_loading():
    print(f"\nTesting model loading:")
    
    try:
        service = PredictionService()
        
        print("   Attempting to load models...")
        success = service.load_models()
        
        if success:
            print("   Models loaded successfully")
            
            info = service.get_model_info()
            print(f"   Model type: {info.get('model_type', 'Unknown')}")
            if 'available_crops' in info:
                print(f"   Available crops: {len(info['available_crops'])}")
                print(f"     {', '.join(info['available_crops'][:5])}{'...' if len(info['available_crops']) > 5 else ''}")
            
            return True
        else:
            print("   Failed to load models")
            return False
            
    except Exception as e:
        print(f"   Error loading models: {e}")
        return False

def test_prediction():
    print(f"\nTesting prediction:")
    
    try:
        service = PredictionService()
        
        test_params = {
            "ph": 6.5,
                    "humidity": 50,
        "temperature": 20,
        "precipitation": 150,
        "sun_hours": 8.0,
        "soil_type": "arcilloso",
        "season": "verano"
        }
        
        print(f"   Test parameters: {test_params}")
        
        success, result, error = service.predict_crop(test_params)
        
        if success:
            print(f"   Prediction successful: {result}")
            return True
        else:
            print(f"   Prediction failed: {error}")
            return False
            
    except Exception as e:
        print(f"   Error during prediction test: {e}")
        return False

def suggest_fixes():
    print(f"\nSuggested fixes:")
    print("   1. Re-train models with compatible scikit-learn version:")
    print("      pip install scikit-learn==1.3.2")
    print("      python src/models/train_model.py")
    print()
    print("   2. Check file permissions and paths")
    print()
    print("   3. Verify Firebase configuration if using cloud deployment")

def main():
    print("AgroTech Verde - Model Diagnostics")
    print("=" * 50)
    
    checks = [
        ("Scikit-learn version", check_sklearn_version),
        ("Directories", check_directories),
        ("Model files", check_model_files),
        ("Model loading", test_model_loading),
        ("Prediction", test_prediction)
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
        except Exception as e:
            print(f"Error during {name} check: {e}")
            results[name] = False
    
    print(f"\nSummary:")
    print("=" * 30)
    
    passed = sum(1 for r in results.values() if r is True)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"   {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed < total:
        suggest_fixes()
    else:
        print("\nAll checks passed! Your models should work correctly.")

if __name__ == "__main__":
    main() 