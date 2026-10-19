"""Shared configuration and prompt mappings for the platform."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TABULAR_DATA_DIR = DATA_DIR / "tabular"
VISION_DATA_DIR = DATA_DIR / "vision"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DATASET_HANDLES = {
    "vision": "emmarex/plantdisease",
    "tabular": "atharvaingle/crop-recommendation-dataset",
}

# 10+ prompt templates mapped to plant disease classes
DISEASE_PROMPTS = {
    "Healthy": [
        "a clear photo of a healthy tomato leaf",
        "a photo of a healthy crop leaf with no visible disease",
    ],
    "Fungal_Blight": [
        "a photo of a tomato leaf showing late blight lesions",
        "a close-up of a tomato leaf with early blight fungal spots",
        "a diseased potato leaf with fungal blight lesions",
    ],
    "Bacterial_Spot": [
        "a photo of a tomato leaf with bacterial spot lesions",
        "a crop leaf showing small dark bacterial disease spots",
    ],
    "Leaf_Mold": [
        "a photo of a tomato leaf infected by leaf mold",
        "a tomato leaf with pale yellow patches and mold underneath",
    ],
    "Mosaic_Virus": [
        "a photo of a leaf with mosaic virus mottling",
        "a diseased crop leaf showing yellow-green mosaic patterns",
    ],
    "Powdery_Mildew": [
        "a photo of a leaf covered with powdery mildew",
        "a crop leaf with white powder-like fungal growth",
    ],
}

DISEASE_REMEDIES = {
    "Healthy": [
        "Continue regular scouting and maintain standard crop monitoring.",
        "Avoid unnecessary chemical treatments to preserve natural leaf microflora."
    ],
    "Fungal_Blight": [
        "Remove and bag affected leaves immediately; do not compost diseased material.",
        "Water strictly at soil level early in the morning and thin canopies to improve airflow.",
        "Apply organic copper-based or bio-fungicide sprays early before sporulation spreads."
    ],
    "Bacterial_Spot": [
        "Avoid handling or pruning wet plants; sanitize pruning shears with 70% alcohol.",
        "Remove severely affected foliage and apply certified organic copper bactericide.",
        "Switch to drip irrigation and rotate crops with non-solanaceous varieties."
    ],
    "Leaf_Mold": [
        "Increase greenhouse or field spacing and ventilation; eliminate stagnant moisture.",
        "Avoid overhead irrigation and wipe support stakes with sanitizing solution.",
        "Prune bottom suckers and lower canopy leaves to reduce humidity pockets."
    ],
    "Mosaic_Virus": [
        "Immediately isolate and rouge heavily affected plants to curb viral vectors.",
        "Control aphid and whitefly vectors using neem oil or insecticidal potassium soaps.",
        "Disinfect hands and tools after contact, as viral particles transmit mechanically."
    ],
    "Powdery_Mildew": [
        "Prune congested center branches to ensure direct sunlight and circulating air.",
        "Apply organic bio-fungicides, diluted potassium bicarbonate, or neem oil extract.",
        "Avoid high-nitrogen fertilizing during active outbreaks to avoid tender leafy growth."
    ],
    "Biological_Anomaly": [
        "Recheck the photo and inspect ambient humidity sensors before field intervention.",
        "Fungal sporulation requires sustained high relative humidity; verify microclimate conditions.",
        "Consult local agricultural university extension agronomists before spraying."
    ],
}

MULTILINGUAL_DISEASE_DATA = {
    "Healthy": {
        "en": {
            "label": "Healthy Leaf Foliage",
            "remedy": "Foliage shows optimal chlorophyll density with no pathogen lesions. Maintain standard crop monitoring and avoid unnecessary spraying to preserve natural microflora."
        },
        "hi": {
            "label": "स्वस्थ पत्तियां",
            "remedy": "पत्ती में कोई रोग के लक्षण नहीं हैं। नियमित निगरानी जारी रखें और अनावश्यक कीटनाशक छिड़काव से बचें।"
        },
        "ta": {
            "label": "ஆரோக்கியமான இலை",
            "remedy": "இலையில் நோய் அறிகுறிகள் எதுவும் இல்லை. வழக்கமான களக் கண்காணிப்பைத் தொடருங்கள் மற்றும் தேவையற்ற தெளிப்புகளைத் தவிர்க்கவும்."
        }
    },
    "Fungal_Blight": {
        "en": {
            "label": "Fungal Blight (Phytophthora/Alternaria)",
            "remedy": "Remove and bag infected leaves immediately; do not compost diseased material. Apply organic copper fungicide or bio-agent sprays. Thin lower canopy to improve airflow."
        },
        "hi": {
            "label": "फंगल झुलसा रोग (ब्लाइट)",
            "remedy": "संक्रमित पत्तियों को तुरंत हटाकर नष्ट करें। जैविक कॉपर कवकनाशी का छिड़काव करें और हवा का संचार बढ़ाने के लिए निचली पत्तियों की छंटाई करें।"
        },
        "ta": {
            "label": "பூஞ்சை கருகல் நோய்",
            "remedy": "பாதிக்கப்பட்ட இலைகளை உடனடியாக அகற்றி எரிக்கவும். கரிம செம்பு பூஞ்சைக் கொல்லியை தெளிக்கவும் மற்றும் காற்றோட்டத்தை அதிகரிக்கவும்."
        }
    },
    "Bacterial_Spot": {
        "en": {
            "label": "Bacterial Spot (Xanthomonas)",
            "remedy": "Avoid working on wet foliage. Disinfect cutting shears with 70% alcohol. Spray approved organic copper bactericide and switch to drip irrigation."
        },
        "hi": {
            "label": "जीवाणु धब्बा रोग",
            "remedy": "गीले पौधों को छूने से बचें। औजारों को 70% अल्कोहल से साफ करें। अनुमोदित कॉपर जीवाणुनाशक का छिड़काव करें और ड्रिप सिंचाई का उपयोग करें।"
        },
        "ta": {
            "label": "பாக்டீரியா புள்ளி நோய்",
            "remedy": "ஈரமான செடிகளை கையாள வேண்டாம். 70% ஆல்கஹால் மூலம் கருவிகளை சுத்தம் செய்யவும். அங்கீகரிக்கப்பட்ட பாக்டீரியா கொல்லியை தெளிக்கவும்."
        }
    },
    "Leaf_Mold": {
        "en": {
            "label": "Leaf Mold (Passalora fulva)",
            "remedy": "Increase plant spacing and greenhouse ventilation. Avoid overhead sprinkling. Prune lower suckers to eliminate stagnant humidity pockets."
        },
        "hi": {
            "label": "पत्ती का फफूंद (लीफ मोल्ड)",
            "remedy": "पौधों के बीच दूरी और वेंटिलेशन बढ़ाएँ। ऊपर से पानी देने से बचें और नमी कम करने के लिए निचली शाखाओं की छंटाई करें।"
        },
        "ta": {
            "label": "இலை அச்சு நோய்",
            "remedy": "செடிகளுக்கு இடையே இடைவெளியை அதிகரித்து காற்றோட்டத்தை உறுதி செய்யவும். மேல் தெளிப்பு நீர்ப்பாசனத்தை தவிர்க்கவும்."
        }
    },
    "Mosaic_Virus": {
        "en": {
            "label": "Mosaic Virus (TMV/CMV)",
            "remedy": "Rogue out heavily infested plants. Control insect vectors (aphids/whiteflies) using neem oil or insecticidal soap. Disinfect farm tools frequently."
        },
        "hi": {
            "label": "मोज़ेक वायरस",
            "remedy": "संक्रमित पौधों को उखाड़कर नष्ट करें। नीम के तेल से एफिड्स और सफेद मक्खी का नियंत्रण करें। कृषि उपकरणों को बार-बार कीटाणुरहित करें।"
        },
        "ta": {
            "label": "மொசைக் வைரஸ்",
            "remedy": "பாதிக்கப்பட்ட செடிகளை பிடுங்கி அழிக்கவும். வேப்ப எண்ணெய் மூலம் பூச்சிகளைக் கட்டுப்படுத்தவும். விவசாயக் கருவிகளை சுத்தப்படுத்தவும்."
        }
    },
    "Powdery_Mildew": {
        "en": {
            "label": "Powdery Mildew (Erysiphe)",
            "remedy": "Prune crowded stems to expose the inner canopy to direct sunlight. Spray dilute potassium bicarbonate or neem oil. Avoid high nitrogen fertilizer."
        },
        "hi": {
            "label": "चूर्णिल आसिता (पाउडरी मिल्ड्यू)",
            "remedy": "धूप पहुँचाने के लिए घनी शाखाओं की छंटाई करें। पोटेशियम बाइकार्बोनेट या नीम तेल का छिड़काव करें। अत्यधिक नाइट्रोजन खाद से बचें।"
        },
        "ta": {
            "label": "சாம்பல் நோய்",
            "remedy": "சூரிய ஒளி படும்படி செடிகளை கவாத்து செய்யவும். பொட்டாசியம் பைகார்பனேட் அல்லது வேப்ப எண்ணெய் தெளிக்கவும். அதிகப்படியான உரத்தை தவிர்க்கவும்."
        }
    },
    "Biological_Anomaly": {
        "en": {
            "label": "Biological Anomaly Warning",
            "remedy": "Symptom prediction conflicts with ambient microclimate telemetry (<25% humidity). Check sensor calibration and consult extension officers before spraying."
        },
        "hi": {
            "label": "जैविक विसंगति चेतावनी",
            "remedy": "पहचाने गए लक्षण परिवेशीय आर्द्रता (<25%) से मेल नहीं खाते हैं। सेंसर की जांच करें और किसी भी स्प्रे से पहले कृषि विशेषज्ञ से सलाह लें।"
        },
        "ta": {
            "label": "உயிரியல் ஒழுங்கின்மை எச்சரிக்கை",
            "remedy": "அறிகுறிகள் சுற்றுச்சூழல் ஈரப்பதத்துடன் (<25%) முரண்படுகின்றன. தெளிப்பதற்கு முன் சென்சார் மற்றும் வேளாண் அதிகாரியை அணுகவும்."
        }
    }
}

def download_kaggle_datasets(force: bool = False) -> dict[str, str]:
    """Download configured datasets on demand; requires KaggleHub network connectivity."""
    import kagglehub
    import shutil
    import os

    TABULAR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VISION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, handle in DATASET_HANDLES.items():
        destination = TABULAR_DATA_DIR if kind == "tabular" else VISION_DATA_DIR
        downloaded = kagglehub.dataset_download(handle, force_download=force)
        paths[kind] = downloaded
        # Copy CSV files to local destination if tabular
        if kind == "tabular":
            for fname in os.listdir(downloaded):
                if fname.endswith(".csv"):
                    shutil.copy(os.path.join(downloaded, fname), destination / fname)
    return paths