"""
Configuration file for Garbage Classification System
Defines class mappings and UI color codes
"""

import os

from i18n import (
    SUPPORTED_LANGUAGES as I18N_SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE as I18N_DEFAULT_LANGUAGE,
    CATEGORY_LABELS,
    PREPARATION_TIPS_I18N,
)

SUPPORTED_LANGUAGES = I18N_SUPPORTED_LANGUAGES
DEFAULT_LANGUAGE = I18N_DEFAULT_LANGUAGE

# Mapping of specific classes to broad waste management categories
# Based on Chinese waste classification standards

CLASS_TO_CATEGORY = {
    # Recyclable (可回收物)
    'paper': 'Recyclable',
    'cardboard': 'Recyclable',
    'brown-glass': 'Recyclable',
    'green-glass': 'Recyclable',
    'white-glass': 'Recyclable',
    'metal': 'Recyclable',
    'plastic': 'Recyclable',
    'clothes': 'Recyclable',
    'shoes': 'Recyclable',
    
    # Hazardous (有害垃圾)
    'battery': 'Hazardous',
    
    # Kitchen/Wet (厨余垃圾)
    'biological': 'Kitchen',
    
    # Residual/Other (其他垃圾)
    'trash': 'Other'
}

# Color codes for UI display
CATEGORY_COLORS = {
    'Recyclable': '#1E88E5',  # Blue
    'Hazardous': '#D32F2F',   # Red
    'Kitchen': '#388E3C',     # Green
    'Other': '#616161'        # Gray
}

# Category labels (defaulting to Chinese for legacy usage)
CATEGORY_LABELS_CN = CATEGORY_LABELS.get('zh', {})

# Get all class names (sorted for consistency)
CLASS_NAMES = sorted(CLASS_TO_CATEGORY.keys())

# Number of classes
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# Map Service Configuration (地图服务配置)
# ============================================================================

# Gaode (Amap) API Configuration
# Get API key from environment variable or use default for demo
GAODE_API_KEY = os.environ.get('GAODE_API_KEY', '8a0877a90a140a273601590e549527da')
GAODE_API_BASE_URL = 'https://restapi.amap.com/v3'

# Recycling Location Database Configuration
RECYCLING_LOCATION_DB_PATH = 'recycling_locations.db'

# Default search radius for nearby recycling points (in kilometers)
DEFAULT_SEARCH_RADIUS = 5.0

# ============================================================================
# Preparation Tips (投掷前处理建议)
# ============================================================================
# Pre-processing guidance (default Chinese, localized via i18n helper)
PREPARATION_TIPS = PREPARATION_TIPS_I18N.get('zh', {})

# ============================================================================
# Achievement System Configuration (成就系统配置)
# ============================================================================

# Achievement definitions with thresholds and descriptions
ACHIEVEMENTS = {
    # Beginner achievements
    'first_classification': {
        'name': '垃圾分类新手',
        'name_en': 'Classification Beginner',
        'description': '完成第一次垃圾分类',
        'description_en': 'Complete your first classification',
        'icon': '🌱',
        'type': 'total_classifications',
        'threshold': 1,
        'rarity': 'common'
    },
    'ten_classifications': {
        'name': '分类小能手',
        'name_en': 'Classification Apprentice',
        'description': '完成10次垃圾分类',
        'description_en': 'Complete 10 classifications',
        'icon': '⭐',
        'type': 'total_classifications',
        'threshold': 10,
        'rarity': 'common'
    },
    'fifty_classifications': {
        'name': '分类达人',
        'name_en': 'Classification Expert',
        'description': '完成50次垃圾分类',
        'description_en': 'Complete 50 classifications',
        'icon': '🏆',
        'type': 'total_classifications',
        'threshold': 50,
        'rarity': 'rare'
    },
    'hundred_classifications': {
        'name': '分类大师',
        'name_en': 'Classification Master',
        'description': '完成100次垃圾分类',
        'description_en': 'Complete 100 classifications',
        'icon': '👑',
        'type': 'total_classifications',
        'threshold': 100,
        'rarity': 'epic'
    },
    'five_hundred_classifications': {
        'name': '环保传奇',
        'name_en': 'Environmental Legend',
        'description': '完成500次垃圾分类',
        'description_en': 'Complete 500 classifications',
        'icon': '🌟',
        'type': 'total_classifications',
        'threshold': 500,
        'rarity': 'legendary'
    },
    
    # Category-specific achievements
    'recyclable_enthusiast': {
        'name': '回收达人',
        'name_en': 'Recyclable Enthusiast',
        'description': '识别50件可回收物',
        'description_en': 'Classify 50 recyclable items',
        'icon': '♻️',
        'type': 'category_count',
        'category': 'Recyclable',
        'threshold': 50,
        'rarity': 'rare'
    },
    'hazardous_guardian': {
        'name': '有害垃圾守护者',
        'name_en': 'Hazardous Guardian',
        'description': '识别10件有害垃圾',
        'description_en': 'Classify 10 hazardous items',
        'icon': '⚠️',
        'type': 'hazardous_count',
        'threshold': 10,
        'rarity': 'epic'
    },
    'hazardous_expert': {
        'name': '有害垃圾专家',
        'name_en': 'Hazardous Expert',
        'description': '识别50件有害垃圾',
        'description_en': 'Classify 50 hazardous items',
        'icon': '🛡️',
        'type': 'hazardous_count',
        'threshold': 50,
        'rarity': 'legendary'
    },
    'kitchen_warrior': {
        'name': '厨余战士',
        'name_en': 'Kitchen Warrior',
        'description': '识别30件厨余垃圾',
        'description_en': 'Classify 30 kitchen waste items',
        'icon': '🍃',
        'type': 'category_count',
        'category': 'Kitchen',
        'threshold': 30,
        'rarity': 'rare'
    },
    'all_rounder': {
        'name': '全能分类师',
        'name_en': 'All-Round Classifier',
        'description': '识别过所有4大类别的垃圾',
        'description_en': 'Classify items from all 4 categories',
        'icon': '🎯',
        'type': 'all_categories',
        'threshold': 1,  # At least 1 in each category
        'rarity': 'epic'
    }
}

# Badge display configuration
BADGE_RARITY_COLORS = {
    'common': '#9E9E9E',      # Gray
    'rare': '#2196F3',        # Blue
    'epic': '#9C27B0',        # Purple
    'legendary': '#FF9800'   # Orange/Gold
}


# Model configuration
MODEL_NAME = 'mobilenet_v3_small'
IMAGE_SIZE = 224
BATCH_SIZE = 32  # Increase to 64 or 128 if GPU memory allows for faster training
NUM_EPOCHS = 50  # Increased for better convergence
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-6  # Minimum learning rate for cosine annealing

# Training improvements
USE_FOCAL_LOSS = True  # Use Focal Loss instead of CrossEntropy (takes priority over label smoothing)
FOCAL_LOSS_GAMMA = 2.0  # Focusing parameter for Focal Loss
USE_LABEL_SMOOTHING = False  # Use label smoothing (only if Focal Loss is False)
LABEL_SMOOTHING = 0.1  # Label smoothing factor
USE_GRADIENT_CLIPPING = True  # Clip gradients to prevent explosion
MAX_GRAD_NORM = 1.0  # Maximum gradient norm
USE_COSINE_SCHEDULER = True  # Use cosine annealing instead of step LR
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs

# Paths
DATA_DIR = 'data/raw'
TRAIN_DIR = 'data/train'
MODEL_SAVE_PATH = 'best_model.pth'
DISTRIBUTION_PLOT_PATH = 'distribution.png'
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'
CONFUSION_MATRIX_DATA_PATH = 'confusion_matrix_data.npz'  # Save confusion matrix data

# ============================================================================
# City/Region Specific Classification Standards
# ============================================================================

# Default/National Standard (used as base)
DEFAULT_CITY_CONFIG = {
    'name': 'Default (National Standard)',
    'name_cn': '默认（国家标准）',
    'class_to_category': CLASS_TO_CATEGORY.copy(),
    'category_labels': CATEGORY_LABELS_CN.copy(),
    'category_colors': CATEGORY_COLORS.copy()
}

# Shanghai Classification Standards
SHANGHAI_CONFIG = {
    'name': 'Shanghai',
    'name_cn': '上海',
    'class_to_category': {
        # Recyclable (可回收物)
        'paper': 'Recyclable',
        'cardboard': 'Recyclable',
        'brown-glass': 'Recyclable',
        'green-glass': 'Recyclable',
        'white-glass': 'Recyclable',
        'metal': 'Recyclable',
        'plastic': 'Recyclable',
        'clothes': 'Recyclable',
        'shoes': 'Recyclable',
        # Hazardous (有害垃圾)
        'battery': 'Hazardous',
        # Kitchen/Wet (湿垃圾)
        'biological': 'Kitchen',
        # Residual/Dry (干垃圾) - Shanghai uses "Dry Waste" instead of "Other"
        'trash': 'Other'
    },
    'category_labels': {
        'Recyclable': '可回收物',
        'Hazardous': '有害垃圾',
        'Kitchen': '湿垃圾',
        'Other': '干垃圾'
    },
    'category_colors': CATEGORY_COLORS.copy()
}

# Beijing Classification Standards
BEIJING_CONFIG = {
    'name': 'Beijing',
    'name_cn': '北京',
    'class_to_category': {
        # Recyclable (可回收物)
        'paper': 'Recyclable',
        'cardboard': 'Recyclable',
        'brown-glass': 'Recyclable',
        'green-glass': 'Recyclable',
        'white-glass': 'Recyclable',
        'metal': 'Recyclable',
        'plastic': 'Recyclable',
        'clothes': 'Recyclable',
        'shoes': 'Recyclable',
        # Hazardous (有害垃圾)
        'battery': 'Hazardous',
        # Kitchen/Wet (厨余垃圾)
        'biological': 'Kitchen',
        # Residual/Other (其他垃圾)
        'trash': 'Other'
    },
    'category_labels': {
        'Recyclable': '可回收物',
        'Hazardous': '有害垃圾',
        'Kitchen': '厨余垃圾',
        'Other': '其他垃圾'
    },
    'category_colors': CATEGORY_COLORS.copy()
}

# Shenzhen Classification Standards
SHENZHEN_CONFIG = {
    'name': 'Shenzhen',
    'name_cn': '深圳',
    'class_to_category': {
        # Recyclable (可回收物)
        'paper': 'Recyclable',
        'cardboard': 'Recyclable',
        'brown-glass': 'Recyclable',
        'green-glass': 'Recyclable',
        'white-glass': 'Recyclable',
        'metal': 'Recyclable',
        'plastic': 'Recyclable',
        'clothes': 'Recyclable',
        'shoes': 'Recyclable',
        # Hazardous (有害垃圾)
        'battery': 'Hazardous',
        # Kitchen/Wet (易腐垃圾) - Shenzhen uses "Perishable Waste"
        'biological': 'Kitchen',
        # Residual/Other (其他垃圾)
        'trash': 'Other'
    },
    'category_labels': {
        'Recyclable': '可回收物',
        'Hazardous': '有害垃圾',
        'Kitchen': '易腐垃圾',
        'Other': '其他垃圾'
    },
    'category_colors': CATEGORY_COLORS.copy()
}

# Guangzhou Classification Standards
GUANGZHOU_CONFIG = {
    'name': 'Guangzhou',
    'name_cn': '广州',
    'class_to_category': {
        # Recyclable (可回收物)
        'paper': 'Recyclable',
        'cardboard': 'Recyclable',
        'brown-glass': 'Recyclable',
        'green-glass': 'Recyclable',
        'white-glass': 'Recyclable',
        'metal': 'Recyclable',
        'plastic': 'Recyclable',
        'clothes': 'Recyclable',
        'shoes': 'Recyclable',
        # Hazardous (有害垃圾)
        'battery': 'Hazardous',
        # Kitchen/Wet (餐厨垃圾) - Guangzhou uses "Kitchen Waste"
        'biological': 'Kitchen',
        # Residual/Other (其他垃圾)
        'trash': 'Other'
    },
    'category_labels': {
        'Recyclable': '可回收物',
        'Hazardous': '有害垃圾',
        'Kitchen': '餐厨垃圾',
        'Other': '其他垃圾'
    },
    'category_colors': CATEGORY_COLORS.copy()
}

# City configurations dictionary
CITY_CONFIGS = {
    'default': DEFAULT_CITY_CONFIG,
    'shanghai': SHANGHAI_CONFIG,
    'beijing': BEIJING_CONFIG,
    'shenzhen': SHENZHEN_CONFIG,
    'guangzhou': GUANGZHOU_CONFIG
}

# Available cities for selection
AVAILABLE_CITIES = {
    'default': 'Default (National Standard) / 默认（国家标准）',
    'shanghai': 'Shanghai / 上海',
    'beijing': 'Beijing / 北京',
    'shenzhen': 'Shenzhen / 深圳',
    'guangzhou': 'Guangzhou / 广州'
}


def get_city_config(city_id='default'):
    """
    Get city-specific configuration.
    
    Args:
        city_id: City identifier (default, shanghai, beijing, shenzhen, guangzhou)
        
    Returns:
        City configuration dictionary
    """
    return CITY_CONFIGS.get(city_id, DEFAULT_CITY_CONFIG)


def get_city_mapping(city_id='default'):
    """
    Get city-specific class to category mapping.
    
    Args:
        city_id: City identifier
        
    Returns:
        Dictionary mapping class names to categories
    """
    config = get_city_config(city_id)
    return config['class_to_category']

