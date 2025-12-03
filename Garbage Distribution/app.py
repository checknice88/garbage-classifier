"""
Streamlit Real-time Garbage Classification Application
"""

# Fix OpenMP library conflict on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import io
from model import GarbageClassifier
from config import (
    CLASS_NAMES, CLASS_TO_CATEGORY, CATEGORY_COLORS,
    MODEL_SAVE_PATH, DISTRIBUTION_PLOT_PATH, NUM_CLASSES, IMAGE_SIZE,
    AVAILABLE_CITIES, get_city_config, get_city_mapping, DATA_DIR,
    PREPARATION_TIPS, ACHIEVEMENTS, BADGE_RARITY_COLORS,
    DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
)
from achievement_system import AchievementSystem
from image_enhancement import enhance_image, compare_images, calculate_brightness
from confusion_analysis import check_boundary_warning, load_confusion_matrix
from map_service import MapService
from i18n import (
    t,
    get_category_label,
    get_preparation_tip,
    get_achievement_text,
)
import streamlit_folium
import hashlib
import time
import uuid








@st.cache_resource
def load_model():
    """
    Load the trained model (cached for performance).
    
    Returns:
        Loaded model, device, and class mapping
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = GarbageClassifier(num_classes=NUM_CLASSES, pretrained=False)
    
    # Load trained weights and class mapping
    idx_to_class = None
    dataset_classes = None
    
    if os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load class mapping if available
        if 'idx_to_class' in checkpoint:
            idx_to_class = checkpoint['idx_to_class']
            dataset_classes = checkpoint.get('dataset_classes', CLASS_NAMES)
            st.success(f"Model loaded from {MODEL_SAVE_PATH}")
            st.info(f"Using class mapping from training: {dataset_classes}")
        else:
            st.warning("Model checkpoint doesn't contain class mapping. Using config classes.")
            # Fallback: create mapping from current dataset
            from data_utils import get_class_mapping
            _, idx_to_class_temp, dataset_classes_temp = get_class_mapping(DATA_DIR)
            idx_to_class = idx_to_class_temp
            dataset_classes = dataset_classes_temp
    else:
        st.error(f"❌ **CRITICAL: Model file not found at {MODEL_SAVE_PATH}**")
        st.error("**The app is using an UNTRAINED model, which will give random predictions!**")
        st.warning("⚠️ **Please train the model first by running:** `python train.py`")
        st.info("The model needs to be trained on your dataset before it can make accurate predictions.")
        # Get mapping from dataset
        from data_utils import get_class_mapping
        _, idx_to_class_temp, dataset_classes_temp = get_class_mapping(DATA_DIR)
        idx_to_class = idx_to_class_temp
        dataset_classes = dataset_classes_temp
    
    model = model.to(device)
    model.eval()
    
    return model, device, idx_to_class, dataset_classes


def preprocess_image(image):
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict(image, model, device, idx_to_class, dataset_classes, city_id='default', use_tta=True):
    """
    Predict garbage class from image with optional Test-Time Augmentation (TTA).
    
    Args:
        image: PIL Image
        model: Trained model
        device: Device to run inference on
        city_id: City identifier for city-specific mapping
        use_tta: Whether to use test-time augmentation for better accuracy
        
    Returns:
        predicted_class: Class name
        predicted_category: Broad category
        confidence: Confidence score
        all_probs: All class probabilities
    """
    model.eval()
    
    # Create a copy of the image to avoid modifying the original
    # This ensures each prediction is independent
    image_copy = image.copy() if hasattr(image, 'copy') else image
    
    if use_tta:
        # Test-Time Augmentation: Average predictions from multiple augmented versions
        # Convert to RGB if needed (on copy)
        if image_copy.mode != 'RGB':
            image_copy = image_copy.convert('RGB')
        
        # Base transform
        base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get predictions from multiple augmentations
        all_outputs = []
        with torch.no_grad():
            # 1. Original image (use copy)
            img_tensor = base_transform(image_copy).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            all_outputs.append(outputs)
            
            # 2. Horizontal flip (create new copy for each transformation)
            img_flipped = image_copy.transpose(Image.FLIP_LEFT_RIGHT)
            img_tensor = base_transform(img_flipped).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            all_outputs.append(outputs)
            
            # 3. Rotated +10 degrees (create new copy)
            img_rotated1 = image_copy.rotate(10, expand=False, fillcolor=(128, 128, 128))
            img_tensor = base_transform(img_rotated1).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            all_outputs.append(outputs)
            
            # 4. Rotated -10 degrees (create new copy)
            img_rotated2 = image_copy.rotate(-10, expand=False, fillcolor=(128, 128, 128))
            img_tensor = base_transform(img_rotated2).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            all_outputs.append(outputs)
        
        # Average the predictions
        avg_output = torch.stack(all_outputs).mean(0)
        probabilities = torch.nn.functional.softmax(avg_output[0], dim=0)
    else:
        # Standard single prediction (use copy)
        image_tensor = preprocess_image(image_copy).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_idx = predicted_idx.item()
    
    # Get class name using the mapping from training
    if idx_to_class is not None and predicted_idx in idx_to_class:
        predicted_class = idx_to_class[predicted_idx]
    else:
        # Fallback to config classes
        if predicted_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_idx]
        else:
            predicted_class = CLASS_NAMES[0]  # Default fallback
            st.warning(f"Predicted index {predicted_idx} out of range. Using default class.")
    
    # Use city-specific mapping
    city_mapping = get_city_mapping(city_id)
    predicted_category = city_mapping[predicted_class]
    confidence_score = confidence.item()
    
    # Get all probabilities
    all_probs = probabilities.cpu().numpy()
    
    return predicted_class, predicted_category, confidence_score, all_probs


def main():
    """
    Main Streamlit application.
    """
    if 'language' not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
    
    # Page configuration
    st.set_page_config(
        page_title=t('app.page_title', st.session_state.language),
        page_icon="🗑️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    def translate(key: str, **kwargs) -> str:
        return t(key, st.session_state.get('language', DEFAULT_LANGUAGE), **kwargs)
    
    # Title
    st.title(translate("app.heading"))
    st.markdown(f"**{translate('app.tagline')}**")
    st.markdown("---")
    
    # Initialize session state for city selection
    if 'selected_city' not in st.session_state:
        st.session_state.selected_city = 'default'
    
    # Initialize achievement system (must be before sidebar to use in sidebar)
    achievement_system = AchievementSystem()
    
    # Initialize user ID in session state (for multi-user support in future)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 'default'
    if 'last_recorded_signature' not in st.session_state:
        st.session_state.last_recorded_signature = None
    if 'last_recorded_run_id' not in st.session_state:
        st.session_state.last_recorded_run_id = None
    if 'pending_achievement_ids' not in st.session_state:
        st.session_state.pending_achievement_ids = []
    
    # Sidebar
    with st.sidebar:
        language_options = list(SUPPORTED_LANGUAGES.keys())
        current_language = st.session_state.language
        selected_language = st.selectbox(
            translate("language.selector_label"),
            options=language_options,
            index=language_options.index(current_language),
            format_func=lambda code: SUPPORTED_LANGUAGES.get(code, {}).get("label", code),
            help=translate("language.selector_help"),
            key='language_selector'
        )
        if selected_language != current_language:
            st.session_state.language = selected_language
            st.rerun()
        
        st.header(translate("sidebar.city_header"))
        
        # City selector
        city_keys = list(AVAILABLE_CITIES.keys())
        current_index = city_keys.index(st.session_state.selected_city) if st.session_state.selected_city in city_keys else 0
        
        selected_city = st.selectbox(
            translate("sidebar.city_selector_label"),
            options=city_keys,
            index=current_index,
            format_func=lambda x: AVAILABLE_CITIES[x],
            help=translate("sidebar.city_selector_help"),
            key='city_selector'
        )
        
        # Update session state
        st.session_state.selected_city = selected_city
        
        # Get city configuration
        city_config = get_city_config(selected_city)
        city_mapping = get_city_mapping(selected_city)
        city_labels = city_config['category_labels']
        city_colors = city_config['category_colors']
        
        # Display selected city info
        city_name_cn = city_config.get('name_cn')
        city_extra = f" ({city_name_cn})" if city_name_cn else ""
        st.info(f"📍 **{city_config['name']}**{city_extra}")
        st.caption(translate("sidebar.city_info_caption"))
        
        st.markdown("---")
        st.header(translate("sidebar.achievements_header"))
        
        # Get user statistics
        user_stats = achievement_system.get_or_create_user(st.session_state.user_id)
        user_achievements = achievement_system.get_user_achievements(st.session_state.user_id)
        
        # Display statistics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric(translate("sidebar.stats.total"), user_stats['total_classifications'])
        with col_stat2:
            st.metric(translate("sidebar.stats.recyclable"), user_stats['recyclable_count'])
        with col_stat3:
            st.metric(translate("sidebar.stats.hazardous"), user_stats['hazardous_count'])
        with col_stat4:
            st.metric(translate("sidebar.stats.kitchen"), user_stats['kitchen_count'])
        
        # Display unlocked achievements
        if user_achievements:
            st.markdown("### " + translate("sidebar.badges_header"))
            badge_cols = st.columns(min(5, len(user_achievements)))
            for idx, achievement_id in enumerate(user_achievements[:5]):
                if achievement_id in ACHIEVEMENTS:
                    achievement = ACHIEVEMENTS[achievement_id]
                    rarity_color = BADGE_RARITY_COLORS.get(achievement['rarity'], '#9E9E9E')
                    localized_text = get_achievement_text(achievement_id, st.session_state.language)
                    with badge_cols[idx % len(badge_cols)]:
                        badge_html = f"""
                        <div style="
                            text-align: center;
                            padding: 10px;
                            background-color: {rarity_color}20;
                            border: 2px solid {rarity_color};
                            border-radius: 10px;
                            margin: 5px;
                        ">
                            <div style="font-size: 30px;">{achievement['icon']}</div>
                            <div style="font-size: 12px; font-weight: bold; color: {rarity_color};">
                                {localized_text.get('name', achievement['name'])}
                            </div>
                        </div>
                        """
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.caption(localized_text.get('description', achievement.get('description', '')))
            
            if len(user_achievements) > 5:
                with st.expander(translate("sidebar.badges_more", count=len(user_achievements))):
                    for achievement_id in user_achievements[5:]:
                        if achievement_id in ACHIEVEMENTS:
                            achievement = ACHIEVEMENTS[achievement_id]
                            rarity_color = BADGE_RARITY_COLORS.get(achievement['rarity'], '#9E9E9E')
                            localized_text = get_achievement_text(achievement_id, st.session_state.language)
                            st.markdown(
                                f"**{achievement['icon']} {localized_text.get('name', achievement['name'])}** - "
                                f"{localized_text.get('description', achievement.get('description', ''))}"
                            )
        else:
            st.info(translate("sidebar.badges_none"))
        
        st.markdown("---")
        st.header(translate("sidebar.dataset_header"))
        
        if os.path.exists(DISTRIBUTION_PLOT_PATH):
            st.subheader(translate("sidebar.dataset_chart_title"))
            st.image(DISTRIBUTION_PLOT_PATH)
            st.caption(translate("sidebar.dataset_chart_caption"))
        else:
            st.info(translate("sidebar.dataset_chart_missing"))
        
        st.markdown("---")
        st.subheader(translate("sidebar.about_header"))
        st.markdown(translate("sidebar.about_text"))
        
        st.markdown("---")
        st.subheader(translate("sidebar.categories_header", city=city_config['name']))
        for category, color in city_colors.items():
            category_label = city_labels.get(category)
            localized_label = get_category_label(category, st.session_state.language)
            if st.session_state.language == 'zh' and category_label:
                display_label = category_label
            elif category_label and category_label != localized_label and st.session_state.language != 'zh':
                display_label = f"{localized_label} / {category_label}"
            else:
                display_label = localized_label
            st.markdown(
                f"<span style='color: {color}; font-weight: bold;'>●</span> "
                f"**{category}** ({display_label})",
                unsafe_allow_html=True
            )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(translate("input.section_title"))
        
        # Image input options
        input_methods = {
            "upload": translate("input.method_upload"),
            "camera": translate("input.method_camera")
        }
        input_option = st.radio(
            translate("input.method_label"),
            options=list(input_methods.keys()),
            format_func=lambda key: input_methods[key],
            horizontal=True
        )
        
        image = None
        image_signature = None
        
        if input_option == "upload":
            uploaded_file = st.file_uploader(
                translate("input.upload_label"),
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help=translate("input.upload_help")
            )
            if uploaded_file is not None:
                # Read image into memory to ensure fresh image each time
                uploaded_file.seek(0)  # Reset file pointer
                file_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(file_bytes))
                # Create a copy to ensure independence
                image = image.copy()
                # Compute image signature for achievement tracking
                image_signature = hashlib.md5(file_bytes).hexdigest()
        else:
            camera_image = st.camera_input(translate("input.camera_label"), key="camera_input")
            if camera_image is not None:
                # Read image into memory to ensure fresh image each time
                image_bytes = camera_image.read()
                image = Image.open(io.BytesIO(image_bytes))
                # Create a copy to ensure independence
                image = image.copy()
                image_signature = hashlib.md5(image_bytes).hexdigest()
        
        # Image enhancement option
        current_signature = image_signature
        previous_signature = st.session_state.get("active_image_signature")
        if current_signature:
            if previous_signature != current_signature:
                st.session_state["active_image_signature"] = current_signature
                st.session_state.pop("last_prediction", None)
                st.session_state["last_recorded_signature"] = None
                st.session_state["last_recorded_run_id"] = None
        else:
            if previous_signature:
                st.session_state.pop("active_image_signature", None)
                st.session_state.pop("last_prediction", None)
                st.session_state["last_recorded_signature"] = None
                st.session_state["last_recorded_run_id"] = None
        
        if image is not None:
            st.markdown(translate("enhancement.section_title"))
            use_enhancement = st.checkbox(
                translate("enhancement.toggle_label"),
                value=True,
                help=translate("enhancement.toggle_help")
            )
            
            if use_enhancement:
                # Apply enhancement
                enhanced_image, enhancement_info = enhance_image(
                    image,
                    auto_brightness=True,
                    denoise=True,
                    sharpen=True
                )
                
                # Show enhancement info
                original_brightness = enhancement_info['original_brightness']
                brightness_status = (
                    translate("enhancement.metric.adjusted")
                    if enhancement_info['brightness_adjusted']
                    else translate("enhancement.metric.normal")
                )
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric(translate("enhancement.metric.original"), f"{original_brightness*100:.1f}%", 
                             delta=f"{brightness_status}")
                with col_info2:
                    st.metric(
                        translate("enhancement.metric.denoise"),
                        translate("enhancement.metric.applied") if enhancement_info['denoised'] else translate("enhancement.metric.skipped")
                    )
                with col_info3:
                    st.metric(
                        translate("enhancement.metric.sharpen"),
                        translate("enhancement.metric.applied") if enhancement_info['sharpened'] else translate("enhancement.metric.skipped")
                    )
                
                # Show comparison
                with st.expander(translate("enhancement.comparison_title")):
                    comparison = compare_images(image, enhanced_image)
                    st.image(comparison, caption=translate("enhancement.comparison_caption"))
                
                # Use enhanced image for prediction
                image = enhanced_image
            else:
                # Show original brightness info
                original_brightness = calculate_brightness(image)
                if original_brightness < 0.3:
                    st.warning(translate("enhancement.dark_warning", value=original_brightness*100))
        
        # Display image
        if image is not None:
            st.image(image, caption=translate("image.caption_ready"))
    
    with col2:
        st.subheader(translate("results.section_title"))
        stored_result = st.session_state.get("last_prediction")
        result_data = None
        has_image = image is not None
        classify_clicked = False
        
        if has_image:
            classify_clicked = st.button(
                translate("results.start_button"),
                key="start_classification",
                type="primary",
                use_container_width=True,
                help=translate("results.start_help")
            )
        
        if classify_clicked and has_image:
            classification_run_id = f"{st.session_state.get('active_image_signature') or 'no_sig'}-{uuid.uuid4().hex}"
            with st.spinner(translate("results.spinner_model")):
                model, device, idx_to_class, dataset_classes = load_model()
            
            current_city = st.session_state.get('selected_city', 'default')
            city_config = get_city_config(current_city)
            city_mapping = get_city_mapping(current_city)
            city_labels = city_config['category_labels']
            city_colors = city_config['category_colors']
            
            with st.spinner(translate("results.spinner_classify")):
                predicted_class, predicted_category, confidence, all_probs = predict(
                    image, model, device, idx_to_class, dataset_classes, city_id=current_city
                )
            
            boundary_warning = check_boundary_warning(
                all_probs, idx_to_class, dataset_classes,
                confidence_threshold=0.4,
                diff_threshold=0.15
            )
            
            if isinstance(idx_to_class, dict):
                idx_mapping = dict(idx_to_class)
            elif idx_to_class is not None:
                idx_mapping = list(idx_to_class)
            else:
                idx_mapping = None
            
            dataset_list = list(dataset_classes) if dataset_classes is not None else None
            
            result_data = {
                "image_signature": st.session_state.get("active_image_signature"),
                "city_id": current_city,
                "predicted_class": predicted_class,
                "predicted_category": predicted_category,
                "confidence": float(confidence),
                "all_probs": all_probs.tolist(),
                "boundary_warning": boundary_warning,
                "idx_to_class": idx_mapping,
                "dataset_classes": dataset_list,
                "timestamp": time.time(),
                "classification_run_id": classification_run_id
            }
            st.session_state["last_prediction"] = result_data
            
            should_record = predicted_class and confidence > 0.1
            if should_record and st.session_state.get("last_recorded_run_id") == classification_run_id:
                should_record = False
            
            if should_record:
                achievement_system.record_classification(
                    user_id=st.session_state.user_id,
                    class_name=predicted_class,
                    category=predicted_category,
                    confidence=confidence,
                    user_id_param=st.session_state.user_id
                )
                
                newly_unlocked = achievement_system.check_and_unlock_achievements(
                    user_id=st.session_state.user_id,
                    achievement_config=ACHIEVEMENTS
                )
                
                if newly_unlocked:
                    st.session_state["pending_achievement_ids"] = newly_unlocked
                
                st.session_state["last_recorded_signature"] = st.session_state.get("active_image_signature")
                st.session_state["last_recorded_run_id"] = classification_run_id
                st.session_state["stats_refresh_trigger"] = time.time()
                st.rerun()
        elif stored_result:
            result_data = stored_result
        
        if result_data:
            result_city = result_data.get("city_id", st.session_state.get('selected_city', 'default'))
            city_config = get_city_config(result_city)
            city_mapping = get_city_mapping(result_city)
            city_labels = city_config['category_labels']
            city_colors = city_config['category_colors']
            
            predicted_class = result_data['predicted_class']
            predicted_category = result_data['predicted_category']
            confidence = result_data['confidence']
            all_probs = np.array(result_data['all_probs'])
            boundary_warning = result_data.get('boundary_warning')
            stored_idx_to_class = result_data.get('idx_to_class')
            stored_dataset_classes = result_data.get('dataset_classes')
            
            def resolve_class_name_from_idx(class_idx: int) -> str:
                if isinstance(stored_idx_to_class, dict):
                    name = stored_idx_to_class.get(class_idx)
                    if name:
                        return name
                elif isinstance(stored_idx_to_class, list):
                    if 0 <= class_idx < len(stored_idx_to_class):
                        return stored_idx_to_class[class_idx]
                if 0 <= class_idx < len(CLASS_NAMES):
                    return CLASS_NAMES[class_idx]
                return f"Class_{class_idx}"
            
            city_name_display = city_config['name']
            if city_config.get('name_cn'):
                city_name_display = f"{city_config['name']} ({city_config['name_cn']})"
            st.caption(translate("results.city_caption", city_name=city_name_display))
            st.markdown(translate("results.result_heading"))
            
            category_color = city_colors[predicted_category]
            city_label_value = city_labels.get(predicted_category)
            localized_category_label = get_category_label(predicted_category, st.session_state.language)
            if st.session_state.language == 'zh' and city_label_value:
                category_label_display = city_label_value
            elif city_label_value and city_label_value != localized_category_label and st.session_state.language != 'zh':
                category_label_display = f"{localized_category_label} / {city_label_value}"
            else:
                category_label_display = localized_category_label
            
            category_line = translate("results.category_label", category=predicted_category, label=category_label_display)
            confidence_line = translate("results.confidence_label", value=confidence * 100)
            
            result_html = f"""
            <div style="
                background-color: {category_color}20;
                border-left: 5px solid {category_color};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <h3 style="color: {category_color}; margin: 0;">
                    {predicted_class.replace('-', ' ').title()}
                </h3>
                <p style="margin: 5px 0; font-size: 18px;">
                    {category_line}
                </p>
                <p style="margin: 5px 0; font-size: 16px;">
                    {confidence_line}
                </p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
            
            pending_achievements = st.session_state.pop("pending_achievement_ids", None)
            if pending_achievements:
                st.markdown("---")
                st.markdown(translate("achievements.new_title"))
                for achievement_id in pending_achievements:
                    if achievement_id in ACHIEVEMENTS:
                        achievement = ACHIEVEMENTS[achievement_id]
                        rarity_color = BADGE_RARITY_COLORS.get(achievement['rarity'], '#9E9E9E')
                        localized_text = get_achievement_text(achievement_id, st.session_state.language)
                        unlock_html = f"""
                        <div style="
                            background: linear-gradient(135deg, {rarity_color}20 0%, {rarity_color}40 100%);
                            border: 3px solid {rarity_color};
                            padding: 20px;
                            border-radius: 15px;
                            margin: 10px 0;
                            text-align: center;
                        ">
                            <div style="font-size: 50px; margin-bottom: 10px;">
                                {achievement['icon']}
                            </div>
                            <h3 style="color: {rarity_color}; margin: 5px 0;">
                                {localized_text.get('name', achievement['name'])}
                            </h3>
                            <p style="margin: 5px 0; font-size: 14px;">
                                {localized_text.get('description', achievement.get('description', ''))}
                            </p>
                        </div>
                        """
                        st.markdown(unlock_html, unsafe_allow_html=True)
                st.balloons()
            
            localized_tip = get_preparation_tip(predicted_class, st.session_state.language)
            if localized_tip:
                st.markdown("---")
                st.markdown(translate("tips.section_title"))
                tip_content = translate(
                    "tips.card_prefix",
                    class_name=predicted_class.replace('-', ' ').title(),
                    tip=localized_tip
                )
                tip_html = f"""
                <div style="
                    background-color: #E3F2FD;
                    border-left: 4px solid #2196F3;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                ">
                    <p style="margin: 0; font-size: 15px; color: #1565C0;">
                        {tip_content}
                    </p>
                </div>
                """
                st.markdown(tip_html, unsafe_allow_html=True)
            
            if boundary_warning:
                st.markdown("---")
                st.markdown(translate("warning.section_title"))
                warning_html = f"""
                <div style="
                    background-color: #FFF3CD;
                    border-left: 4px solid #FFC107;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                ">
                    <p style="margin: 0; font-size: 14px; color: #856404;">
                        {boundary_warning['tip']}
                    </p>
                </div>
                """
                st.markdown(warning_html, unsafe_allow_html=True)
                
                if boundary_warning['checks']:
                    st.markdown(translate("warning.checks_title"))
                    for check in boundary_warning['checks']:
                        st.markdown(f"- {check}")
                    
                    st.markdown(f"\n**{translate('warning.tip_prefix')}**")
                    st.markdown(f"- {boundary_warning['class1'].replace('-', ' ').title()}: {boundary_warning['prob1']*100:.1f}%")
                    st.markdown(f"- {boundary_warning['class2'].replace('-', ' ').title()}: {boundary_warning['prob2']*100:.1f}%")
                    st.markdown(f"- Difference: {boundary_warning['prob_diff']*100:.1f}%")
            
            st.progress(confidence)
            
            st.markdown(translate("top.section_title"))
            top3_indices = np.argsort(all_probs)[-3:][::-1]
            
            for i, idx in enumerate(top3_indices):
                class_name = resolve_class_name_from_idx(idx)
                prob = all_probs[idx]
                category = city_mapping.get(class_name, "Unknown")
                color = city_colors.get(category, "#607D8B")
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f"**{i+1}. {class_name.replace('-', ' ').title()}** "
                        f"({category})"
                    )
                with col_b:
                    st.markdown(f"**{prob*100:.1f}%**")
                
                st.markdown(
                    f'<div style="background-color: {color}40; height: 8px; '
                    f'width: {prob*100}%; border-radius: 4px; margin-bottom: 10px;"></div>',
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            popup_key = f"recycling_popup_{predicted_category}"
            user_choice_key = f"user_choice_{predicted_category}"
            
            if user_choice_key not in st.session_state:
                st.info(
                    f"{translate('location.prompt_title', label=category_label_display)}\n\n"
                    f"{translate('location.prompt_body')}"
                )
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button(translate("location.prompt_yes"), key=f"yes_{popup_key}", type="primary"):
                        st.session_state[user_choice_key] = True
                        st.rerun()
                with col2:
                    if st.button(translate("location.prompt_no"), key=f"no_{popup_key}"):
                        st.session_state[user_choice_key] = False
                        st.rerun()
                with col3:
                    if st.button(translate("location.prompt_later"), key=f"later_{popup_key}"):
                        st.session_state[user_choice_key] = None
                        st.rerun()
            
            show_location_finder = st.session_state.get(user_choice_key, False)
            
            if show_location_finder:
                map_service = MapService()
                default_location_label = translate("location.default_location")
                
                st.markdown(translate("location.input_title"))
                st.info(translate("location.input_hint"))
                
                user_lat = None
                user_lon = None
                start_label = None
                coordinates_ready = False
                address_input = None
                address_text_for_query = None
                
                location_methods = {
                    "address": translate("location.method_address"),
                    "coordinates": translate("location.method_coordinates")
                }
                location_method = st.radio(
                    translate("location.method_label"),
                    list(location_methods.keys()),
                    format_func=lambda key: location_methods[key],
                    horizontal=True,
                    key=f"location_method_{predicted_category}"
                )
                
                if location_method == "address":
                    address_input = st.text_input(
                        translate("location.address_label"),
                        key=f"address_input_{predicted_category}",
                        placeholder=translate("location.address_placeholder")
                    )
                    
                    if address_input:
                        with st.spinner(translate("location.address_spinner")):
                            coords = map_service.geocode(address_input)
                            if coords:
                                user_lat, user_lon = coords
                                cleaned_label = address_input.strip()
                                start_label = cleaned_label if cleaned_label else default_location_label
                                address_text_for_query = start_label
                                coordinates_ready = True
                                st.success(translate("location.address_success", lat=user_lat, lon=user_lon))
                            else:
                                st.warning(translate("location.address_fail"))
                                coordinates_ready = False
                else:
                    col_lat, col_lon = st.columns(2)
                    with col_lat:
                        user_lat = st.number_input(
                            translate("location.coords_lat"),
                            value=31.2304,
                            format="%.6f",
                            key=f"lat_input_{predicted_category}"
                        )
                    with col_lon:
                        user_lon = st.number_input(
                            translate("location.coords_lon"),
                            value=121.4737,
                            format="%.6f",
                            key=f"lon_input_{predicted_category}"
                        )
                    
                    start_label_input = st.text_input(
                        translate("location.coords_name_label"),
                        key=f"start_label_{predicted_category}",
                        placeholder=translate("location.coords_name_placeholder")
                    )
                    
                    if user_lat and user_lon:
                        coordinates_ready = True
                        start_label = start_label_input.strip() if start_label_input else default_location_label
                        address_text_for_query = start_label
                
                
                search_triggered = False
                if coordinates_ready and user_lat and user_lon:
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        search_triggered = st.button(
                            translate("location.search_button"),
                            key=f"search_button_{predicted_category}",
                            type="primary",
                            use_container_width=True
                        )
                
                if search_triggered and user_lat and user_lon:
                    try:
                        with st.spinner(translate("location.search_spinner")):
                            poi_search_url = map_service.generate_poi_search_url(
                                category=predicted_category,
                                language=st.session_state.language,
                                latitude=user_lat,
                                longitude=user_lon,
                                address_text=address_text_for_query
                            )
                        st.success(translate("location.search_success"))
                        st.markdown("---")
                        link_text = translate("location.search_link_text", label=category_label_display)
                        st.markdown(
                            f"[{link_text}]({poi_search_url})"
                        )
                        st.caption(translate("location.search_caption"))
                    except Exception as e:
                        st.error(translate("location.search_error", error=e))
                else:
                    st.info(translate("location.search_prompt"))
            
            with st.expander(translate("results.all_probs_title")):
                prob_data = []
                classes_to_use = stored_dataset_classes if stored_dataset_classes is not None else CLASS_NAMES
                for idx in range(len(classes_to_use)):
                    class_name = resolve_class_name_from_idx(idx)
                    prob = all_probs[idx] if idx < len(all_probs) else 0.0
                    category = city_mapping.get(class_name, "Unknown")
                    category_label = city_labels.get(category, "Unknown")
                    prob_data.append({
                        "Class": class_name.replace('-', ' ').title(),
                        "Category": f"{category} ({category_label})",
                        "Probability": f"{prob*100:.2f}%"
                    })
                
                df = pd.DataFrame(prob_data)
                df = df.sort_values("Probability", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                st.dataframe(df, hide_index=True)
        else:
            if has_image:
                st.info(translate("results.need_click"))
            else:
                st.info(translate("results.need_image"))
                st.markdown(translate("results.howto_title"))
                st.markdown(translate("results.howto_upload"))
                st.markdown(translate("results.howto_camera"))
                st.markdown(translate("results.howto_button"))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        f"{translate('footer.text')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

