import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import time
from model import MONETInference
from data import load_isic_dataset, get_comprehensive_dermatology_concepts
from utils import analyze_prediction_confidence

# Set page configuration
st.set_page_config(
    page_title="MONET Dermatology Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #333;
    margin: 1rem 0;
}
.result-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.confidence-high {
    background-color: #d4edda;
    border-left-color: #28a745;
}
.confidence-medium {
    background-color: #fff3cd;
    border-left-color: #ffc107;
}
.confidence-low {
    background-color: #f8d7da;
    border-left-color: #dc3545;
}
.progress-text {
    font-size: 1.1rem;
    font-weight: 500;
    color: #495057;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'monet_model' not in st.session_state:
        st.session_state.monet_model = None
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'sample_images' not in st.session_state:
        st.session_state.sample_images = None
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

@st.cache_resource
def load_model():
    """Load the MONET model (cached)."""
    return MONETInference(device="auto")

@st.cache_data
def load_dataset_cached():
    """Load the ISIC dataset (cached)."""
    return load_isic_dataset(split="train", num_samples=500)

def load_sample_images(dataset, num_samples=5):
    """Load random sample images from dataset."""
    if dataset is None:
        return None
    
    np.random.seed(int(time.time()))  # Different seed each time
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False).tolist()
    sample_images = []
    
    for idx in indices:
        sample = dataset[idx]
        sample_images.append({
            'index': idx,
            'image': sample['image'],
            'metadata': {k: v for k, v in sample.items() if k != 'image'}
        })
    
    return sample_images

def display_image_selector(sample_images):
    """Display image selector interface."""
    st.markdown("<div class='sub-header'>📸 Select an Image from Dataset</div>", unsafe_allow_html=True)
    
    if sample_images is None:
        st.error("No dataset images available. Please check your internet connection.")
        return None
    
    # Display images in columns
    cols = st.columns(5)
    selected_idx = None
    
    for i, sample in enumerate(sample_images):
        with cols[i]:
            st.image(sample['image'], caption=f"Image {sample['index']}", use_container_width=True)
            if st.button(f"Select Image {i+1}", key=f"select_{i}"):
                selected_idx = i
    
    if selected_idx is not None:
        selected_sample = sample_images[selected_idx]
        st.success(f"Selected Image {selected_idx + 1} (Dataset Index: {selected_sample['index']})")
        
        # Display metadata
        if selected_sample['metadata']:
            st.write("**Image Metadata:**")
            for key, value in selected_sample['metadata'].items():
                st.write(f"- {key}: {value}")
        
        return selected_sample['image']
    
    return None

def display_file_uploader():
    """Display file upload interface."""
    st.markdown("<div class='sub-header'>📁 Upload Your Own Image</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a dermatology image for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.write("**Image Information:**")
            st.write(f"- Filename: {uploaded_file.name}")
            st.write(f"- Size: {image.size}")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Format: {image.format}")
        
        return image
    
    return None

def run_inference(image, monet_model):
    """Run MONET inference on the selected image."""
    st.markdown("<div class='sub-header'>🔬 Running Analysis</div>", unsafe_allow_html=True)
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load concepts
        status_text.markdown("<div class='progress-text'>Loading dermatology concepts...</div>", unsafe_allow_html=True)
        progress_bar.progress(20)
        time.sleep(0.5)
        
        dermatology_concepts = get_comprehensive_dermatology_concepts()
        
        # Step 2: Preprocessing
        status_text.markdown("<div class='progress-text'>Preprocessing image...</div>", unsafe_allow_html=True)
        progress_bar.progress(40)
        time.sleep(0.5)
        
        # Step 3: Model inference
        status_text.markdown(f"<div class='progress-text'>Running MONET inference on {len(dermatology_concepts)} concepts...</div>", unsafe_allow_html=True)
        progress_bar.progress(60)
        
        results = monet_model.annotate_concepts(image, dermatology_concepts)
        
        # Step 4: Analysis
        status_text.markdown("<div class='progress-text'>Analyzing results...</div>", unsafe_allow_html=True)
        progress_bar.progress(80)
        time.sleep(0.3)
        
        analysis = analyze_prediction_confidence(results, threshold=0.1)
        
        # Step 5: Complete
        status_text.markdown("<div class='progress-text'>Analysis complete!</div>", unsafe_allow_html=True)
        progress_bar.progress(100)
        time.sleep(0.3)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results, analysis
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error during inference: {str(e)}")
        return None, None

def display_results(image, results, analysis):
    """Display analysis results."""
    st.markdown("<div class='sub-header'>📊 Analysis Results</div>", unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Analyzed Image", use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        st.metric("Max Confidence Score", f"{analysis['max_score']:.4f}")
        st.metric("Mean Confidence Score", f"{analysis['mean_score']:.4f}")
        st.metric("High Confidence Concepts", analysis['high_confidence_count'])
    
    with col2:
        # Top concepts visualization
        st.markdown("### 🏆 Top 10 Concept Predictions")
        
        top_concepts = results['ranked_concepts'][:10]
        
        for i, item in enumerate(top_concepts):
            score = item['score']
            concept = item['concept']
            
            # Determine confidence level and styling
            if score > 0.1:
                confidence_class = "confidence-high"
                confidence_label = "HIGH"
            elif score > 0.05:
                confidence_class = "confidence-medium"
                confidence_label = "MEDIUM"
            else:
                confidence_class = "confidence-low"
                confidence_label = "LOW"
            
            # Create styled result card
            st.markdown(f"""
            <div class='result-card {confidence_class}'>
                <strong>{i+1}. {concept}</strong><br>
                Score: {score:.4f} | Confidence: {confidence_label}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed results section
    st.markdown("### 📋 Detailed Results")
    
    # High confidence concepts
    high_confidence_concepts = [
        item for item in results['ranked_concepts'] 
        if item['score'] > 0.1
    ]
    
    if high_confidence_concepts:
        st.success(f"Found {len(high_confidence_concepts)} high-confidence concepts (score > 0.1):")
        for item in high_confidence_concepts:
            st.write(f"• **{item['concept']}**: {item['score']:.4f}")
    else:
        st.warning("No high-confidence concepts found (all scores < 0.1)")
        st.info("Top 3 concepts by score:")
        for item in analysis['top_3_concepts']:
            st.write(f"• **{item['concept']}**: {item['score']:.4f}")
    
    # Generate and display visualization
    st.markdown("### 📊 Visualization")
    
    try:
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top concepts for visualization
        top_k = 8
        top_concepts = results['ranked_concepts'][:top_k]
        concepts = [item['concept'] for item in top_concepts]
        scores = [item['score'] for item in top_concepts]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(concepts)), scores)
        ax.set_yticks(range(len(concepts)))
        ax.set_yticklabels(concepts, fontsize=10)
        ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_k} MONET Concept Annotations', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Color bars based on score
        max_score = max(scores) if scores else 1
        for i, bar in enumerate(bars):
            normalized_score = scores[i] / max_score
            color = plt.cm.viridis(normalized_score)
            bar.set_color(color)
            
            # Add score labels on bars
            width = bar.get_width()
            ax.text(width + max_score * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{scores[i]:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown("<div class='main-header'>🔬 MONET Dermatology Concept Analyzer</div>", unsafe_allow_html=True)
    st.markdown("Advanced AI-powered analysis of dermatological images using MONET (Medical cONcept annoTation)")
    
    # Sidebar for model status and controls
    with st.sidebar:
        st.markdown("## 🛠️ Model Status")
        
        # Load model
        if st.session_state.monet_model is None:
            with st.spinner("Loading MONET model..."):
                st.session_state.monet_model = load_model()
            st.success("✅ MONET model loaded successfully!")
        else:
            st.success("✅ MONET model ready")
        
        # Load dataset
        if st.session_state.dataset is None:
            with st.spinner("Loading ISIC dataset..."):
                st.session_state.dataset = load_dataset_cached()
            if st.session_state.dataset:
                st.success(f"✅ Dataset loaded ({len(st.session_state.dataset)} samples)")
            else:
                st.error("❌ Failed to load dataset")
        else:
            st.success(f"✅ Dataset ready ({len(st.session_state.dataset)} samples)")
        
        st.markdown("---")
        
        # Restart button
        if st.button("🔄 Restart Analysis", help="Clear all selections and start over"):
            st.session_state.selected_image = None
            st.session_state.results = None
            st.session_state.sample_images = None
            st.rerun()
    
    # Main interface
    if not st.session_state.processing:
        # Image selection interface
        st.markdown("## 🖼️ Image Selection")
        
        # Tabs for different input methods
        tab1, tab2 = st.tabs(["📚 Dataset Images", "📁 Upload Image"])
        
        selected_image = None
        
        with tab1:
            # Load sample images if not already loaded
            if st.session_state.sample_images is None:
                if st.button("🎲 Load 5 Random Images"):
                    with st.spinner("Loading random images..."):
                        st.session_state.sample_images = load_sample_images(st.session_state.dataset)
                    st.rerun()
            else:
                selected_image = display_image_selector(st.session_state.sample_images)
                
                if st.button("🎲 Load New Random Images"):
                    st.session_state.sample_images = load_sample_images(st.session_state.dataset)
                    st.rerun()
        
        with tab2:
            uploaded_image = display_file_uploader()
            if uploaded_image:
                selected_image = uploaded_image
        
        # Run analysis button
        if selected_image is not None:
            st.session_state.selected_image = selected_image
            
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run MONET Analysis", key="run_analysis", help="Analyze the selected image"):
                    st.session_state.processing = True
                    st.rerun()
    
    # Run inference if processing
    if st.session_state.processing:
        results, analysis = run_inference(st.session_state.selected_image, st.session_state.monet_model)
        
        if results and analysis:
            st.session_state.results = (results, analysis)
            st.session_state.processing = False
            st.rerun()
        else:
            st.session_state.processing = False
    
    # Display results if available
    if st.session_state.results:
        results, analysis = st.session_state.results
        display_results(st.session_state.selected_image, results, analysis)
        
        # Option to run another analysis
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Analyze Another Image", key="analyze_another"):
                st.session_state.selected_image = None
                st.session_state.results = None
                st.rerun()

if __name__ == "__main__":
    main() 