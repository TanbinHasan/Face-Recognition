# File: app.py
import streamlit as st
import numpy as np
import cv2
import time
from pathlib import Path
from deepface import DeepFace

# Page config
st.set_page_config(page_title="AI Face Recognition", page_icon="🤖", layout="centered")

# Simple styling
st.markdown("""
<style>
.header {background: linear-gradient(90deg, #4F46E5, #7C3AED); padding: 1rem; border-radius: 8px; color: white; text-align: center;}
.card {background: #F9FAFB; padding: 1rem; border-radius: 8px; border-left: 4px solid #4F46E5; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h2>🤖 AI Face Recognition</h2></div>', unsafe_allow_html=True)

# Performance note
st.info("ℹ️ **Performance Note**: AI processing takes 30-60 seconds per operation on CPU. GPU warnings in terminal are normal.")

# Initialize paths
DATASET_DIR = Path("Dataset")
EMBEDDINGS_FILE = "embeddings.npy"
DATASET_DIR.mkdir(exist_ok=True)

# Navigation
tab1, tab2, tab3 = st.tabs(["📸 Create Dataset", "🧠 Train Model", "🔍 Recognize"])

# TAB 1: Create Dataset
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    name = st.text_input("👤 Person Name:")
    samples = st.selectbox("📊 Samples:", [20, 30, 50], index=1)
    
    if name.strip():
        camera = st.camera_input("📷 Take Photo")
        
        if camera:
            # Save image
            person_dir = DATASET_DIR / name
            person_dir.mkdir(exist_ok=True)
            
            bytes_data = camera.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            count = len(list(person_dir.glob("*.jpg"))) + 1
            cv2.imwrite(str(person_dir / f"{name}_{count}.jpg"), image)
            
            progress = min(count / samples, 1.0)
            st.progress(progress)
            st.success(f"✅ Saved {count}/{samples}")
            
            if count >= samples:
                st.balloons()
                st.success(f"🎉 Dataset complete for {name}!")
    else:
        st.warning("⚠️ Enter a name first")
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Train Model
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    people_count = len([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    
    if people_count > 0:
        st.info(f"📊 Found {people_count} person(s)")
        
        if st.button("🚀 Train Model", type="primary"):
            try:
                # Show initial loading message
                with st.spinner("🔄 Initializing AI models... (This may take 30-60 seconds)"):
                    time.sleep(1)  # Brief pause to show the message
                
                embeddings = {}
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                people = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
                total_people = len(people)
                
                for i, person_dir in enumerate(people):
                    person_name = person_dir.name
                    
                    # Update status
                    status_placeholder.markdown(f"🔄 **Processing: {person_name}** ({i+1}/{total_people})")
                    
                    embeddings[person_name] = []
                    images = list(person_dir.glob("*.jpg"))
                    processed = 0
                    
                    for j, img_path in enumerate(images):
                        # Show progress for each person
                        if j % 5 == 0:  # Update every 5 images
                            status_placeholder.markdown(f"🔄 **Processing: {person_name}** - Image {j+1}/{len(images)}")
                        
                        try:
                            # Convert image to RGB format that DeepFace expects
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                
                                result = DeepFace.represent(
                                    img_rgb, 
                                    model_name="Facenet", 
                                    enforce_detection=False
                                )
                                embeddings[person_name].append(result[0]["embedding"])
                                processed += 1
                        except Exception as e:
                            continue
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_people)
                    
                    # Show completion for this person
                    status_placeholder.markdown(f"✅ **{person_name}**: {processed}/{len(images)} images processed")
                    time.sleep(0.5)  # Brief pause to show completion
                
                # Final save step
                status_placeholder.markdown("💾 **Saving model...**")
                np.save(EMBEDDINGS_FILE, embeddings)
                
                # Success message
                status_placeholder.markdown("🎉 **Training completed successfully!**")
                st.success("✅ Model trained and ready for recognition!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("💡 Try with fewer images or check image quality")
    else:
        st.warning("⚠️ Create dataset first")
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Recognize
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if Path(EMBEDDINGS_FILE).exists():
        try:
            with st.spinner("🔄 Loading trained model..."):
                embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
            st.success(f"✅ Model loaded with {len(embeddings)} people")
        except:
            st.error("❌ Failed to load model")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📸 **Tip**: Make sure your face is well-lit and centered")
            camera = st.camera_input("🎥 Capture for Recognition")
            
            if camera:
                try:
                    bytes_data = camera.getvalue()
                    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Create placeholder for status updates
                    status_placeholder = st.empty()
                    result_placeholder = st.empty()
                    
                    with st.spinner("🤖 AI is analyzing your face... (30-60 seconds)"):
                        # Step 1: Face embedding
                        status_placeholder.info("🔍 Step 1/3: Extracting face features...")
                        
                        face_result = DeepFace.represent(
                            image_rgb, 
                            model_name="Facenet", 
                            enforce_detection=False
                        )
                        face_embed = face_result[0]["embedding"]
                        
                        # Step 2: Face matching
                        status_placeholder.info("🎯 Step 2/3: Comparing with known faces...")
                        
                        best_match = "Unknown"
                        max_similarity = 0
                        
                        for person, embeds in embeddings.items():
                            for embed in embeds:
                                # Calculate cosine similarity
                                dot_product = np.dot(face_embed, embed)
                                norm_a = np.linalg.norm(face_embed)
                                norm_b = np.linalg.norm(embed)
                                similarity = dot_product / (norm_a * norm_b)
                                
                                if similarity > max_similarity:
                                    max_similarity = similarity
                                    best_match = person
                        
                        # Step 3: Face analysis
                        status_placeholder.info("📊 Step 3/3: Analyzing face attributes...")
                        
                        analysis_result = DeepFace.analyze(
                            image_rgb, 
                            actions=["age", "gender", "emotion"], 
                            enforce_detection=False
                        )
                        analysis = analysis_result[0] if isinstance(analysis_result, list) else analysis_result
                    
                    # Clear status and show results
                    status_placeholder.empty()
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image_rgb, width=300, caption="Captured Image")
                    
                    with col2:
                        st.markdown("### 📊 Recognition Results")
                        
                        if max_similarity > 0.6:  # Lower threshold for better detection
                            st.metric("👤 Identity", best_match)
                            st.metric("🎯 Confidence", f"{max_similarity:.1%}")
                            
                            # Color-coded confidence
                            if max_similarity > 0.8:
                                st.success("🟢 High confidence match!")
                            elif max_similarity > 0.7:
                                st.info("🟡 Good match")
                            else:
                                st.warning("🟠 Low confidence match")
                        else:
                            st.metric("👤 Identity", "Unknown")
                            st.metric("🎯 Confidence", "Too low")
                            st.error("🔴 No matching person found")
                        
                        # Face analysis
                        age = analysis.get("age", "N/A")
                        gender_data = analysis.get("gender", {})
                        emotion_data = analysis.get("emotion", {})
                        
                        if isinstance(gender_data, dict):
                            gender = max(gender_data, key=gender_data.get)
                        else:
                            gender = str(gender_data)
                            
                        if isinstance(emotion_data, dict):
                            emotion = max(emotion_data, key=emotion_data.get)
                        else:
                            emotion = str(emotion_data)
                        
                        st.metric("🎂 Age", f"{int(age)}")
                        st.metric("⚧ Gender", gender)
                        st.metric("😊 Emotion", emotion)
                        
                except Exception as e:
                    st.error(f"❌ Recognition failed: {str(e)}")
                    with st.expander("🔧 Troubleshooting"):
                        st.write("• Make sure your face is clearly visible")
                        st.write("• Try better lighting")
                        st.write("• Ensure the camera is stable")
                        st.write(f"• Technical error: {str(e)}")
    else:
        st.warning("⚠️ No trained model found. Please train the model first.")
        st.info("💡 Go to 'Train Model' tab after creating a dataset")
    
    st.markdown('</div>', unsafe_allow_html=True)