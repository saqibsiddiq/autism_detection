import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time

def get_rtc_configuration():
    """Get WebRTC configuration with multiple fallback options"""
    return RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
            {"urls": ["stun:openrelay.metered.ca:80"]},
            {"urls": ["stun:relay.metered.ca:80"]},
            {"urls": ["stun:stun.ekiga.net"]}
        ],
        "iceTransportPolicy": "all",
        "iceCandidatePoolSize": 10
    })

def create_webrtc_streamer_with_fallback(key, video_processor_factory, **kwargs):
    """Create WebRTC streamer with fallback options for camera issues"""
    
    # Add connection troubleshooting info
    with st.expander("📹 Camera Connection Help", expanded=False):
        st.markdown("""
        **If camera connection fails:**
        
        1. **Allow camera permission** in your browser
        2. **Check camera usage** - close other apps using camera
        3. **Try different browser** (Chrome, Firefox, Safari)
        4. **Refresh the page** and try again
        5. **Check network connection** - stable internet required
        
        **Network troubleshooting:**
        - Try connecting from different network
        - Disable VPN if using one
        - Check firewall settings
        
        **Browser permissions:**
        - Click camera icon in address bar
        - Allow camera access for this site
        - Refresh page after granting permission
        """)
    
    # Connection status indicator
    connection_status = st.empty()
    
    try:
        # Configure WebRTC with robust settings
        rtc_config = get_rtc_configuration()
        
        # Create streamer with timeout handling
        webrtc_ctx = webrtc_streamer(
            key=key,
            video_processor_factory=video_processor_factory,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"min": 15, "ideal": 30, "max": 60}
                },
                "audio": False
            },
            async_processing=True,
            **kwargs
        )
        
        # Check connection status
        if webrtc_ctx.state.playing:
            connection_status.success("🟢 Camera connected successfully!")
        elif webrtc_ctx.state.signalling:
            connection_status.info("🟡 Connecting to camera...")
        else:
            connection_status.warning("🔴 Camera not connected. Click 'START' to begin.")
            
        return webrtc_ctx
        
    except Exception as e:
        connection_status.error(f"❌ Camera connection error: {str(e)}")
        
        # Show alternative options
        st.error("""
        **Camera Connection Failed**
        
        This can happen due to:
        - Network firewall blocking WebRTC
        - Browser not supporting required features
        - Camera permission denied
        - Multiple apps using camera
        
        **Quick fixes:**
        1. Refresh the page and try again
        2. Allow camera permission in browser
        3. Close other camera apps
        4. Try a different browser
        """)
        
        return None

def show_camera_setup_guide():
    """Show comprehensive camera setup guide"""
    st.markdown("""
    ### 📹 Camera Setup Guide
    
    **Before starting the test:**
    
    1. **Position yourself properly:**
       - Sit 18-24 inches from screen
       - Face the camera directly
       - Ensure good lighting on your face
    
    2. **Browser setup:**
       - Use Chrome, Firefox, or Safari
       - Allow camera permission when prompted
       - Close other apps using camera
    
    3. **Environment:**
       - Quiet space with minimal distractions
       - Stable internet connection
       - Good lighting (avoid backlighting)
    """)

def check_camera_availability():
    """Check if camera is available and working"""
    try:
        # Simple camera test
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        return False
    except:
        return False

def show_connection_diagnostics():
    """Show connection diagnostics and troubleshooting"""
    with st.expander("🔧 Connection Diagnostics", expanded=False):
        
        st.markdown("**System Check:**")
        
        # Check camera availability
        camera_available = check_camera_availability()
        if camera_available:
            st.success("✅ Camera device detected")
        else:
            st.warning("⚠️ No camera device found or camera in use")
        
        # Show browser info
        st.markdown("**Browser Compatibility:**")
        st.info("""
        - ✅ Chrome (recommended)
        - ✅ Firefox
        - ✅ Safari (macOS)
        - ❌ Internet Explorer (not supported)
        """)
        
        # Network test
        st.markdown("**Network Requirements:**")
        st.info("""
        - Stable internet connection required
        - WebRTC support needed
        - STUN server access required
        - No VPN blocking WebRTC traffic
        """)
        
        # Manual troubleshooting steps
        if st.button("🔄 Test Camera Access"):
            if check_camera_availability():
                st.success("Camera test successful! Try starting the assessment again.")
            else:
                st.error("Camera test failed. Please check the troubleshooting steps above.")

def create_simple_camera_test():
    """Create a simple camera test without WebRTC"""
    st.markdown("### 📱 Simple Camera Test")
    
    if st.button("Test Camera", key="simple_camera_test"):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    st.success("✅ Camera is working! You can proceed with the tests.")
                    # Show a frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Camera Test Image", width=300)
                else:
                    st.error("❌ Camera found but cannot capture frames")
                cap.release()
            else:
                st.error("❌ No camera device found")
        except Exception as e:
            st.error(f"❌ Camera test failed: {str(e)}")
            
        st.markdown("""
        **If camera test works but WebRTC fails:**
        - Browser may be blocking WebRTC
        - Network firewall blocking connections
        - Try different browser or network
        """)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        # (Optional) Process the image here
        # For now, just return the frame as-is
        return av.VideoFrame.from_ndarray(img, format="bgr24")