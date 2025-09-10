import React from 'react'
import axios from 'axios'

export default function FaceTest() {
  const videoRef = React.useRef(null)

  React.useEffect(() => {
    async function initCam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
      } catch (e) {
        console.error('Camera error', e)
      }
    }
    initCam()
    return () => {
      const v = videoRef.current
      if (v?.srcObject) {
        const tracks = v.srcObject.getTracks()
        tracks.forEach(t => t.stop())
      }
    }
  }, [])

  return (
    <div>
      <h3>Face Recognition Test (Preview)</h3>
      <video ref={videoRef} style={{ width: '100%', maxWidth: 800, background: '#000', borderRadius: 8 }} muted playsInline />
      <p style={{ marginTop: 8 }}>Webcam preview is working. Gaze processing will be added with MediaPipe.</p>
    </div>
  )
}


