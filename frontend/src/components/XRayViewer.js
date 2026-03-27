import React, { useRef, useState, useEffect } from 'react';
import UploadArea from './UploadArea';
import './XRayViewer.css';

export default function XRayViewer({
  imageUrl,
  predictions,
  appState,
  onFileSelect,
}) {
  const showUpload = appState === 'idle' || appState === 'error';
  const showImage = imageUrl && appState !== 'idle';
  const showOverlays = appState === 'success' && predictions;
  const isNoCaries = appState === 'no-caries';

  const imgRef = useRef(null);
  const [imgSize, setImgSize] = useState({ width: 0, height: 0, naturalWidth: 1, naturalHeight: 1 });

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const update = () => {
      setImgSize({
        width: img.clientWidth,
        height: img.clientHeight,
        naturalWidth: img.naturalWidth || 1,
        naturalHeight: img.naturalHeight || 1,
      });
    };
    img.addEventListener('load', update);
    window.addEventListener('resize', update);
    update();
    return () => {
      img.removeEventListener('load', update);
      window.removeEventListener('resize', update);
    };
  }, [imageUrl, appState]);

  /* Convert pixel-based bounding boxes to percentage positions relative to image */
  const toPercent = (box) => {
    const { naturalWidth: nw, naturalHeight: nh } = imgSize;
    return {
      left: `${(box.x_min / nw) * 100}%`,
      top: `${(box.y_min / nh) * 100}%`,
      width: `${((box.x_max - box.x_min) / nw) * 100}%`,
      height: `${((box.y_max - box.y_min) / nh) * 100}%`,
    };
  };

  /* Use annotated image from backend when available, otherwise show raw upload */
  const displaySrc =
    showOverlays && predictions.annotated_base64
      ? `data:image/png;base64,${predictions.annotated_base64}`
      : imageUrl;

  return (
    <div className="xray-viewer" role="img" aria-label="X-ray viewer area">
      <div className="xray-viewer-canvas">
        {showUpload && <UploadArea onFileSelect={onFileSelect} />}

        {showImage && (
          <img
            ref={imgRef}
            className="xray-viewer-image"
            src={displaySrc}
            alt="Dental X-ray being analyzed"
          />
        )}

        {(appState === 'success' || isNoCaries) && (
          <div className="xray-inference-badge">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            INFERENCE COMPLETE
          </div>
        )}

        {showOverlays && predictions.bounding_boxes && predictions.bounding_boxes.length > 0 && (
          <div className="xray-overlay">
            {predictions.bounding_boxes.map((box, i) => {
              const pos = toPercent(box);
              return (
                <div
                  key={i}
                  className="xray-bbox"
                  style={pos}
                >
                  <span className="xray-bbox-label">
                    {box.label} {Math.round(box.confidence * 100)}%
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="xray-controls">
        <button className="xray-control-btn" aria-label="Zoom in">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
            <line x1="11" y1="8" x2="11" y2="14" />
            <line x1="8" y1="11" x2="14" y2="11" />
          </svg>
        </button>
        <button className="xray-control-btn" aria-label="Zoom out">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
            <line x1="8" y1="11" x2="14" y2="11" />
          </svg>
        </button>

        <div className="xray-control-separator" />

        <button className="xray-control-btn" aria-label="Adjust contrast">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 2a10 10 0 0 1 0 20z" fill="currentColor" opacity="0.3" />
          </svg>
        </button>

        <button className="xray-control-btn" aria-label="Crosshair measurement">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="22" y1="12" x2="18" y2="12" />
            <line x1="6" y1="12" x2="2" y2="12" />
            <line x1="12" y1="6" x2="12" y2="2" />
            <line x1="12" y1="22" x2="12" y2="18" />
          </svg>
        </button>

        <div className="xray-control-separator" />

        <button className="xray-control-btn" aria-label="Rotate image">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
        </button>

        <button className="xray-control-btn" aria-label="Toggle grid overlay">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <line x1="3" y1="9" x2="21" y2="9" />
            <line x1="3" y1="15" x2="21" y2="15" />
            <line x1="9" y1="3" x2="9" y2="21" />
            <line x1="15" y1="3" x2="15" y2="21" />
          </svg>
        </button>
      </div>
    </div>
  );
}
