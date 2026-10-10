import React, { useState, useRef, useEffect, useCallback } from 'react';

export default function ComparisonSlider({ originalSrc, overlaySrc, alt = 'Comparison' }) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef(null);
  const isDragging = useRef(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const updateSize = () => {
      if (containerRef.current) {
        setContainerWidth(containerRef.current.clientWidth);
      }
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(containerRef.current);
    window.addEventListener('resize', updateSize);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateSize);
    };
  }, []);

  const handleMove = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const position = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(position);
  }, []);

  const handleTouchMove = (e) => {
    if (isDragging.current && e.touches[0]) {
      handleMove(e.touches[0].clientX);
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging.current) {
      handleMove(e.clientX);
    }
  };

  const handleStart = () => {
    isDragging.current = true;
  };

  const handleEnd = () => {
    isDragging.current = false;
  };

  return (
    <div
      ref={containerRef}
      className="slider-container"
      onMouseMove={handleMouseMove}
      onMouseDown={handleStart}
      onMouseUp={handleEnd}
      onMouseLeave={handleEnd}
      onTouchMove={handleTouchMove}
      onTouchStart={handleStart}
      onTouchEnd={handleEnd}
      style={{ position: 'relative', width: '100%', minHeight: '400px', backgroundColor: '#0f172a' }}
    >
      {/* Background Image: Original Optical Scene */}
      <img
        src={originalSrc}
        alt="Original optical satellite raster"
        className="slider-image-base"
        style={{ width: '100%', height: 'auto', display: 'block', maxHeight: '520px', objectFit: 'contain' }}
        draggable={false}
      />

      {/* Foreground Clipped Image: Predicted Flood Overlay */}
      <div
        className="slider-image-overlay"
        style={{
          width: `${sliderPosition}%`,
          position: 'absolute',
          top: 0,
          left: 0,
          bottom: 0,
          overflow: 'hidden'
        }}
      >
        <img
          src={overlaySrc}
          alt="Predicted flood segmentation overlay"
          style={{
            width: containerWidth > 0 ? `${containerWidth}px` : (containerRef.current ? `${containerRef.current.clientWidth}px` : '100%'),
            height: '100%',
            maxWidth: 'none',
            objectFit: 'contain'
          }}
          draggable={false}
        />
      </div>

      {/* Vertical Slider Handle Line */}
      <div
        className="slider-handle"
        style={{ left: `${sliderPosition}%` }}
      >
        <div className="slider-handle-button">
          <span style={{ fontSize: '10px', fontWeight: 'bold' }}>&#8596;</span>
        </div>
      </div>

      {/* Badges indicating sides */}
      <div style={{
        position: 'absolute',
        bottom: '12px',
        left: '12px',
        backgroundColor: 'rgba(15, 23, 42, 0.75)',
        color: '#ffffff',
        padding: '3px 8px',
        borderRadius: '3px',
        fontSize: '11px',
        fontFamily: 'JetBrains Mono, monospace',
        pointerEvents: 'none'
      }}>
        Flood Overlay
      </div>
      <div style={{
        position: 'absolute',
        bottom: '12px',
        right: '12px',
        backgroundColor: 'rgba(15, 23, 42, 0.75)',
        color: '#ffffff',
        padding: '3px 8px',
        borderRadius: '3px',
        fontSize: '11px',
        fontFamily: 'JetBrains Mono, monospace',
        pointerEvents: 'none'
      }}>
        Original Satellite
      </div>
    </div>
  );
}
