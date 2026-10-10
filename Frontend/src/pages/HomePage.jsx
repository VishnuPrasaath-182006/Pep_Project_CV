import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, BookOpen, Layers, ShieldCheck, Zap, Compass, ArrowRight } from 'lucide-react';

export default function HomePage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem', padding: '1rem 0' }}>
      {/* Hero Section */}
      <section style={{
        textAlign: 'center',
        padding: '3.5rem 1.5rem',
        background: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        boxShadow: '0 1px 3px 0 rgba(15, 23, 42, 0.04)'
      }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 12px',
          backgroundColor: '#eff6ff',
          border: '1px solid #bfdbfe',
          borderRadius: '9999px',
          color: '#2563eb',
          fontSize: '0.75rem',
          fontFamily: 'JetBrains Mono, monospace',
          fontWeight: 600,
          marginBottom: '1.25rem'
        }}>
          <ShieldCheck size={14} />
          <span>Research-Grade Earth Observation Platform &bull; PyTorch 2.x</span>
        </div>

        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: 700,
          color: '#102a43',
          letterSpacing: '-0.025em',
          lineHeight: 1.2,
          maxWidth: '840px',
          margin: '0 auto 1rem auto'
        }}>
          Flood Inundation Detection &amp; Semantic Segmentation
        </h1>

        <p style={{
          fontSize: '1.1rem',
          color: '#475569',
          maxWidth: '680px',
          margin: '0 auto 2rem auto',
          lineHeight: 1.6
        }}>
          AI-powered satellite imagery analysis for flood extent identification using advanced
          semantic segmentation models, featuring 2D Selective State Space (UNet-Mamba) architectures.
        </p>

        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <Link to="/analyze" className="btn btn-primary btn-lg">
            <Activity size={18} />
            <span>Start Satellite Analysis</span>
            <ArrowRight size={16} />
          </Link>
          <Link to="/models" className="btn btn-secondary btn-lg">
            <BookOpen size={18} />
            <span>Explore Architectures</span>
          </Link>
        </div>
      </section>

      {/* Model Architecture Matrix Cards */}
      <section>
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#102a43' }}>
            Supported Deep Learning Architectures
          </h2>
          <p style={{ fontSize: '0.9rem', color: '#64748b', marginTop: '0.25rem' }}>
            Multi-model semantic segmentation framework built for high-resolution multispectral imagery
          </p>
        </div>

        <div className="grid-3">
          {/* UNet-Mamba */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{
                display: 'inline-flex',
                padding: '8px',
                backgroundColor: '#eff6ff',
                borderRadius: '6px',
                color: '#2563eb',
                marginBottom: '1rem'
              }}>
                <Zap size={22} />
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#102a43', marginBottom: '0.5rem' }}>
                UNet-Mamba (CAVE-Flood)
              </h3>
              <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5, marginBottom: '1rem' }}>
                Combines ResT hierarchical encoder with 2D Selective State Space (SS2D) sequence modeling.
                Captures global context with linear complexity, excelling in continuous hydrological boundaries.
              </p>
            </div>
            <div style={{ paddingTop: '0.75rem', borderTop: '1px solid #f1f5f9' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace' }}>
                <span style={{ color: '#64748b' }}>Status:</span>
                <span style={{ color: '#059669', fontWeight: 600 }}>Active (171 MB Weights)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', marginTop: '4px' }}>
                <span style={{ color: '#64748b' }}>Benchmark mIoU:</span>
                <span style={{ fontWeight: 600 }}>75.42% (Held-Out Test)</span>
              </div>
            </div>
          </div>

          {/* Standard U-Net */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{
                display: 'inline-flex',
                padding: '8px',
                backgroundColor: '#f1f5f9',
                borderRadius: '6px',
                color: '#0f766e',
                marginBottom: '1rem'
              }}>
                <Layers size={22} />
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#102a43', marginBottom: '0.5rem' }}>
                Standard U-Net (Baseline)
              </h3>
              <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5, marginBottom: '1rem' }}>
                Classical 4-stage convolutional encoder-decoder baseline with symmetric lateral skip connections.
                Serves as benchmark standard for spatial feature preservation.
              </p>
            </div>
            <div style={{ paddingTop: '0.75rem', borderTop: '1px solid #f1f5f9' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace' }}>
                <span style={{ color: '#64748b' }}>Status:</span>
                <span style={{ color: '#64748b' }}>Pending Training</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', marginTop: '4px' }}>
                <span style={{ color: '#64748b' }}>Historical mIoU:</span>
                <span style={{ fontWeight: 600 }}>73.26%</span>
              </div>
            </div>
          </div>

          {/* SegFormer */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{
                display: 'inline-flex',
                padding: '8px',
                backgroundColor: '#f8fafc',
                borderRadius: '6px',
                color: '#0ea5e9',
                marginBottom: '1rem'
              }}>
                <Compass size={22} />
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#102a43', marginBottom: '0.5rem' }}>
                SegFormer-B2
              </h3>
              <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: 1.5, marginBottom: '1rem' }}>
                Hierarchical vision transformer with positional-encoding-free Mix Transformer (MiT) encoder
                and lightweight All-MLP decoder for computational efficiency.
              </p>
            </div>
            <div style={{ paddingTop: '0.75rem', borderTop: '1px solid #f1f5f9' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace' }}>
                <span style={{ color: '#64748b' }}>Status:</span>
                <span style={{ color: '#64748b' }}>Pending Training</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace', marginTop: '4px' }}>
                <span style={{ color: '#64748b' }}>Historical mIoU:</span>
                <span style={{ fontWeight: 600 }}>55.98% (19-Fold LOEO)</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Scientific Workflow Pipeline */}
      <section className="card" style={{ padding: '2rem' }}>
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#102a43', marginBottom: '1rem' }}>
          End-to-End Analysis Workflow
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.5rem' }}>
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#2563eb', fontFamily: 'JetBrains Mono, monospace' }}>
              STEP 01
            </div>
            <div style={{ fontWeight: 600, color: '#102a43', marginTop: '0.25rem', marginBottom: '0.25rem' }}>
              Ingest Satellite Raster
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Upload PlanetScope 4-band GeoTIFF, Sentinel scenes, or standard optical imagery.
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#2563eb', fontFamily: 'JetBrains Mono, monospace' }}>
              STEP 02
            </div>
            <div style={{ fontWeight: 600, color: '#102a43', marginTop: '0.25rem', marginBottom: '0.25rem' }}>
              Inspect Metadata
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Inspect dimensions, radiometric bit-depth, coordinate system (CRS), and band allocation.
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#2563eb', fontFamily: 'JetBrains Mono, monospace' }}>
              STEP 03
            </div>
            <div style={{ fontWeight: 600, color: '#102a43', marginTop: '0.25rem', marginBottom: '0.25rem' }}>
              Tiled Model Inference
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Execute 512&times;512 sliding window tiled inference with overlap smoothing.
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#2563eb', fontFamily: 'JetBrains Mono, monospace' }}>
              STEP 04
            </div>
            <div style={{ fontWeight: 600, color: '#102a43', marginTop: '0.25rem', marginBottom: '0.25rem' }}>
              Export Delineation
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Download binary PNG masks, vivid flood overlays, and georeferenced GeoTIFFs.
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
