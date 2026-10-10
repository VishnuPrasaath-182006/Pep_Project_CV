import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle } from 'lucide-react';
import { api } from '../../services/api';
import Badge from '../common/Badge';

export default function ModelSelector({ selectedModel, onSelectModel, disabled }) {
  const [models, setModels] = useState([]);

  useEffect(() => {
    api.getModels()
      .then(res => setModels(res.models))
      .catch(() => {});
  }, []);

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-header">
        <h3 className="card-title">
          <Cpu size={18} color="#2563eb" />
          <span>Model Architecture Selection</span>
        </h3>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        {models.map(model => {
          const isSelected = selectedModel === model.id;

          return (
            <div
              key={model.id}
              onClick={() => {
                if (!disabled) onSelectModel(model.id);
              }}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                padding: '0.75rem 1rem',
                borderRadius: '4px',
                border: isSelected ? '2px solid #2563eb' : '1px solid #e2e8f0',
                backgroundColor: isSelected ? '#eff6ff' : '#ffffff',
                cursor: disabled ? 'not-allowed' : 'pointer',
                opacity: 1,
                transition: 'all 0.15s ease'
              }}
            >
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                <input
                  type="radio"
                  name="modelSelection"
                  checked={isSelected}
                  disabled={disabled}
                  onChange={() => onSelectModel(model.id)}
                  style={{ marginTop: '3px', cursor: 'pointer' }}
                />
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.875rem', color: '#102a43' }}>
                    {model.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>
                    {model.architecture}
                  </div>
                  {model.historical_metrics && (
                    <div style={{ fontSize: '0.7rem', fontFamily: 'JetBrains Mono, monospace', color: '#0f766e', marginTop: '4px' }}>
                      Benchmark: {((model.historical_metrics.mean_iou || model.historical_metrics.macro_miou || 0) * 100).toFixed(1)}% mIoU &bull; {((model.historical_metrics.dice || 0) * 100).toFixed(1)}% F1
                    </div>
                  )}
                </div>
              </div>

              <div>
                <Badge variant="ready">
                  <CheckCircle size={11} />
                  <span>Active / Ready</span>
                </Badge>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
