import React from 'react';

export default function Badge({ variant = 'default', children, className = '' }) {
  const variantClass = {
    ready: 'chip-ready',
    success: 'chip-ready',
    pending: 'chip-pending',
    alert: 'chip-alert',
    default: 'chip-pending',
  }[variant] || 'chip-pending';

  return (
    <span className={`telemetry-chip ${variantClass} ${className}`}>
      {children}
    </span>
  );
}
