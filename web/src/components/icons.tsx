type P = { className?: string };

const base = {
  width: 16,
  height: 16,
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export const DocIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M9 1.5H4.5A1.5 1.5 0 0 0 3 3v10A1.5 1.5 0 0 0 4.5 14.5h7A1.5 1.5 0 0 0 13 13V5.5L9 1.5Z" />
    <path d="M9 1.5V5a.5.5 0 0 0 .5.5H13" />
  </svg>
);

export const PlusIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M8 3.5v9M3.5 8h9" />
  </svg>
);

export const CloseIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M4 4l8 8M12 4l-8 8" />
  </svg>
);

export const SendIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M2.5 8h10M8.5 3.5 13 8l-4.5 4.5" />
  </svg>
);

export const StopIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <rect x="4.5" y="4.5" width="7" height="7" rx="1.5" fill="currentColor" stroke="none" />
  </svg>
);

export const CheckIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="m3.5 8.5 3 3 6-7" />
  </svg>
);

export const AlertIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <circle cx="8" cy="8" r="6.25" />
    <path d="M8 5v3.5M8 10.8v.2" />
  </svg>
);

export const ZoomOut = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M4 8h8" />
  </svg>
);

export const ZoomIn = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M8 4v8M4 8h8" />
  </svg>
);

export const SparkIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M8 2.5 9.4 6.6 13.5 8 9.4 9.4 8 13.5 6.6 9.4 2.5 8l4.1-1.4L8 2.5Z" />
  </svg>
);

export const GearIcon = ({ className }: P) => (
  <svg {...base} className={className} aria-hidden="true">
    <circle cx="8" cy="8" r="2.25" />
    <path d="M12.9 9.8a1.2 1.2 0 0 0 .24 1.32l.04.05a1.45 1.45 0 1 1-2.05 2.05l-.05-.04a1.2 1.2 0 0 0-1.32-.24 1.2 1.2 0 0 0-.73 1.1v.13a1.45 1.45 0 0 1-2.9 0v-.07a1.2 1.2 0 0 0-.79-1.1 1.2 1.2 0 0 0-1.32.24l-.05.04A1.45 1.45 0 1 1 1.92 11.2l.04-.05a1.2 1.2 0 0 0 .24-1.32 1.2 1.2 0 0 0-1.1-.73H1a1.45 1.45 0 0 1 0-2.9h.07a1.2 1.2 0 0 0 1.1-.79 1.2 1.2 0 0 0-.24-1.32l-.04-.05A1.45 1.45 0 1 1 3.94 1.99l.05.04a1.2 1.2 0 0 0 1.32.24H5.4a1.2 1.2 0 0 0 .73-1.1V1a1.45 1.45 0 0 1 2.9 0v.07a1.2 1.2 0 0 0 .73 1.1 1.2 1.2 0 0 0 1.32-.24l.05-.04a1.45 1.45 0 1 1 2.05 2.05l-.04.05a1.2 1.2 0 0 0-.24 1.32v.06a1.2 1.2 0 0 0 1.1.73H15a1.45 1.45 0 0 1 0 2.9h-.07a1.2 1.2 0 0 0-1.1.73Z" />
  </svg>
);
