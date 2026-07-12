import { motion } from "motion/react";

export function CircleIndicator({
  progress = 100,
  color = "var(--chart-1)",
  trackColor = "var(--color-muted)",
  size = 40,
  thickness = 16,
}: {
  progress?: number;
  color?: string;
  trackColor?: string;
  size?: number;
  thickness?: number;
}) {
  const value = Math.max(0, Math.min(progress, 100)) / 100;

  return (
    <div className="relative">
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        className="indicator"
      >
        <path
          d="M50 10 A40 40 0 1 1 50 90 A40 40 0 1 1 50 10"
          fill="none"
          stroke={trackColor}
          strokeWidth={thickness}
        />

        <motion.path
          d="M50 10 A40 40 0 1 1 50 90 A40 40 0 1 1 50 10"
          fill="none"
          stroke={color}
          strokeWidth={thickness}
          strokeLinecap="round"
          pathLength={1}
          strokeDasharray="1 1"
          initial={{ strokeDashoffset: 1 }}
          whileInView={{ strokeDashoffset: 1 - value }}
          viewport={{ once: true, margin: "0px 0px -72px 0px" }}
          transition={{
            duration: 0.8,
            ease: "easeOut",
          }}
        />
      </svg>
    </div>
  );
}
