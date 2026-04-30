"use client";

import { JSX, useEffect, useState } from "react";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { color } from "chart.js/helpers";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

/* ---------- Types ---------- */
interface MetricsData {
  ann: number;
  lr: number;
  rf: number;
}

interface MetricCardProps {
  title: string;
  value: number;
  best: boolean;
}

/* ---------- Component ---------- */
export default function Metrics(): JSX.Element {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [bestModel, setBestModel] = useState<string>("");

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch("http://localhost:5000/metrics");
        const data: MetricsData = await res.json();
        setMetrics(data);

        const entries = Object.entries(data);
        const [bestKey] = entries.reduce(
          (prev, curr) => (curr[1] > prev[1] ? curr : prev),
          ["", -Infinity],
        );

        const modelNames: Record<string, string> = {
          ann: "ANN",
          lr: "Logistic Regression",
          rf: "Random Forest",
        };

        setBestModel(modelNames[bestKey] || "");
      } catch (err) {
        console.error("Error fetching metrics:", err);
      }
    };

    fetchMetrics();
  }, []);

  if (!metrics) {
    return (
      <p className="text-center mt-10 text-brand-fg">Loading metrics...</p>
    );
  }

  const chartData = {
    labels: ["ANN", "Logistic Regression", "Random Forest"],
    datasets: [
      {
        label: "Accuracy (%)",
        data: [metrics.ann, metrics.lr, metrics.rf],
        backgroundColor: [
          "#22c55e",
          "#3b82f6",
          "#f97316",
        ],
        borderRadius: 6,
      },
    ],
  };

  return (
    <div className="max-w-5xl mx-auto p-4">
      <h2 className="text-3xl font-bold text-center mb-6 text-brand-fg">
        Model Performance Metrics
      </h2>

      {/* 📱 Responsive Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        {Object.entries(metrics).map(([key, value]) => {
          const modelNames: Record<string, string> = {
            ann: "ANN",
            lr: "Logistic Regression",
            rf: "Random Forest",
          };
          return (
            <MetricCard
              key={key}
              title={modelNames[key]}
              value={value}
              best={bestModel === modelNames[key]}
            />
          );
        })}
      </div>

      {/* 📊 Bar Chart */}
      <div className="bg-black/10 backdrop-blur-md border border-white/20 p-4 rounded-xl shadow-md">
        <Bar
          data={chartData}
          options={{
            responsive: true,
            plugins: {
              legend: {
                labels: {
                  color: "var(--color-brand-fg)", // Colors the "Accuracy (%)" legend text
                },
              },
            },
            scales: {
              y: {
                ticks: {
                  color: "var(--color-brand-fg)", // Colors the Y-axis numbers (0, 10, 20...)
                },
                grid: {
                  color: "rgba(247, 244, 243, 0.1)", // Subtle grid lines
                },
              },
              x: {
                ticks: {
                  color: "var(--color-brand-fg)", // Colors the X-axis labels (ANN, Random Forest, etc.)
                },
                grid: {
                  display: false, // Cleaner look by removing vertical grid lines
                },
              },
            },
          }}
        />
      </div>
    </div>
  );
}

/* ---------- Metric Card ---------- */
function MetricCard({ title, value, best }: MetricCardProps): JSX.Element {
  return (
  <div
    className={`relative group p-6 rounded-2xl text-center 
    backdrop-blur-xl border overflow-hidden
    ${
      best
        ? "border-green-400 shadow-[0_0_25px_rgba(34,197,94,0.4)]"
        : "border-white/20"
    }
    bg-black/10`}
  >
    {/* Glow Effect */}
    {best && (
      <div className="absolute inset-0 bg-green-500/10 blur-2xl opacity-40"></div>
    )}

    {/* Title */}
    <h3 className="text-lg font-semibold tracking-wide text-black/90">
      {title}
    </h3>

    {/* Accuracy Value */}
    <p className="text-4xl font-extrabold mt-3 bg-gradient-to-r from-green-600 to-emerald-800 bg-clip-text text-transparent">
      {value.toFixed(2)}%
    </p>

    {/* Best Badge */}
    {best && (
      <div className="absolute top-1 right-1 bg-green-500 text-white text-xs px-3 py-1 rounded-full shadow-lg animate-pulse">
        🏆 Best Model
      </div>
    )}

    {/* Subtle Bottom Line */}
    <div className="mt-4 h-[2px] w-24 mx-auto bg-gradient-to-r from-green-500 to-transparent opacity-40 group-hover:opacity-100 transition"></div>
  </div>
);
}
