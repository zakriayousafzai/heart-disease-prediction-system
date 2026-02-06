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

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

/* ---------- Types ---------- */
interface MetricsData {
  ann: number;
  rf: number;
  lr: number;
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
          ["", -Infinity]
        );

        const modelNames: Record<string, string> = {
          ann: "ANN",
          rf: "Random Forest",
          lr: "Logistic Regression",
        };

        setBestModel(modelNames[bestKey] || "");
      } catch (err) {
        console.error("Error fetching metrics:", err);
      }
    };

    fetchMetrics();
  }, []);

  if (!metrics) {
    return <p className="text-center mt-10 text-gray-500">Loading metrics...</p>;
  }

  const chartData = {
    labels: ["ANN", "Random Forest", "Logistic Regression"],
    datasets: [
      {
        label: "Accuracy (%)",
        data: [metrics.ann, metrics.rf, metrics.lr],
        backgroundColor: ["#3b82f6", "#22c55e", "#f97316"],
        borderRadius: 6,
      },
    ],
  };

  return (
    <div className="max-w-5xl mx-auto p-4">
      <h2 className="text-3xl font-bold text-center mb-6">
        Model Performance Metrics
      </h2>

      {/* 📱 Responsive Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        {Object.entries(metrics).map(([key, value]) => {
          const modelNames: Record<string, string> = {
            ann: "ANN",
            rf: "Random Forest",
            lr: "Logistic Regression",
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
      <div className="bg-white p-4 rounded-xl shadow-md">
        <Bar data={chartData} options={{ responsive: true }} />
      </div>
    </div>
  );
}

/* ---------- Metric Card ---------- */
function MetricCard({ title, value, best }: MetricCardProps): JSX.Element {
  return (
    <div
      className={`p-5 rounded-xl shadow-md text-center border-2 transition-transform duration-200 hover:scale-105 ${
        best ? "border-green-500 bg-green-50" : "border-gray-200"
      }`}
    >
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="text-2xl font-bold mt-2">{(value).toFixed(2)}%</p>
      {best && (
        <span className="inline-block mt-2 text-green-600 font-semibold">
          🟢 Best Model
        </span>
      )}
    </div>
  );
}
