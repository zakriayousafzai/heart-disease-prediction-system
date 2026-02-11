"use client";

import { JSX, useEffect, useState } from "react";

/* ---------- Types ---------- */
interface HistoryRecord {
  id: number;
  age: number;
  sex: string;
  chest_pain: string;
  ann_prediction: string;
  ann_probability: number;
  timestamp: string;
}

/* ---------- Component ---------- */
export default function HistoryPage(): JSX.Element {
  const [data, setData] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch("http://localhost:5000/history")
      .then((res) => res.json())
      .then((result: HistoryRecord[]) => {
        setData(result);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching history:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <p className="text-center mt-10">Loading history...</p>;
  }

  return (
    <div className="w-full rounded-2xl shadow-xl p-6 md:p-10 bg-black/10 backdrop-blur-md border border-white/20">
      <h1 className="text-2xl font-bold mb-4 text-brand-fg">
        Prediction History (Last 50 Records)
      </h1>

      <div className="overflow-x-auto bg-gray-500">
        <table className="w-full border border-gray-300 text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="border px-3 py-2">ID</th>
              <th className="border px-3 py-2">Age</th>
              <th className="border px-3 py-2">Sex</th>
              <th className="border px-3 py-2">Chest Pain</th>
              <th className="border px-3 py-2">ANN Result</th>
              <th className="border px-3 py-2">Probability (%)</th>
              <th className="border px-3 py-2">Timestamp</th>
            </tr>
          </thead>

          <tbody>
            {data.map((item) => (
              <tr key={item.id} className="text-center">
                <td className="border px-2 py-1">{item.id}</td>
                <td className="border px-2 py-1">{item.age}</td>
                <td className="border px-2 py-1">{item.sex}</td>
                <td className="border px-2 py-1">{item.chest_pain}</td>
                <td className="border px-2 py-1 font-semibold">
                  {item.ann_prediction}
                </td>
                <td className="border px-2 py-1">
                  {item.ann_probability.toFixed(2)}
                </td>
                <td className="border px-2 py-1">
                  {new Date(item.timestamp).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
