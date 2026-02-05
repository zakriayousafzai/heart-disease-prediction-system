"use client";

import { useState } from "react";
import axios from "axios";

export default function PredictPage() {
  const [form, setForm] = useState({
    age: "",
    sex: "Male",
    chest_pain_type: "Typical Angina",
    resting_bp: "",
    cholesterol: "",
    fasting_bs: "<= 120 mg/dl",
    resting_ecg: "Normal",
    max_hr: "",
    exercise_angina: "No",
    oldpeak: "",
    st_slope: "Upsloping",
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:5000/predict", {
        age: Number(form.age),
        sex: form.sex,
        chest_pain_type: form.chest_pain_type,
        resting_bp: Number(form.resting_bp),
        cholesterol: Number(form.cholesterol),
        fasting_bs: form.fasting_bs,
        resting_ecg: form.resting_ecg,
        max_hr: Number(form.max_hr),
        exercise_angina: form.exercise_angina,
        oldpeak: Number(form.oldpeak),
        st_slope: form.st_slope,
      });

      setResult(response.data);
    } catch {
      alert("❌ Backend error. Check Flask server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-gray-100 flex justify-center items-center p-4">
      <div className="bg-white w-full max-w-4xl rounded-2xl shadow-lg p-6 md:p-10">
        <h1 className="text-3xl font-bold text-center text-red-600 mb-2">
          🫀 Heart Disease Prediction
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Please fill the form carefully. All fields are required.
        </p>

        <form
          onSubmit={handleSubmit}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {/* Age */}
          <div>
            <label className="block font-medium mb-1">Age (years)</label>
            <input
              type="number"
              name="age"
              className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:ring-2 focus:ring-red-500 focus:outline-none"
              placeholder="e.g. 55"
              required
              onChange={handleChange}
            />
          </div>

          {/* Sex */}
          <div>
            <label className="block font-medium mb-1">Gender</label>
            <select
              name="sex"
              className="w-full rounded-lg border border-gray-300 px-4 py-2 bg-white focus:ring-2 focus:ring-red-500"
              onChange={handleChange}
            >
              <option>Male</option>
              <option>Female</option>
            </select>
          </div>

          {/* Chest Pain */}
          <div>
            <label className="block font-medium mb-1">Chest Pain Type</label>
            <select
              name="chest_pain_type"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>Typical Angina</option>
              <option>Atypical Angina</option>
              <option>Non-anginal Pain</option>
              <option>Asymptomatic</option>
            </select>
          </div>

          {/* Resting BP */}
          <div>
            <label className="block font-medium mb-1">
              Resting Blood Pressure
            </label>
            <input
              type="number"
              name="resting_bp"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 140 mmHg"
              required
              onChange={handleChange}
            />
          </div>

          {/* Cholesterol */}
          <div>
            <label className="block font-medium mb-1">Cholesterol</label>
            <input
              type="number"
              name="cholesterol"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 240 mg/dl"
              required
              onChange={handleChange}
            />
          </div>

          {/* Fasting Blood Sugar */}
          <div>
            <label className="block font-medium mb-1">
              Fasting Blood Sugar
            </label>
            <select
              name="fasting_bs"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>{'<= 120 mg/dl'}</option>
              <option>{'> 120 mg/dl'}</option>
            </select>
          </div>

          {/* ECG */}
          <div>
            <label className="block font-medium mb-1">Resting ECG</label>
            <select
              name="resting_ecg"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>Normal</option>
              <option>ST-T wave Abnormality</option>
              <option>Left Ventricular Hypertrophy</option>
            </select>
          </div>

          {/* Max HR */}
          <div>
            <label className="block font-medium mb-1">
              Maximum Heart Rate
            </label>
            <input
              type="number"
              name="max_hr"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 160"
              required
              onChange={handleChange}
            />
          </div>

          {/* Exercise Angina */}
          <div>
            <label className="block font-medium mb-1">
              Exercise Induced Angina
            </label>
            <select
              name="exercise_angina"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>No</option>
              <option>Yes</option>
            </select>
          </div>

          {/* Oldpeak */}
          <div>
            <label className="block font-medium mb-1">Oldpeak</label>
            <input
              type="number"
              step="0.1"
              name="oldpeak"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              placeholder="e.g. 1.5"
              required
              onChange={handleChange}
            />
          </div>

          {/* ST Slope */}
          <div>
            <label className="block font-medium mb-1">ST Slope</label>
            <select
              name="st_slope"
              className="w-full rounded-lg border border-gray-300 px-4 py-2"
              onChange={handleChange}
            >
              <option>Upsloping</option>
              <option>Flat</option>
              <option>Downsloping</option>
            </select>
          </div>

          {/* Submit */}
          <button
            type="submit"
            className="md:col-span-2 mt-4 bg-red-600 hover:bg-red-700 text-white py-3 rounded-xl text-lg font-semibold transition"
          >
            {loading ? "🔍 Analyzing..." : "❤️ Predict Heart Risk"}
          </button>
        </form>

        {/* Result */}
        {result && (
          <div className="mt-8 bg-green-50 border border-green-200 rounded-xl p-5">
            <h2 className="text-xl font-bold mb-3">📊 Prediction Result</h2>
            <p><b>ANN Risk:</b> {result.ann_prediction.result}</p>
            <p><b>Probability:</b> {result.ann_prediction.probability}%</p>
            <p><b>Random Forest:</b> {result.rf_prediction}</p>
            <p><b>Logistic Regression:</b> {result.lr_prediction}</p>
          </div>
        )}
      </div>
    </div>
  );
}
