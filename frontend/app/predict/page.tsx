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

  const handleChange = (e: any) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(
        "http://localhost:5000/predict",
        {
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
        }
      );

      setResult(response.data);
    } catch (err) {
      alert("❌ Backend error. Check Flask server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex justify-center items-center p-6">
      <div className="bg-white w-full max-w-3xl p-8 rounded-xl shadow">
        <h1 className="text-2xl font-bold text-center mb-6">
          🫀 Heart Disease Risk Prediction
        </h1>

        <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">

          <input type="number" name="age" placeholder="Age"
            className="input" required onChange={handleChange} />

          <select name="sex" className="input" onChange={handleChange}>
            <option>Male</option>
            <option>Female</option>
          </select>

          <select name="chest_pain_type" className="input" onChange={handleChange}>
            <option>Typical Angina</option>
            <option>Atypical Angina</option>
            <option>Non-anginal Pain</option>
            <option>Asymptomatic</option>
          </select>

          <input type="number" name="resting_bp" placeholder="Resting BP"
            className="input" required onChange={handleChange} />

          <input type="number" name="cholesterol" placeholder="Cholesterol"
            className="input" required onChange={handleChange} />

          <select name="fasting_bs" className="input" onChange={handleChange}>
            <option>{'<= 120 mg/dl'}</option>
            <option>{'> 120 mg/dl'}</option>
          </select>

          <select name="resting_ecg" className="input" onChange={handleChange}>
            <option>Normal</option>
            <option>ST-T wave Abnormality</option>
            <option>Left Ventricular Hypertrophy</option>
          </select>

          <input type="number" name="max_hr" placeholder="Max Heart Rate"
            className="input" required onChange={handleChange} />

          <select name="exercise_angina" className="input" onChange={handleChange}>
            <option>No</option>
            <option>Yes</option>
          </select>

          <input type="number" step="0.1" name="oldpeak" placeholder="Oldpeak"
            className="input" required onChange={handleChange} />

          <select name="st_slope" className="input" onChange={handleChange}>
            <option>Upsloping</option>
            <option>Flat</option>
            <option>Downsloping</option>
          </select>

          <button
            type="submit"
            className="col-span-2 bg-red-600 text-white py-2 rounded mt-4"
          >
            {loading ? "Analyzing..." : "Predict Risk"}
          </button>
        </form>

        {result && (
          <div className="mt-6 p-4 border rounded bg-gray-50">
            <h2 className="font-bold mb-2">Prediction Result</h2>
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
